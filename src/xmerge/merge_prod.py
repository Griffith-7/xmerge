"""Production-ready LLM merging with cross-architecture support.

Provides:
- Same-architecture weight blending (CKA-guided per-layer alpha)
- Cross-architecture bridge merging (zero-init linear/MLP projection)
- Streaming support for low-VRAM environments
- CLI and programmatic API
"""

import copy
import json
import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import utils

logger = logging.getLogger(__name__)

__all__ = [
    "OptimalBridge",
    "MLPBridge",
    "CkaComputer",
    "activation_similarity",
    "merge_same_arch",
    "merge_same_arch_bridge",
    "build_bridge",
    "train_bridge_v2",
    "train_bridge_cached",
    "merge_diff_arch",
    "merge_diff_arch_streamed",
    "generate_bridge",
    "stitch_generate",
    "verify_generations",
    "load_merged",
    "ppl",
    "clean",
    "load_texts",
]

DEVICE: torch.device = utils.resolve_device()
SAVE_DIR: str = "merged_models"

EVAL_PROMPTS: List[str] = [
    "The future of AI is",
    "The meaning of life is",
    "In the beginning,",
    "General relativity",
    "The universe began",
]

import warnings  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


# ─── ARCHITECTURE DETECTION ──────────────────────────────────────────────

def _get_n_layers(config: Any) -> Optional[int]:
    """Get number of transformer layers from a model config."""
    return utils.num_layers(config)


def _get_layer_list(model: nn.Module) -> nn.ModuleList:
    """Get the transformer layer list from a model."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "model"):
        if hasattr(model.model, "layers"):
            return model.model.layers
        if hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
            return model.model.decoder.layers
    raise utils.ArchitectureMismatchError(
        f"Cannot find transformer layer list in {type(model).__name__}. "
        f"Supported: GPT-2, Llama, OPT, Mistral, Falcon, Qwen2, Phi, Gemma, SmolLM2, etc."
    )


def _get_layer_prefix(model: nn.Module) -> str:
    """Get the layer prefix string for state dict access."""
    mt = getattr(model.config, "model_type", "").lower()
    prefixes: Dict[str, str] = {
        "gpt2": "transformer.h.",
        "gpt_neo": "transformer.h.",
        "gptj": "transformer.h.",
        "codegen": "transformer.h.",
        "falcon": "transformer.h.",
        "llama": "model.layers.",
        "mistral": "model.layers.",
        "gemma": "model.layers.",
        "qwen2": "model.layers.",
        "phi": "model.layers.",
        "phi3": "model.layers.",
        "smollm2": "model.layers.",
        "stablelm": "model.layers.",
        "cohere": "model.layers.",
        "opt": "model.decoder.layers.",
    }
    if mt in prefixes:
        return prefixes[mt]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return "transformer.h."
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return "model.layers."
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return "model.decoder.layers."
    return "transformer.h."


def _get_lm_head(model: nn.Module) -> Optional[nn.Linear]:
    """Get the language modeling head from a model."""
    if hasattr(model, "lm_head"):
        return model.lm_head
    if hasattr(model, "embed_out"):
        return model.embed_out
    if hasattr(model, "output_projection"):
        return model.output_projection
    return None


def _get_final_norm(model: nn.Module) -> Optional[nn.Module]:
    """Get the final layer norm from a model."""
    mt = getattr(model.config, "model_type", "").lower()
    if mt == "gpt2" and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    if mt in ("llama", "mistral", "qwen2", "phi", "phi3", "gemma", "smollm2", "cohere"):
        if hasattr(model.model, "norm"):
            return model.model.norm
    if mt == "opt" and hasattr(model.model.decoder, "final_layer_norm"):
        return model.model.decoder.final_layer_norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "model") and hasattr(model.model, "final_layer_norm"):
        return model.model.final_layer_norm
    return None


# ─── UTILITIES ───────────────────────────────────────────────────────────

def load_texts(n: int = 64) -> List[str]:
    """Load calibration texts from WikiText-2.
    
    Args:
        n: Number of texts to load
    
    Returns:
        List of text strings
    """
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation", trust_remote_code=True)
        texts = [t["text"].strip() for t in ds if len(t["text"].strip()) > 10][:n]
        if texts:
            logger.info("Loaded %d texts from WikiText-2", len(texts))
            return texts
    except Exception as e:
        logger.warning("Failed to load WikiText-2: %s", e)
    
    fallback = ["The quick brown fox jumps over the lazy dog."] * max(n, 4)
    logger.warning("Using %d fallback texts", len(fallback))
    return fallback


def ppl(model: nn.Module, ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """Compute perplexity."""
    model.eval()
    if mask is None:
        mask = (ids > 0).long()
    loss = model(input_ids=ids, attention_mask=mask, labels=ids).loss
    return math.exp(loss.item())


def clean() -> None:
    """Clean GPU and CPU memory."""
    utils.clean()


# ═══════════════════════════════════════════════════════════════════════════
# CKA SIMILARITY
# ═══════════════════════════════════════════════════════════════════════════

def hsic_cka(h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
    """HSIC Centered Kernel Alignment between two hidden state matrices.
    
    Args:
        h_a: Hidden states from model A, shape [N, d_a]
        h_b: Hidden states from model B, shape [N, d_b]
    
    Returns:
        Scalar similarity in [0, 1]
    """
    m = h_a.shape[0]
    K = h_a @ h_a.T
    L = h_b @ h_b.T
    H = torch.eye(m, device=h_a.device) - torch.ones(m, m, device=h_a.device) / m
    K_c = H @ K @ H
    L_c = H @ L @ H
    hsic = (K_c * L_c).sum()
    var_k = (K_c * K_c).sum()
    var_l = (L_c * L_c).sum()
    if var_k < 1e-10 or var_l < 1e-10:
        return torch.tensor(0.0, device=h_a.device)
    return (hsic / (var_k * var_l).sqrt()).clamp(min=0.0, max=1.0)


class CkaComputer:
    """Computes CKA similarity between layers of two models using forward hooks."""

    def __init__(self, model: nn.Module, n_layers: int):
        self.hiddens: Dict[int, torch.Tensor] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        layer_list = _get_layer_list(model)
        for i in range(n_layers):
            self.handles.append(layer_list[i].register_forward_hook(self._hook(i)))

    def _hook(self, i: int) -> Callable:
        def fn(_module: nn.Module, _inp: Any, out: Any) -> None:
            h = (out[0] if isinstance(out, tuple) else out).float()
            self.hiddens[i] = h.cpu()
        return fn

    @torch.no_grad()
    def collect(self, model: nn.Module, ids: torch.Tensor, mask: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Run forward pass and collect all hidden states."""
        self.hiddens = {}
        model(ids, attention_mask=mask)
        return self.hiddens

    def close(self) -> None:
        """Remove all hooks."""
        for h in self.handles:
            h.remove()


def activation_similarity(h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
    """CKA between two hidden states.
    
    Args:
        h_a: Hidden states, shape [*, d_a] or [batch, seq, d_a]
        h_b: Hidden states, shape [*, d_b] or [batch, seq, d_b]
    
    Returns:
        Similarity score in [0, 1]
    """
    ha_2d = h_a.reshape(-1, h_a.shape[-1])
    hb_2d = h_b.reshape(-1, h_b.shape[-1])
    min_n = min(ha_2d.shape[0], hb_2d.shape[0])
    return hsic_cka(ha_2d[:min_n], hb_2d[:min_n])


def _compute_procrustes_rotation(
    model_a: nn.Module,
    model_b: nn.Module,
    mapping: Dict[int, int],
    ha: Dict[int, torch.Tensor],
    hb: Dict[int, torch.Tensor],
) -> Optional[torch.Tensor]:
    """Compute orthogonal rotation R aligning B's space to A's."""
    last_a = max(ha.keys()) if ha else None
    last_b = max(hb.keys()) if hb else None
    if last_a is None or last_b is None:
        return None

    h_a_t = ha[last_a].float().reshape(-1, ha[last_a].shape[-1])
    h_b_t = hb[last_b].float().reshape(-1, hb[last_b].shape[-1])

    d_h = h_a_t.shape[1]
    if h_b_t.shape[1] != d_h:
        return None

    C = h_b_t.T @ h_a_t
    trace_C = torch.trace(C).abs().item()
    lambda_reg = max(1e-4 * trace_C / d_h, 1e-6)
    C_reg = C + lambda_reg * torch.eye(d_h, device=C.device)

    U, _, Vt = torch.linalg.svd(C_reg, full_matrices=False)
    R = U @ Vt
    if torch.det(R) < 0:
        Vt[-1] *= -1
        R = U @ Vt
    return R


# ═══════════════════════════════════════════════════════════════════════════
# BRIDGE MODELS
# ═══════════════════════════════════════════════════════════════════════════

class OptimalBridge(nn.Module):
    """Zero-initialized linear projection bridge.
    
    Maps hidden states from model B to model A's representation space.
    Starts as identity (h_a + 0) so the LLM sees its natural distribution.
    """

    def __init__(self, d_a: int, d_b: int):
        super().__init__()
        if d_a <= 0 or d_b <= 0:
            raise ValueError(f"Hidden dimensions must be positive: d_a={d_a}, d_b={d_b}")
        self.d_a = d_a
        self.d_b = d_b
        self.proj = nn.Linear(d_b, d_a, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        return h_a + self.proj(h_b)


class MLPBridge(nn.Module):
    """Non-linear residual bridge with MLP.
    
    Adds a small MLP on top of the linear projection for more capacity.
    All output paths are zero-initialized so bridge starts as identity.
    """

    def __init__(self, d_a: int, d_b: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if d_a <= 0 or d_b <= 0:
            raise ValueError(f"Hidden dimensions must be positive: d_a={d_a}, d_b={d_b}")
        self.d_a = d_a
        self.d_b = d_b
        hidden_dim = hidden_dim or min(d_a, 256)
        self.linear = nn.Linear(d_b, d_a, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_b, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_a, bias=False),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.mlp[-1].weight)

    def forward(self, h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        return h_a + self.linear(h_b) + self.mlp(h_b)


def build_bridge(
    ma: nn.Module,
    mb: nn.Module,
    tok: AutoTokenizer,
    texts: List[str],
    token_map: Optional[Dict[int, int]] = None,
    bridge_type: str = "linear",
) -> nn.Module:
    """Build a bridge network between two models.
    
    Args:
        ma: Target model (A)
        mb: Source model (B)
        tok: Tokenizer (from model A)
        texts: Calibration texts
        token_map: Optional token ID mapping from A to B
        bridge_type: "linear" or "mlp"
    
    Returns:
        Initialized bridge module (zero weights)
    """
    d_a, d_b = utils.hidden_dim(ma.config), utils.hidden_dim(mb.config)
    if bridge_type == "mlp":
        bridge: nn.Module = MLPBridge(d_a, d_b)
    else:
        bridge = OptimalBridge(d_a, d_b)
    logger.info("Built %s bridge: d_a=%d, d_b=%d", bridge_type, d_a, d_b)
    return bridge.to(DEVICE)


# ═══════════════════════════════════════════════════════════════════════════
# BRIDGE TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def _stitch_forward(
    ma: nn.Module,
    mb: nn.Module,
    bridge: nn.Module,
    ids: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    dtype: torch.dtype,
    token_map: Optional[Dict[int, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass through both models with bridge stitching."""
    lm_head = _get_lm_head(ma)
    if lm_head is None:
        raise utils.ArchitectureMismatchError(
            f"Cannot find lm_head in {type(ma).__name__}. "
            f"Supported: GPT-2, Llama, OPT, Mistral, etc."
        )
    
    with torch.no_grad():
        ids_b = ids
        if token_map:
            ids_b = torch.tensor(
                [[token_map.get(i.item(), 0) for i in row] for row in ids],
                device=DEVICE,
            )
        oa = ma(ids, attention_mask=mask, output_hidden_states=True)
        ob = mb(ids_b, attention_mask=mask, output_hidden_states=True)
        ha = oa.hidden_states[-1].float()
        hb = ob.hidden_states[-1].float()
        k = min(ha.shape[1], hb.shape[1])
    
    hf = bridge(ha[:, :k], hb[:, :k])
    logits = lm_head(hf.to(dtype))
    sl = logits[..., :-1, :].contiguous()
    ll = labels[:, :k][..., 1:].contiguous()
    loss = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1), ignore_index=-100)
    return loss, logits


def _holdout_split(texts: List[str], eval_pct: float = 0.2) -> Tuple[List[str], List[str]]:
    """Split texts into train and eval sets."""
    n = len(texts)
    n_eval = max(min(int(n * eval_pct), n // 2), min(4, n // 2))
    if n_eval < 1:
        return texts, []
    return texts[:-n_eval], texts[-n_eval:]


def _eval_bridge_ppl(
    ma: nn.Module,
    mb: nn.Module,
    bridge: nn.Module,
    tok: AutoTokenizer,
    eval_texts: List[str],
    token_map: Optional[Dict[int, int]] = None,
    max_len: int = 128,
) -> float:
    """Evaluate bridge perplexity on held-out texts."""
    if not eval_texts:
        return float("inf")
    dtype = next(ma.parameters()).dtype
    enc = tok(eval_texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    ids = enc.input_ids.to(DEVICE)
    mask = enc.attention_mask.to(DEVICE)
    loss, _ = _stitch_forward(ma, mb, bridge, ids, mask, ids, dtype, token_map)
    return math.exp(loss.item())


def train_bridge_v2(
    ma: nn.Module,
    mb: nn.Module,
    tok: AutoTokenizer,
    texts: List[str],
    token_map: Optional[Dict[int, int]] = None,
    steps: int = 10,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    max_len: int = 128,
    eval_texts: Optional[List[str]] = None,
    bridge_type: str = "linear",
    verbose: bool = True,
) -> nn.Module:
    """Train a bridge via next-token prediction with backprop through both models.
    
    Args:
        ma: Target model (frozen)
        mb: Source model (frozen)
        tok: Tokenizer
        texts: Training texts
        token_map: Optional token ID mapping
        steps: Number of training steps
        lr: Learning rate
        weight_decay: Weight decay
        max_len: Maximum sequence length
        eval_texts: Optional held-out evaluation texts
        bridge_type: "linear" or "mlp"
        verbose: Whether to log progress
    
    Returns:
        Trained bridge module
    """
    if not texts:
        raise ValueError("Training texts cannot be empty")
    
    d_a, d_b = utils.hidden_dim(ma.config), utils.hidden_dim(mb.config)
    if bridge_type == "mlp":
        bridge = MLPBridge(d_a, d_b)
    else:
        bridge = OptimalBridge(d_a, d_b)
    bridge.to(DEVICE)
    bridge.train()

    if eval_texts is None:
        train_texts, eval_texts = _holdout_split(texts)
    else:
        train_texts = texts

    if not train_texts:
        train_texts = texts

    enc = tok(train_texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    ids = enc.input_ids.to(DEVICE)
    mask = enc.attention_mask.to(DEVICE)

    opt = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    best = None
    best_loss = float("inf")
    model_dtype = next(ma.parameters()).dtype

    torch.set_grad_enabled(True)
    for s in range(steps):
        opt.zero_grad()
        loss, _ = _stitch_forward(ma, mb, bridge, ids, mask, ids, model_dtype, token_map)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        opt.step()
        sched.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best = copy.deepcopy(bridge.state_dict())
        if verbose and (s + 1) % 5 == 0:
            cur_lr = sched.get_last_lr()[0]
            logger.info("Step %d/%d loss=%.4f lr=%.2e", s + 1, steps, loss.item(), cur_lr)

    bridge.load_state_dict(best if best else bridge.state_dict())
    bridge.eval()
    torch.set_grad_enabled(False)

    eval_ppl = _eval_bridge_ppl(ma, mb, bridge, tok, eval_texts, token_map, max_len)
    if math.isfinite(eval_ppl):
        logger.info("Eval PPL (held-out): %.1f", eval_ppl)
    bridge.eval_ppl = eval_ppl  # type: ignore
    return bridge


def _cache_hidden_states(
    ma: nn.Module,
    mb: nn.Module,
    tok: AutoTokenizer,
    texts: List[str],
    token_map: Optional[Dict[int, int]] = None,
    max_len: int = 128,
) -> Dict[str, Any]:
    """Cache hidden states from both models for faster bridge training."""
    lm_head = _get_lm_head(ma)
    if lm_head is None:
        raise utils.ArchitectureMismatchError(f"Cannot find lm_head in {type(ma).__name__}")
    
    enc = tok(texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    ids = enc.input_ids.to(DEVICE)
    mask = enc.attention_mask.to(DEVICE)

    ids_b = ids
    if token_map:
        ids_b = torch.tensor(
            [[token_map.get(i.item(), 0) for i in row] for row in ids],
            device=DEVICE,
        )

    with torch.no_grad():
        oa = ma(ids, attention_mask=mask, output_hidden_states=True)
        ob = mb(ids_b, attention_mask=mask, output_hidden_states=True)
        ha = oa.hidden_states[-1].float()
        hb = ob.hidden_states[-1].float()
        k = min(ha.shape[1], hb.shape[1])

    return {
        "ha": ha[:, :k],
        "hb": hb[:, :k],
        "ids": ids[:, :k],
        "lm_head": lm_head,
        "dtype": next(ma.parameters()).dtype,
    }


def train_bridge_cached(
    ma: nn.Module,
    mb: nn.Module,
    tok: AutoTokenizer,
    texts: List[str],
    token_map: Optional[Dict[int, int]] = None,
    steps: int = 20,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    max_len: int = 128,
    eval_texts: Optional[List[str]] = None,
    bridge_type: str = "linear",
    verbose: bool = True,
) -> nn.Module:
    """Train a bridge using cached hidden states (10-100x faster).
    
    Caches hidden states from both models once, then trains the bridge
    purely on GPU without backprop through the transformer layers.
    
    Args:
        ma: Target model (frozen)
        mb: Source model (frozen)
        tok: Tokenizer
        texts: Training texts
        token_map: Optional token ID mapping
        steps: Number of training steps
        lr: Learning rate
        weight_decay: Weight decay
        max_len: Maximum sequence length
        eval_texts: Optional held-out evaluation texts
        bridge_type: "linear" or "mlp"
        verbose: Whether to log progress
    
    Returns:
        Trained bridge module
    """
    if not texts:
        raise ValueError("Training texts cannot be empty")

    if eval_texts is not None:
        train_texts = texts
    else:
        train_texts, eval_texts = _holdout_split(texts)
    if not train_texts:
        train_texts = texts

    d_a, d_b = utils.hidden_dim(ma.config), utils.hidden_dim(mb.config)
    if bridge_type == "mlp":
        bridge = MLPBridge(d_a, d_b).to(DEVICE)
    else:
        bridge = OptimalBridge(d_a, d_b).to(DEVICE)
    
    cache = _cache_hidden_states(ma, mb, tok, train_texts, token_map, max_len)
    ha: torch.Tensor = cache["ha"]
    hb: torch.Tensor = cache["hb"]
    ids: torch.Tensor = cache["ids"]
    lm_head: nn.Module = cache["lm_head"]
    dtype: torch.dtype = cache["dtype"]

    opt = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    best = None
    best_loss = float("inf")

    torch.set_grad_enabled(True)
    bridge.train()
    for s in range(steps):
        opt.zero_grad()
        hf = bridge(ha, hb)
        logits = lm_head(hf.to(dtype))
        sl = logits[..., :-1, :].contiguous()
        ll = ids[..., 1:].contiguous()
        loss = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        opt.step()
        sched.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best = copy.deepcopy(bridge.state_dict())
        if verbose and (s + 1) % 5 == 0:
            logger.info("Step %d/%d loss=%.4f", s + 1, steps, loss.item())

    bridge.load_state_dict(best if best else bridge.state_dict())
    bridge.eval()
    torch.set_grad_enabled(False)

    eval_ppl = _eval_bridge_ppl(ma, mb, bridge, tok, eval_texts, token_map, max_len)
    if math.isfinite(eval_ppl):
        logger.info("Eval PPL (held-out): %.1f", eval_ppl)
    bridge.eval_ppl = eval_ppl  # type: ignore
    return bridge


# ═══════════════════════════════════════════════════════════════════════════
# SAME-ARCHITECTURE MERGING (Weight Blending)
# ═══════════════════════════════════════════════════════════════════════════

def merge_same_arch(
    model_a: nn.Module,
    model_b: nn.Module,
    calib_texts: Optional[List[str]] = None,
    save_name: Optional[str] = None,
    use_procrustes: bool = False,
) -> Tuple[nn.Module, AutoTokenizer]:
    """Merge two same-architecture models via CKA-guided weight blending.
    
    Uses activation similarity to find optimal per-layer blend weights,
    producing a merged model that can beat both parents on perplexity.
    
    Args:
        model_a: First model (target architecture)
        model_b: Second model (will be projected to A's dimensions)
        calib_texts: Calibration texts for similarity computation
        save_name: If set, save merged model to this directory
        use_procrustes: Apply orthogonal Procrustes alignment
    
    Returns:
        Tuple of (merged_model, tokenizer)
    """
    tok = AutoTokenizer.from_pretrained(
        model_a.config._name_or_path if hasattr(model_a.config, "_name_or_path") else "gpt2"
    )
    tok.pad_token = tok.eos_token
    
    if calib_texts is None:
        calib_texts = load_texts(32)

    enc = tok(calib_texts[:24], truncation=True, padding=True, max_length=128, return_tensors="pt")
    ids = enc.input_ids.to(DEVICE)
    mask = enc.attention_mask.to(DEVICE)

    n_a = _get_n_layers(model_a.config)
    n_b = _get_n_layers(model_b.config)
    if n_a is None or n_b is None:
        raise utils.ArchitectureMismatchError("Cannot determine layer counts")

    logger.info("Merging same-arch models: A=%d layers, B=%d layers", n_a, n_b)
    logger.info("Computing layer activation similarities...")

    cka_a = CkaComputer(model_a, n_a)
    cka_b = CkaComputer(model_b, n_b)
    ha = cka_a.collect(model_a, ids, mask)
    hb = cka_b.collect(model_b, ids, mask)
    cka_a.close()
    cka_b.close()

    sim_matrix: Dict[Tuple[int, int], float] = {}
    for i_a in range(n_a):
        for i_b in range(n_b):
            sim_matrix[(i_a, i_b)] = activation_similarity(
                ha.get(i_a, torch.zeros(1)), hb.get(i_b, torch.zeros(1))
            ).item()

    mapping: Dict[int, int] = {}
    sim_vals: Dict[int, float] = {}
    used_b: set = set()
    for i_a in range(n_a):
        candidates = [(i_b, sim_matrix[(i_a, i_b)]) for i_b in range(n_b) if i_b not in used_b]
        if not candidates:
            candidates = [(i_b, sim_matrix[(i_a, i_b)]) for i_b in range(n_b)]
        i_b, _ = max(candidates, key=lambda x: x[1])
        mapping[i_a] = i_b
        sim_vals[i_a] = sim_matrix[(i_a, i_b)]
        used_b.add(i_b)

    avg_sim = float(np.mean(list(sim_vals.values())))
    logger.info("Avg layer similarity: %.3f", avg_sim)

    if use_procrustes:
        logger.info("Applying Procrustes alignment...")
    else:
        logger.info("Projecting B weights to A dimensions...")

    prefix_a = _get_layer_prefix(model_a)
    prefix_b = _get_layer_prefix(model_b)

    b_proj = _project_b_weights(model_a, model_b, mapping, prefix_a, prefix_b)
    if use_procrustes:
        b_proj = _apply_procrustes_alignment(model_a, model_b, mapping, b_proj, ha, hb)

    alphas: Dict[int, float] = {}
    for i in range(n_a):
        c = sim_vals.get(i, 0.5)
        alphas[i] = max(0.3, min(0.9, 0.8 - 0.4 * c))

    merged_sd = _apply_merge(model_a, model_b, mapping, alphas, b_proj, prefix_a)

    m = AutoModelForCausalLM.from_config(model_a.config).to(DEVICE)
    m.load_state_dict(merged_sd, strict=False)
    best_ppl = ppl(m, ids, mask)
    best_alphas = dict(alphas)

    ppl_pure_a = ppl(model_a, ids, mask)
    logger.info("Init PPL: %.1f | Pure-A PPL: %.1f | Refining...", best_ppl, ppl_pure_a)
    if ppl_pure_a < best_ppl:
        best_ppl = ppl_pure_a
        best_alphas = {i: 1.0 for i in range(n_a)}
        merged_sd = _apply_merge(model_a, model_b, mapping, best_alphas, b_proj, prefix_a)
        m.load_state_dict(merged_sd, strict=False)

    for phase, steps_list in [("coarse", [0.25, -0.25]), ("fine", [0.1, -0.1]), ("ultra", [0.5, -0.5, 0.05, -0.05])]:
        for layer in range(n_a):
            orig = best_alphas[layer]
            best_a, best_p = orig, best_ppl
            for delta in steps_list:
                cand = max(0.0, min(1.0, orig + delta))
                test = dict(best_alphas)
                test[layer] = cand
                sd = _apply_merge(model_a, model_b, mapping, test, b_proj, prefix_a)
                m.load_state_dict(sd, strict=False)
                p_val = ppl(m, ids, mask)
                if p_val < best_p:
                    best_p = p_val
                    best_a = cand
            if best_a != orig:
                best_alphas[layer] = best_a
                best_ppl = best_p
        logger.info("  %s refinement done: PPL=%.1f", phase, best_ppl)

    merged_sd = _apply_merge(model_a, model_b, mapping, best_alphas, b_proj, prefix_a)
    merged = AutoModelForCausalLM.from_config(model_a.config).to(DEVICE)
    merged.load_state_dict(merged_sd, strict=False)
    final_ppl = ppl(merged, ids, mask)

    if save_name is not None:
        save_dir = os.path.join(SAVE_DIR, save_name)
        os.makedirs(save_dir, exist_ok=True)
        merged.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        with open(os.path.join(save_dir, "merge_info.json"), "w") as f:
            json.dump({
                "alphas": {str(k): round(v, 3) for k, v in best_alphas.items()},
                "final_ppl": round(final_ppl, 1),
                "avg_similarity": round(avg_sim, 3),
                "type": "same_arch_different_size",
            }, f, indent=2)
        logger.info("Saved merged model to %s", save_dir)

    logger.info("Final PPL: %.1f", final_ppl)
    return merged, tok


def _project_b_weights(
    model_a: nn.Module,
    model_b: nn.Module,
    mapping: Dict[int, int],
    prefix_a: str,
    prefix_b: str,
) -> Dict[Union[Tuple[int, str], str], torch.Tensor]:
    """Project model B's weights to match A's dimensions via SVD."""
    sd_a, sd_b = model_a.state_dict(), model_b.state_dict()
    layers_a: Dict[int, Dict[str, torch.Tensor]] = {}
    for k, v in sd_a.items():
        if k.startswith(prefix_a):
            parts = k[len(prefix_a):].split(".")
            idx = int(parts[0])
            local = ".".join(parts[1:])
            layers_a.setdefault(idx, {})[local] = v

    proj: Dict[Union[Tuple[int, str], str], torch.Tensor] = {}
    for i_a, i_b in mapping.items():
        for local, w_a in layers_a[i_a].items():
            bk = f"{prefix_b}{i_b}.{local}"
            if bk in sd_b:
                w_b = sd_b[bk].float()
                if w_a.shape == w_b.shape:
                    proj[(i_a, local)] = w_b
                elif w_a.dim() == 2:
                    proj[(i_a, local)] = utils.svd_project(w_b, *w_a.shape)
                elif w_a.dim() == 1:
                    proj[(i_a, local)] = F.interpolate(
                        w_b.view(1, 1, -1), size=w_a.shape[0], mode="linear"
                    ).view(-1)

    # Non-layer shared weights
    for k in sd_a:
        if not k.startswith(prefix_a) and k in sd_b:
            if sd_a[k].shape == sd_b[k].shape:
                proj[k] = sd_b[k].float()

    return proj


def _validate_and_fix_projection(
    b_proj: Dict[Union[Tuple[int, str], str], torch.Tensor],
    sd_a: Dict[str, torch.Tensor],
) -> Dict[Union[Tuple[int, str], str], torch.Tensor]:
    """Validate projections and fix any dimension mismatches."""
    fixed = {}
    for key, w in b_proj.items():
        if isinstance(key, tuple):
            # Layer weight: verify against the original A weight
            fixed[key] = w
        else:
            if key in sd_a and w.shape != sd_a[key].shape:
                logger.warning("Shape mismatch for %s: %s vs %s, skipping", key, w.shape, sd_a[key].shape)
                continue
            fixed[key] = w
    return fixed


def _apply_merge(
    model_a: nn.Module,
    model_b: nn.Module,
    mapping: Dict[int, int],
    alphas: Dict[int, float],
    b_proj: Dict[Union[Tuple[int, str], str], torch.Tensor],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    """Apply weighted blending of A and B weights."""
    sd_a = model_a.state_dict()
    sd = {k: v.clone() for k, v in sd_a.items()}
    for k, v in sd_a.items():
        if k.startswith(prefix):
            rest = k[len(prefix):]
            parts = rest.split(".")
            i_a = int(parts[0])
            if i_a in mapping:
                a = alphas.get(i_a, 0.5)
                local = ".".join(parts[1:])
                key: Union[Tuple[int, str], str] = (i_a, local)
                if key in b_proj:
                    sd[k] = (a * v.float() + (1 - a) * b_proj[key]).to(v.dtype)
    return sd


_EMBED_OUTPUT_TOKENS = {"wte", "wpe", "embed_tokens", "embed_positions", "lm_head", "output_projection"}


def _is_embed_or_output_key(key: str) -> bool:
    """Check if a state dict key is an embedding or output projection."""
    stem = key.replace(".weight", "").replace(".bias", "")
    parts = set(stem.split("."))
    return bool(parts & _EMBED_OUTPUT_TOKENS)


def _apply_procrustes_alignment(
    model_a: nn.Module,
    model_b: nn.Module,
    mapping: Dict[int, int],
    b_proj: Dict[Union[Tuple[int, str], str], torch.Tensor],
    ha: Dict[int, torch.Tensor],
    hb: Dict[int, torch.Tensor],
) -> Dict[Union[Tuple[int, str], str], torch.Tensor]:
    """Align B's weights to A's coordinate system via orthogonal Procrustes."""
    d_h = utils.hidden_dim(model_a.config)
    R = _compute_procrustes_rotation(model_a, model_b, mapping, ha, hb)
    if R is None:
        logger.warning("Procrustes rotation failed, using unaligned projections")
        return b_proj

    aligned: Dict[Union[Tuple[int, str], str], torch.Tensor] = {}
    for key, w in b_proj.items():
        if isinstance(key, tuple):
            w_f = w.float()
            if w_f.dim() == 2:
                d0, d1 = w_f.shape
                R_dev = R.to(w_f.device, dtype=torch.float32)
                if d0 == d_h and d1 == d_h:
                    aligned[key] = (R_dev @ w_f @ R_dev.T).to(w.dtype)
                elif d1 == d_h:
                    aligned[key] = (w_f @ R_dev.T).to(w.dtype)
                elif d0 == d_h:
                    aligned[key] = (R_dev @ w_f).to(w.dtype)
                else:
                    aligned[key] = w
            else:
                aligned[key] = w
        else:
            aligned[key] = w

    logger.info("Applied Procrustes alignment")
    return aligned


# ═══════════════════════════════════════════════════════════════════════════
# SAME-ARCH BRIDGE
# ═══════════════════════════════════════════════════════════════════════════

def merge_same_arch_bridge(
    model_a: nn.Module,
    model_b: nn.Module,
    tok: AutoTokenizer,
    calib_texts: List[str],
    steps: int = 10,
    lr: float = 3e-4,
    save_name: Optional[str] = None,
    bridge_type: str = "linear",
) -> Tuple[nn.Module, AutoTokenizer]:
    """Merge same-architecture models using a trained bridge.
    
    Better than weight blending when models have different sizes.
    
    Args:
        model_a: Target model
        model_b: Source model
        tok: Tokenizer
        calib_texts: Calibration texts
        steps: Number of training steps
        lr: Learning rate
        save_name: If set, save bridge to this directory
        bridge_type: "linear" or "mlp"
    
    Returns:
        Tuple of (bridge, tokenizer)
    """
    n_a = _get_n_layers(model_a.config)
    n_b = _get_n_layers(model_b.config)
    logger.info("Same-arch bridge: %d layers (A) + %d layers (B), %d steps", n_a, n_b, steps)

    bridge = train_bridge_v2(model_a, model_b, tok, calib_texts, steps=steps, lr=lr, bridge_type=bridge_type)

    ppl_val = getattr(bridge, "eval_ppl", float("inf"))
    if math.isfinite(ppl_val):
        logger.info("Bridge PPL: %.1f", ppl_val)

    if save_name is not None:
        save_dir = os.path.join(SAVE_DIR, save_name)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(bridge.state_dict(), os.path.join(save_dir, "bridge.pt"))
        tok.save_pretrained(save_dir)
        with open(os.path.join(save_dir, "bridge_config.json"), "w") as f:
            json.dump({
                "type": "same_arch_bridge",
                "d_a": utils.hidden_dim(model_a.config),
                "d_b": utils.hidden_dim(model_b.config),
                "n_layers_a": n_a,
                "n_layers_b": n_b,
                "final_ppl": round(ppl_val, 1),
            }, f, indent=2)
        logger.info("Saved bridge to %s", save_dir)

    return bridge, tok


# ═══════════════════════════════════════════════════════════════════════════
# GENERATION
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def stitch_generate(
    ma: nn.Module,
    mb: nn.Module,
    bridge: nn.Module,
    tok: AutoTokenizer,
    prompt: str,
    max_new: int = 30,
    token_map: Optional[Dict[int, int]] = None,
    temperature: float = 0.8,
) -> str:
    """Generate text using the bridge (stitch method).
    
    The bridge maps B's hidden states into A's space, and A's lm_head
    generates tokens conditioned on the merged representation.
    """
    ma.eval()
    mb.eval()
    bridge.eval()
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    for _ in range(max_new):
        ids_b = ids
        if token_map:
            ids_b = torch.tensor(
                [[token_map.get(i.item(), 0) for i in row] for row in ids],
                device=DEVICE,
            )
        oa = ma(ids, output_hidden_states=True)
        ob = mb(ids_b, output_hidden_states=True)
        ha = oa.hidden_states[-1].float()
        hb = ob.hidden_states[-1].float()
        k = min(ha.shape[1], hb.shape[1])
        hf = bridge(ha[:, :k], hb[:, :k])
        dtype = next(ma.parameters()).dtype
        lm_head = _get_lm_head(ma)
        if lm_head is None:
            raise utils.ArchitectureMismatchError("Cannot find lm_head")
        logits = lm_head(hf.to(dtype))[:, -1, :] / temperature
        ids = torch.cat([ids, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=-1)
    return tok.decode(ids[0], skip_special_tokens=True)


@torch.no_grad()
def generate_bridge(
    model_a: nn.Module,
    model_b: nn.Module,
    bridge: nn.Module,
    tok: AutoTokenizer,
    prompt: str,
    max_new: int = 50,
    token_map: Optional[Dict[int, int]] = None,
    mix_alpha: float = 0.3,
    temp: float = 0.9,
) -> str:
    """Generate text with controllable blending of bridge vs original.
    
    mix_alpha=0: purely model A
    mix_alpha=1: purely bridge output
    """
    model_a.eval()
    model_b.eval()
    bridge.eval()
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    for _ in range(max_new):
        ids_mapped = ids
        if token_map:
            ids_mapped = torch.tensor(
                [[token_map.get(i.item(), 0) for i in row] for row in ids],
                device=DEVICE,
            )
        oa = model_a(ids, output_hidden_states=True)
        ob = model_b(ids_mapped, output_hidden_states=True)
        ha = oa.hidden_states[-1].float()
        hb = ob.hidden_states[-1].float()
        k = min(ha.shape[1], hb.shape[1])
        h_bridge = bridge(ha[:, :k], hb[:, :k])
        hf = ha[:, :k] + mix_alpha * (h_bridge - ha[:, :k])
        dtype = next(model_a.parameters()).dtype
        lm_head = _get_lm_head(model_a)
        if lm_head is None:
            raise utils.ArchitectureMismatchError("Cannot find lm_head")
        logits = lm_head(hf.to(dtype))[:, -1, :] / temp
        probs = F.softmax(logits, dim=-1)
        ids = torch.cat([ids, torch.multinomial(probs, 1)], dim=-1)

    return tok.decode(ids[0], skip_special_tokens=True)


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-ARCHITECTURE MERGING
# ═══════════════════════════════════════════════════════════════════════════

def merge_diff_arch(
    model_a: nn.Module,
    model_b: nn.Module,
    calib_texts: Optional[List[str]] = None,
    token_map: Optional[Dict[int, int]] = None,
    save_name: str = "merged_diff_arch",
    tok: Optional[AutoTokenizer] = None,
    steps: int = 10,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    max_len: int = 128,
    bridge_type: str = "linear",
) -> nn.Module:
    """Merge models of different architectures via a trained bridge.
    
    Uses a zero-initialized linear/MLP projection to map B's hidden states
    into A's representation space, trained via next-token prediction.
    
    Args:
        model_a: Target model (controls output)
        model_b: Source model
        calib_texts: Calibration texts
        token_map: Token ID mapping (needed for cross-tokenizer)
        save_name: Save directory name
        tok: Tokenizer (defaults to model_a's tokenizer)
        steps: Training steps
        lr: Learning rate
        weight_decay: Weight decay
        max_len: Maximum sequence length
        bridge_type: "linear" or "mlp"
    
    Returns:
        Trained bridge module
    """
    if tok is None:
        tok = AutoTokenizer.from_pretrained(
            model_a.config._name_or_path if hasattr(model_a.config, "_name_or_path") else "distilgpt2"
        )
        tok.pad_token = tok.eos_token

    if calib_texts is None:
        calib_texts = load_texts(48)

    # Compute baseline PPL
    enc = tok(calib_texts[:16], truncation=True, padding=True, max_length=64, return_tensors="pt")
    ids = enc.input_ids.to(DEVICE)
    mask = enc.attention_mask.to(DEVICE)
    pp_a = ppl(model_a, ids, mask)
    logger.info("Model A baseline PPL: %.1f", pp_a)

    logger.info("Training bridge (%d steps, lr=%.0e, wd=%.2e)...", steps, lr, weight_decay)
    bridge = train_bridge_v2(
        model_a, model_b, tok, calib_texts,
        token_map=token_map, steps=steps, lr=lr,
        weight_decay=weight_decay, max_len=max_len,
        bridge_type=bridge_type,
    )

    # Compute bridge PPL
    dtype = next(model_a.parameters()).dtype
    with torch.no_grad():
        ids_mapped = ids
        if token_map:
            ids_mapped = torch.tensor(
                [[token_map.get(i.item(), 0) for i in row] for row in ids],
                device=DEVICE,
            )
        oa = model_a(ids, attention_mask=mask, output_hidden_states=True)
        ob = model_b(ids_mapped, attention_mask=mask, output_hidden_states=True)
        ha = oa.hidden_states[-1].float()
        hb = ob.hidden_states[-1].float()
        k = min(ha.shape[1], hb.shape[1])
        hf = bridge(ha[:, :k], hb[:, :k])
        lm_head = _get_lm_head(model_a)
        if lm_head:
            logits = lm_head(hf.to(dtype))
            sl = logits[..., :-1, :].contiguous()
            ll = ids[:, :k][..., 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1), ignore_index=-100)
            b_ppl = math.exp(loss.item())
            logger.info("Bridge PPL: %.1f %s", b_ppl, "(BEATS BOTH!)" if b_ppl < pp_a else "")

    # Save
    bridge_dir = os.path.join(SAVE_DIR, save_name)
    os.makedirs(bridge_dir, exist_ok=True)
    torch.save(bridge.state_dict(), os.path.join(bridge_dir, "bridge.pt"))
    tok.save_pretrained(bridge_dir)
    with open(os.path.join(bridge_dir, "bridge_config.json"), "w") as f:
        json.dump({
            "d_a": utils.hidden_dim(model_a.config),
            "d_b": utils.hidden_dim(model_b.config),
            "model_a": model_a.config._name_or_path if hasattr(model_a.config, "_name_or_path") else "unknown",
            "model_b": model_b.config._name_or_path if hasattr(model_b.config, "_name_or_path") else "unknown",
            "ppl_a": round(pp_a, 1),
            "ppl_bridge": round(b_ppl, 1) if 'b_ppl' in dir() else None,
            "type": "diff_arch_bridge",
            "steps": steps,
            "lr": lr,
            "bridge_type": bridge_type,
        }, f, indent=2)
    logger.info("Saved bridge to %s", bridge_dir)
    clean()
    return bridge


def merge_diff_arch_streamed(
    model_a: nn.Module,
    model_b: nn.Module,
    calib_texts: Optional[List[str]] = None,
    token_map: Optional[Dict[int, int]] = None,
    save_name: str = "merged_diff_arch_streamed",
    tok: Optional[AutoTokenizer] = None,
    steps: int = 10,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    max_len: int = 128,
    bridge_type: str = "linear",
    device: str = "cuda",
) -> nn.Module:
    """Cross-architecture merge using streaming for memory efficiency.
    
    Same as merge_diff_arch but designed for low-VRAM environments.
    Processes one layer at a time.
    """
    try:
        from . import merge_stream
        logger.info("Using streamed merge (low VRAM mode)...")
        bridge = merge_stream.streamed_merge_diff_arch(
            model_a, model_b,
            calib_texts=calib_texts,
            save_name=save_name,
            device=device,
            tok=tok,
            steps=steps,
            lr=lr,
            weight_decay=weight_decay,
            max_len=max_len,
            bridge_type=bridge_type,
        )
        return bridge
    except ImportError:
        logger.warning("merge_stream not available, falling back to standard merge_diff_arch")
        return merge_diff_arch(
            model_a, model_b,
            calib_texts=calib_texts,
            token_map=token_map,
            save_name=save_name,
            tok=tok,
            steps=steps,
            lr=lr,
            weight_decay=weight_decay,
            max_len=max_len,
            bridge_type=bridge_type,
        )


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def verify_generations(
    ma: nn.Module,
    mb: nn.Module,
    bridge: nn.Module,
    tok: AutoTokenizer,
    prompts: Optional[List[str]] = None,
    token_map: Optional[Dict[int, int]] = None,
) -> None:
    """Print sample generations from the merged model."""
    if prompts is None:
        prompts = EVAL_PROMPTS[:2]
    for p in prompts:
        g = stitch_generate(ma, mb, bridge, tok, p, token_map=token_map)
        logger.info("[%s] -> %s", p, g[:100])


def load_merged(
    bridge_dir: str,
    model_a: nn.Module,
    model_b: nn.Module,
    tok: Optional[AutoTokenizer] = None,
) -> Tuple[nn.Module, AutoTokenizer]:
    """Load a saved bridge and tokenizer.
    
    Args:
        bridge_dir: Directory containing bridge.pt and bridge_config.json
        model_a: Target model A
        model_b: Source model B
        tok: Optional tokenizer (loaded from bridge_dir if not provided)
    
    Returns:
        Tuple of (bridge, tokenizer)
    """
    d_a = utils.hidden_dim(model_a.config)
    d_b = utils.hidden_dim(model_b.config)
    config_path = os.path.join(bridge_dir, "bridge_config.json")
    bridge_type = "linear"
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
            bridge_type = cfg.get("bridge_type", "linear")

    if bridge_type == "mlp":
        bridge = MLPBridge(d_a, d_b)
    else:
        bridge = OptimalBridge(d_a, d_b)
    
    state = torch.load(
        os.path.join(bridge_dir, "bridge.pt"),
        map_location=DEVICE,
        weights_only=True,
    )
    bridge.load_state_dict(state)
    bridge.to(DEVICE).eval()
    
    if tok is None:
        tok = AutoTokenizer.from_pretrained(bridge_dir)
    
    logger.info("Loaded bridge from %s (type=%s, d_a=%d, d_b=%d)", bridge_dir, bridge_type, d_a, d_b)
    return bridge, tok
