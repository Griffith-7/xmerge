"""Shared utilities for all merge approaches."""

import gc
import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

__all__ = [
    "svd_project",
    "svd_project_activation_aware",
    "proportional_map",
    "hidden_dim",
    "num_layers",
    "compute_ppl",
    "generate_text",
    "load_calibration",
    "build_token_map",
    "validate_model_pair",
    "resolve_device",
    "clean",
    "XMergeError",
    "ArchitectureMismatchError",
    "DimensionMismatchError",
    "InsufficientMemoryError",
]


class XMergeError(Exception):
    """Base exception for xmerge errors."""


class ArchitectureMismatchError(XMergeError):
    """Raised when models have incompatible architectures."""


class DimensionMismatchError(XMergeError):
    """Raised when hidden dimensions are incompatible."""


class InsufficientMemoryError(XMergeError):
    """Raised when there isn't enough GPU/CPU memory."""


_DEVICE: Optional[torch.device] = None


def resolve_device(prefer: str = "cuda") -> torch.device:
    """Resolve the best available device.
    
    Args:
        prefer: Preferred device type ("cuda" or "cpu")
    
    Returns:
        torch.device: The resolved device
    """
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    
    if prefer == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("Using CUDA device: %s (%.1f GB)", gpu_name, gpu_mem)
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    _DEVICE = device
    return device


def clean() -> None:
    """Aggressively clean GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def svd_project(
    W: torch.Tensor,
    out_t: int,
    in_t: int,
) -> torch.Tensor:
    """Project weight matrix to target dimensions via SVD.
    
    Uses truncated SVD to project a weight matrix to new dimensions,
    preserving the most important singular directions.
    
    Args:
        W: Weight matrix of shape [d_out, d_in]
        out_t: Target output dimension
        in_t: Target input dimension
    
    Returns:
        Projected weight matrix of shape [out_t, in_t]
    
    Raises:
        ValueError: If W is not 2-dimensional
    """
    if W.dim() != 2:
        raise ValueError(f"Expected 2D weight matrix, got {W.dim()}D")
    if W.shape == (out_t, in_t):
        return W
    
    W = W.float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    m, n = U.shape[0], Vh.shape[1]
    k = min(m, n, out_t, in_t)
    
    W2 = torch.zeros(out_t, in_t, dtype=W.dtype, device=W.device)
    reconstructed = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k]
    W2[: min(m, out_t), : min(n, in_t)] = reconstructed[: min(m, out_t), : min(n, in_t)]
    return W2


def svd_project_activation_aware(
    W: torch.Tensor,
    out_t: int,
    in_t: int,
    activations: torch.Tensor,
) -> torch.Tensor:
    """SVD projection weighted by activation importance.
    
    Uses activation statistics to select which singular directions to keep,
    preserving the directions most important for the actual data distribution.
    
    Args:
        W: Weight matrix of shape [d_out, d_in]
        out_t: Target output dimension
        in_t: Target input dimension
        activations: Input activations of shape [n_samples, d_in]
    
    Returns:
        Projected weight matrix of shape [out_t, in_t]
    """
    if W.shape == (out_t, in_t):
        return W
    if activations.dim() != 2:
        raise ValueError(f"Expected 2D activations, got {activations.dim()}D")
    
    W = W.float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    m, n = U.shape[0], Vh.shape[1]
    k_full = min(m, n)
    max_k = min(m, n, out_t, in_t)
    
    with torch.no_grad():
        proj = activations.float() @ Vh[:k_full].T
        importance = proj.norm(dim=0)
    
    top_idx = importance.topk(max_k).indices.sort().values
    
    W2 = torch.zeros(out_t, in_t, dtype=W.dtype, device=W.device)
    reconstructed = U[:, top_idx] @ torch.diag(S[top_idx]) @ Vh[top_idx]
    W2[: min(m, out_t), : min(n, in_t)] = reconstructed[: min(m, out_t), : min(n, in_t)]
    return W2


def proportional_map(n_a: int, n_b: int) -> Dict[int, int]:
    """Map n_a layers to n_b layers proportionally.
    
    Creates a mapping from each layer index in model A to the closest
    corresponding layer index in model B, proportional to layer count.
    
    Args:
        n_a: Number of layers in model A
        n_b: Number of layers in model B
    
    Returns:
        Dict mapping layer index in A to layer index in B
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError(f"Layer counts must be positive: n_a={n_a}, n_b={n_b}")
    return {i: min(int((i + 0.5) * n_b / n_a), n_b - 1) for i in range(n_a)}


def hidden_dim(cfg: Any) -> int:
    """Extract hidden dimension from HuggingFace model config.
    
    Args:
        cfg: HuggingFace model config object
    
    Returns:
        Hidden dimension size
    """
    return (
        getattr(cfg, "hidden_size", None)
        or getattr(cfg, "n_embd", None)
        or getattr(cfg, "d_model", None)
        or 768
    )


def num_layers(cfg: Any) -> Optional[int]:
    """Extract number of transformer layers from HuggingFace model config.
    
    Args:
        cfg: HuggingFace model config object
    
    Returns:
        Number of layers, or None if not found
    """
    return (
        getattr(cfg, "num_hidden_layers", None)
        or getattr(cfg, "n_layer", None)
        or getattr(cfg, "num_layers", None)
    )


@torch.no_grad()
def compute_ppl(
    model: torch.nn.Module,
    ids: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """Compute perplexity of a model on tokenized data.
    
    Args:
        model: The language model
        ids: Token IDs of shape [batch, seq_len]
        mask: Optional attention mask
    
    Returns:
        Perplexity value
    
    Raises:
        ValueError: If ids is empty or model returns NaN loss
    """
    if ids.numel() == 0:
        raise ValueError("Cannot compute PPL on empty input")
    
    model.eval()
    loss = model(input_ids=ids, attention_mask=mask, labels=ids).loss
    if torch.isnan(loss) or torch.isinf(loss):
        raise ValueError(f"Model returned invalid loss: {loss.item()}")
    
    return math.exp(loss.item())


@torch.no_grad()
def generate_text(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str = "The future of AI is",
    max_new: int = 40,
    temperature: float = 0.8,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    """Generate text from a model.
    
    Args:
        model: The language model
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt text
        max_new: Maximum number of new tokens to generate
        temperature: Sampling temperature (0 = greedy)
        top_p: Nucleus sampling threshold
        repetition_penalty: Penalty for repeated tokens
    
    Returns:
        Generated text string
    """
    model.eval()
    device = next(model.parameters()).device
    inp = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inp,
        max_new_tokens=max_new,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else 1.0,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def load_calibration(
    model_name: str = "gpt2",
    n: int = 32,
    seq: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, AutoTokenizer]:
    """Load calibration data from WikiText-2.
    
    Args:
        model_name: Name of model to load tokenizer for
        n: Number of calibration sequences to load
        seq: Maximum sequence length
    
    Returns:
        Tuple of (input_ids, attention_mask, tokenizer)
    """
    device = resolve_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t["text"] for t in ds if len(t["text"].strip()) > 50][: n * 2]
    except Exception as e:
        logger.warning("Failed to load WikiText-2: %s. Using fallback texts.", e)
        texts = ["The quick brown fox jumps over the lazy dog."] * n
    
    if not texts:
        texts = ["The quick brown fox jumps over the lazy dog."] * n
        logger.warning("No calibration texts found, using fallback.")
    
    enc = tokenizer(
        texts[:n],
        truncation=True,
        padding=True,
        max_length=seq,
        return_tensors="pt",
    )
    return enc.input_ids.to(device), enc.attention_mask.to(device), tokenizer


def build_token_map(
    tok_a: AutoTokenizer,
    tok_b: AutoTokenizer,
    strategy: str = "multi",
) -> Dict[int, int]:
    """Build a token mapping from tokenizer A to tokenizer B.
    
    Maps each token ID in tokenizer A to the closest corresponding token ID
    in tokenizer B. Uses multiple strategies for better coverage:
    
    1. "single": Original single-token mapping (fast, lower coverage)
    2. "multi": Multi-token with overlap scoring (slower, higher coverage)
    
    Args:
        tok_a: Source tokenizer
        tok_b: Target tokenizer
        strategy: Mapping strategy ("single" or "multi")
    
    Returns:
        Dict mapping token IDs from A to token IDs in B
    
    Raises:
        ValueError: If strategy is unknown
    """
    if strategy == "single":
        return _build_token_map_single(tok_a, tok_b)
    elif strategy == "multi":
        return _build_token_map_multi(tok_a, tok_b)
    else:
        raise ValueError(f"Unknown token map strategy: {strategy}. Use 'single' or 'multi'.")


def _build_token_map_single(tok_a: AutoTokenizer, tok_b: AutoTokenizer) -> Dict[int, int]:
    """Single-token mapping: map each A token to first B subword."""
    token_map: Dict[int, int] = {}
    pad_id = tok_b.pad_token_id or 0
    for i in range(tok_a.vocab_size):
        s = tok_a.decode([i]).strip()
        if not s:
            token_map[i] = pad_id
            continue
        bid = tok_b.encode(s, add_special_tokens=False)
        token_map[i] = bid[0] if bid else pad_id
    
    coverage = sum(1 for v in token_map.values() if v != pad_id) / len(token_map) * 100
    logger.info("Token map (%s): %d entries, %.1f%% non-zero coverage",
                 tok_a.name_or_path, len(token_map), coverage)
    return token_map


def _build_token_map_multi(tok_a: AutoTokenizer, tok_b: AutoTokenizer) -> Dict[int, int]:
    """Multi-token mapping with overlap scoring for higher coverage.
    
    For each A token, encodes it through B's tokenizer. If it maps to
    a single B token, uses that. If it splits into multiple B tokens,
    scores by token overlap and picks the best single match.
    """
    token_map: Dict[int, int] = {}
    pad_id = tok_b.pad_token_id or 0
    b_vocab = set(range(tok_b.vocab_size))

    for i in range(tok_a.vocab_size):
        s = tok_a.decode([i]).strip()
        if not s:
            token_map[i] = pad_id
            continue
        
        bid = tok_b.encode(s, add_special_tokens=False)
        
        if len(bid) == 1:
            token_map[i] = bid[0]
        elif len(bid) > 1:
            # Try to find a single B token that decodes to the same string
            for b_id in sorted(b_vocab):
                b_str = tok_b.decode([b_id]).strip()
                if b_str and (b_str == s or b_str in s or s in b_str):
                    token_map[i] = b_id
                    break
            else:
                token_map[i] = bid[0]
        else:
            token_map[i] = pad_id
    
    coverage = sum(1 for v in token_map.values() if v != pad_id) / len(token_map) * 100
    logger.info("Token map (multi): %d entries, %.1f%% non-zero coverage",
                 len(token_map), coverage)
    return token_map


def validate_model_pair(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    require_same_arch: bool = False,
) -> Dict[str, Any]:
    """Validate that two models can be merged.
    
    Checks architecture compatibility, hidden dimensions, and device placement.
    
    Args:
        model_a: First model
        model_b: Second model
        require_same_arch: If True, require identical architectures
    
    Returns:
        Dict with validation metadata: d_a, d_b, n_layers_a, n_layers_b, etc.
    
    Raises:
        ArchitectureMismatchError: If architectures are incompatible
        DimensionMismatchError: If dimensions are incompatible
    """
    cfg_a = model_a.config
    cfg_b = model_b.config
    
    model_type_a = getattr(cfg_a, "model_type", "unknown")
    model_type_b = getattr(cfg_b, "model_type", "unknown")
    
    d_a = hidden_dim(cfg_a)
    d_b = hidden_dim(cfg_b)
    
    n_layers_a = num_layers(cfg_a)
    n_layers_b = num_layers(cfg_b)
    
    if n_layers_a is None or n_layers_b is None:
        raise ArchitectureMismatchError(
            f"Cannot determine layer counts: A={n_layers_a}, B={n_layers_b}"
        )
    
    info = {
        "model_a": model_type_a,
        "model_b": model_type_b,
        "d_a": d_a,
        "d_b": d_b,
        "n_layers_a": n_layers_a,
        "n_layers_b": n_layers_b,
        "same_arch": model_type_a == model_type_b,
        "same_size": d_a == d_b and n_layers_a == n_layers_b,
    }
    
    logger.info(
        "Model pair: %s (d=%d, %d layers) + %s (d=%d, %d layers)",
        model_type_a, d_a, n_layers_a,
        model_type_b, d_b, n_layers_b,
    )
    
    if require_same_arch and model_type_a != model_type_b:
        raise ArchitectureMismatchError(
            f"Models have different architectures: {model_type_a} vs {model_type_b}. "
            f"Use cross-architecture bridge (method='bridge') instead."
        )
    
    return info
