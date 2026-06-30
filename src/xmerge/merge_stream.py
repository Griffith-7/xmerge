"""
Layer-by-layer streaming for xmerge — enables merging 7B models on 4GB VRAM.

Strategy for low RAM:
  1. Models loaded via accelerate with CPU offload (weights stay on disk until needed)
  2. Forward passes process 1 layer at a time on GPU, rest on disk
  3. For bridge training: cache hidden states from ONE model at a time, then train bridge purely on GPU

Approach 1 — Weight blending:    Streamed CKA + CPU weight merge
Approach 2 — Bridge v2:          Streamed forward each step
Approach 3 — Bridge cached:      Stream cache once per model, train bridge on GPU
Approach 4 — Full pipeline:      Cache + train + save + eval
"""

import contextlib
import copy
import gc
import json
import logging
import math
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import merge_prod, utils

__all__ = [
    "clean",
    "StreamedForward",
    "load_model_streamed",
    "load_one_model_at_a_time",
    "activation_similarity",
    "compute_cka_streamed",
    "streamed_merge_same_arch",
    "streamed_train_bridge_v2",
    "streamed_train_bridge_cached",
    "streamed_merge_diff_arch",
    "StreamedGenerator",
]

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────

def clean() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_embedding_modules(model: nn.Module) -> Dict[str, nn.Module]:
    mt = getattr(model.config, "model_type", "").lower()
    if mt == "gpt2":
        return {"wte": model.transformer.wte, "wpe": model.transformer.wpe}
    if mt in ("llama", "mistral", "qwen2", "phi", "phi3", "smollm2", "gemma", "zephyr"):
        return {"embed_tokens": model.model.embed_tokens}
    if mt == "opt":
        d: Dict[str, nn.Module] = {"embed_tokens": model.model.decoder.embed_tokens}
        if hasattr(model.model.decoder, "embed_positions"):
            d["embed_positions"] = model.model.decoder.embed_positions
        return d
    for name, mod in model.named_modules():
        if "wte" in name or "embed_tokens" in name:
            if isinstance(mod, nn.Embedding):
                return {name: mod}
    return {}


def _get_final_norm(model: nn.Module) -> Optional[nn.Module]:
    return merge_prod._get_final_norm(model)


def _copy_lm_head(model: nn.Module, device: torch.device) -> Optional[nn.Linear]:
    lh = merge_prod._get_lm_head(model)
    if lh is None:
        return None
    has_bias = lh.bias is not None
    dtype = lh.weight.dtype
    copy_lh = nn.Linear(lh.in_features, lh.out_features, bias=has_bias, dtype=dtype)
    copy_lh.load_state_dict(lh.state_dict())
    return copy_lh.to(device)


# ── Streamed Forward ─────────────────────────────────────────────────────

class StreamedForward:
    """Forward pass with layer-by-layer GPU streaming.

    Only 1 layer on GPU at a time. Supports hooks for CKA.
    Models should be loaded via accelerate CPU offload or regular CPU.
    """

    def __init__(self, model: nn.Module, device: str = "cuda", hooks: Optional[Dict] = None) -> None:
        self.model = model
        self.device = device
        self.model.config.use_cache = False
        self.layers = merge_prod._get_layer_list(model)
        self.n_layers = len(self.layers)
        self.final_norm = _get_final_norm(model)
        self.embeddings = _get_embedding_modules(model)
        self.model_type = getattr(model.config, "model_type", "").lower()
        self.hooks = hooks or {}

    def _apply_hooks(self, i: int, hidden: torch.Tensor) -> None:
        if i in self.hooks:
            self.hooks[i](hidden)

    def _compute_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        device = self.device
        if "wte" in self.embeddings and "wpe" in self.embeddings:
            wte = self.embeddings["wte"].to(device)
            wpe = self.embeddings["wpe"].to(device)
            pos = torch.arange(0, input_ids.shape[1], device=device).unsqueeze(0)
            hidden = wte(input_ids.to(device)) + wpe(pos)
            self.embeddings["wte"].to("cpu")
            self.embeddings["wpe"].to("cpu")
        elif "embed_tokens" in self.embeddings:
            et = self.embeddings["embed_tokens"].to(device)
            hidden = et(input_ids.to(device))
            self.embeddings["embed_tokens"].to("cpu")
            if "embed_positions" in self.embeddings:
                ep = self.embeddings["embed_positions"].to(device)
                pos = torch.arange(0, input_ids.shape[1], device=device).unsqueeze(0)
                hidden = hidden + ep(pos)
                self.embeddings["embed_positions"].to("cpu")
        else:
            et = list(self.embeddings.values())[0].to(device)
            hidden = et(input_ids.to(device))
            list(self.embeddings.values())[0].to("cpu")
        clean()
        return hidden

    @torch.no_grad()
    def __call__(
        self, input_ids: torch.Tensor, output_hidden_states: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        hidden = self._compute_embeddings(input_ids)
        self._apply_hooks(-1, hidden.cpu())

        if output_hidden_states:
            all_hidden: List[torch.Tensor] = [hidden.cpu()]

        position_embeddings = None
        if self.model_type in ("llama", "mistral", "qwen2", "gemma"):
            model_obj = self.model.model
            if hasattr(model_obj, "rotary_emb"):
                seq_len = hidden.shape[1]
                position_ids = torch.arange(0, seq_len, device=self.device).unsqueeze(0)
                cos, sin = model_obj.rotary_emb(hidden.to(self.device), position_ids)
                position_embeddings = (cos.to(hidden.dtype), sin.to(hidden.dtype))

        for i in range(self.n_layers):
            layer = self.layers[i].to(self.device)
            hidden = hidden.to(self.device)
            if position_embeddings is not None:
                hidden_out = layer(hidden, position_embeddings=position_embeddings)
            else:
                hidden_out = layer(hidden)  # type: ignore[operator]
            hidden = hidden_out[0] if isinstance(hidden_out, tuple) else hidden_out
            self.layers[i] = self.layers[i].to("cpu")
            hidden_cpu = hidden.cpu()
            self._apply_hooks(i, hidden_cpu)

            if output_hidden_states:
                all_hidden.append(hidden_cpu.clone())

            del layer, hidden_out
            clean()

        hidden = hidden_cpu
        if self.final_norm is not None:
            self.final_norm = self.final_norm.to("cpu")
            hidden = self.final_norm(hidden)

        if output_hidden_states:
            return hidden, all_hidden
        return hidden


# ── Model Loading (memory-efficient) ─────────────────────────────────────

def load_model_streamed(
    model_name: str,
    dtype: torch.dtype = torch.float16,
    cache_dir: Optional[str] = None,
    use_offload: bool = True,
    offload_folder: str = "offload",
) -> Tuple[nn.Module, AutoTokenizer]:

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading %s (dtype=%s)%s", model_name, dtype, " with offload" if use_offload else "")

    if use_offload:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="cpu",
            offload_folder=offload_folder,
            offload_state_dict=True,
            cache_dir=cache_dir,
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            cache_dir=cache_dir,
        ).eval()
        for p in model.parameters():
            p.data = p.data.cpu()

    clean()
    return model, tokenizer


@contextlib.contextmanager
def load_one_model_at_a_time(
    model_name: str, dtype: torch.dtype = torch.float16, cache_dir: Optional[str] = None
):
    logger.info("Loading %s into memory...", model_name)
    model, tok = load_model_streamed(model_name, dtype, cache_dir, use_offload=False)
    try:
        yield model, tok
    finally:
        logger.info("Unloading %s...", model_name)
        del model
        clean()


# ═══════════════════════════════════════════════════════════════════════════
# APPROACH 1 — Weight Blending (streamed CKA)
# ═══════════════════════════════════════════════════════════════════════════

def activation_similarity(h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
    ha_2d = h_a.reshape(-1, h_a.shape[-1])
    hb_2d = h_b.reshape(-1, h_b.shape[-1])
    min_n = min(ha_2d.shape[0], hb_2d.shape[0])
    return merge_prod.hsic_cka(ha_2d[:min_n], hb_2d[:min_n])


def compute_cka_streamed(
    model_a: nn.Module, model_b: nn.Module, input_ids: torch.Tensor, device: str = "cuda"
) -> Tuple[Dict[Tuple[int, int], float], Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    n_a = len(merge_prod._get_layer_list(model_a))
    n_b = len(merge_prod._get_layer_list(model_b))
    hiddens_a: Dict[int, torch.Tensor] = {}
    hiddens_b: Dict[int, torch.Tensor] = {}

    def make_hook(store: Dict) -> Callable:
        def hook(idx: int):
            def fn(h: torch.Tensor) -> None:
                store[idx] = h.float().cpu()
            return fn
        return hook

    hooks_a = {i: make_hook(hiddens_a)(i) for i in range(n_a)}
    hooks_a[-1] = make_hook(hiddens_a)(-1)
    hooks_b = {i: make_hook(hiddens_b)(i) for i in range(n_b)}
    hooks_b[-1] = make_hook(hiddens_b)(-1)

    sa = StreamedForward(model_a, device, hooks=hooks_a)
    sb = StreamedForward(model_b, device, hooks=hooks_b)

    logger.info("  Streaming forward through model A...")
    sa(input_ids)
    logger.info("  Streaming forward through model B...")
    sb(input_ids)

    sim: Dict[Tuple[int, int], float] = {}
    for i_a in range(n_a):
        for i_b in range(n_b):
            ha = hiddens_a.get(i_a)
            hb = hiddens_b.get(i_b)
            if ha is None or hb is None:
                sim[(i_a, i_b)] = 0.0
            else:
                sim[(i_a, i_b)] = activation_similarity(ha, hb).item()

    return sim, hiddens_a, hiddens_b


def _interpolate_weights(w_b: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    if w_b.shape == target_shape:
        return w_b
    if w_b.dim() == 1:
        return F.interpolate(w_b.view(1, 1, -1), size=target_shape[0], mode="linear").view(-1)
    if w_b.dim() == 2:
        return utils.svd_project(w_b, target_shape[0], target_shape[1])
    return w_b


def _blend_weights(
    sd_a: Dict[str, torch.Tensor],
    b_proj: Dict,
    prefix_a: str,
    layers_a: Dict,
    alphas: Dict[int, float],
    n_a: int,
) -> Dict[str, torch.Tensor]:
    merged_sd: Dict[str, torch.Tensor] = {}
    for k, v in sd_a.items():
        if merge_prod._is_embed_or_output_key(k):
            merged_sd[k] = v.clone()
        elif k in b_proj and not k.startswith(prefix_a):
            avg_a = sum(alphas.values()) / max(len(alphas), 1)
            merged_sd[k] = (avg_a * v.float() + (1 - avg_a) * b_proj[k]).to(v.dtype)
        else:
            merged_sd[k] = v.clone()

    for k in sd_a:
        if k.startswith(prefix_a):
            rest = k[len(prefix_a):]
            parts = rest.split(".")
            i_a = int(parts[0])
            local = ".".join(parts[1:])
            a = alphas.get(i_a, 0.5)
            key = (i_a, local)
            if key in b_proj:
                try:
                    merged_sd[k] = (a * sd_a[k].float() + (1 - a) * b_proj[key]).to(sd_a[k].dtype)
                except Exception:
                    merged_sd[k] = sd_a[k].clone()
    return merged_sd


def streamed_merge_improved(
    model_a: nn.Module,
    model_b: nn.Module,
    calib_texts: Optional[List[str]] = None,
    save_name: Optional[str] = None,
    device: str = "cuda",
    use_procrustes: bool = False,
) -> Tuple[nn.Module, AutoTokenizer, float, Dict[int, float]]:
    logger.info("=" * 60)
    logger.info("  IMPROVED Weight Blending (interpolated layers + alpha search)")
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        model_a.config._name_or_path or "distilgpt2"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if calib_texts is None:
        calib_texts = list(merge_prod.load_texts(8)) if hasattr(merge_prod, "load_texts") else ["default text"] * 8

    enc = tokenizer(calib_texts[:4], truncation=True, padding=True, max_length=64, return_tensors="pt")
    ids = enc.input_ids

    n_a = utils.num_layers(model_a.config)
    n_b = utils.num_layers(model_b.config)
    logger.info("  Layers: A=%d, B=%d", n_a, n_b)

    logger.info("  Computing CKA similarity...")
    sim, ha, hb = compute_cka_streamed(model_a, model_b, ids, device)
    avg_sim = float(np.mean(list(sim.values()))) if sim else 0.0

    logger.info("  Building interpolated layer mapping...")
    prefix_a = merge_prod._get_layer_prefix(model_a)
    prefix_b = merge_prod._get_layer_prefix(model_b)
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()

    layers_a: Dict[int, Dict] = {}
    for k, v in sd_a.items():
        if k.startswith(prefix_a):
            parts = k[len(prefix_a):].split(".")
            idx, local = int(parts[0]), ".".join(parts[1:])
            layers_a.setdefault(idx, {})[local] = v

    layers_b: Dict[int, Dict] = {}
    for k, v in sd_b.items():
        if k.startswith(prefix_b):
            parts = k[len(prefix_b):].split(".")
            idx, local = int(parts[0]), ".".join(parts[1:])
            layers_b.setdefault(idx, {})[local] = v

    mapping: Dict[int, int] = {}
    for i_a in range(n_a):
        best_b = max(range(n_b), key=lambda ib: sim.get((i_a, ib), 0))
        mapping[i_a] = best_b

    logger.info("  CKA-based mapping done (avg sim: %.3f)", avg_sim)

    b_proj: Dict[Union[Tuple[int, str], str], torch.Tensor] = {}
    skipped: List[Tuple[int, str, str]] = []
    for i_a, i_b in mapping.items():
        for local, w_a in layers_a[i_a].items():
            if i_b not in layers_b:
                continue
            w_b = layers_b[i_b].get(local)
            if w_b is None:
                continue
            try:
                w_b_proj = _interpolate_weights(w_b.float(), w_a.shape)
                b_proj[(i_a, local)] = w_b_proj
            except Exception as e:
                skipped.append((i_a, local, str(e)))

    for k in sd_a:
        if not k.startswith(prefix_a) and k in sd_b:
            if sd_a[k].shape == sd_b[k].shape:
                b_proj[k] = sd_b[k].float()

    if skipped:
        logger.info("  Skipped %d incompatible weights", len(skipped))

    if use_procrustes and ha and hb:
        logger.info("  Applying orthogonal Procrustes alignment...")
        try:
            b_proj = merge_prod._apply_procrustes_alignment(
                model_a, model_b, mapping, b_proj, ha, hb
            )
        except Exception as e:
            logger.info("  Procrustes skipped: %s", e)

    logger.info("  Optimizing per-layer alphas...")
    best_alpha_val = 0.5
    best_loss = float("inf")
    for a_test in [0.3, 0.5, 0.7]:
        merged_sd = _blend_weights(sd_a, b_proj, prefix_a, layers_a, {i: a_test for i in range(n_a)}, n_a)
        try:
            merged_test = AutoModelForCausalLM.from_config(model_a.config)
            merged_test.load_state_dict(merged_sd, strict=False)
            sf = StreamedForward(merged_test, device)
            h = sf(ids)
            lm = merge_prod._get_lm_head(merged_test)
            if lm is not None:
                logits = lm(h.to(dtype=next(merged_test.parameters()).dtype))
                sl, ll = logits[..., :-1, :].contiguous(), ids[..., 1:].contiguous()
                loss = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1), ignore_index=-100)
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_alpha_val = a_test
        except Exception:
            pass
        del merged_test
        clean()

    alphas = {i: best_alpha_val for i in range(n_a)}

    merged_sd = _blend_weights(sd_a, b_proj, prefix_a, layers_a, alphas, n_a)
    merged = AutoModelForCausalLM.from_config(model_a.config)
    merged.load_state_dict(merged_sd, strict=False)

    logger.info("  Computing merged model PPL...")
    stream = StreamedForward(merged, device)
    hidden = stream(ids)
    lm_head = merge_prod._get_lm_head(merged)
    if lm_head is not None:
        logits = lm_head(hidden.to(dtype=next(merged.parameters()).dtype))
        sl, ll = logits[..., :-1, :].contiguous(), ids[..., 1:].contiguous()
        loss = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1), ignore_index=-100)
        ppl_val = math.exp(loss.item())
    else:
        ppl_val = float("inf")

    if save_name:
        save_dir = os.path.join(merge_prod.SAVE_DIR, save_name)
        os.makedirs(save_dir, exist_ok=True)
        merged.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    clean()
    return merged, tokenizer, ppl_val, alphas


streamed_merge_same_arch = streamed_merge_improved


# ═══════════════════════════════════════════════════════════════════════════
# APPROACH 2 — Bridge v2 (streamed forward each step)
# ═══════════════════════════════════════════════════════════════════════════

def _holdout_split(
    texts: Sequence[str], eval_pct: float = 0.2
) -> Tuple[Sequence[str], Sequence[str]]:
    n = len(texts)
    n_eval = max(min(int(n * eval_pct), n // 2), min(4, n // 2))
    if n_eval < 1:
        return texts, []
    return texts[:-n_eval], texts[-n_eval:]


def _streamed_eval_ppl(
    model_a: nn.Module,
    model_b: nn.Module,
    bridge: nn.Module,
    tokenizer: AutoTokenizer,
    eval_texts: Sequence[str],
    device: str,
    max_len: int = 128,
    batch_size: int = 2,
) -> float:
    if not eval_texts:
        return float("inf")
    lm_head = _copy_lm_head(model_a, device)
    if lm_head is None:
        return float("inf")
    model_dtype = next(model_a.parameters()).dtype
    enc = tokenizer(list(eval_texts), truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    ids = enc.input_ids
    n = ids.shape[0]
    total_loss = 0.0
    with torch.no_grad():
        for b in range(0, n, batch_size):
            batch_ids = ids[b:b + batch_size]
            ha = StreamedForward(model_a, device)(batch_ids)
            hb = StreamedForward(model_b, device)(batch_ids)
            k = min(ha.shape[1], hb.shape[1])
            ha_gpu = ha[:, :k].float().to(device)
            hb_gpu = hb[:, :k].float().to(device)
            ids_gpu = batch_ids[:, :k].to(device)
            hf = bridge(ha_gpu, hb_gpu)
            logits = lm_head(hf.to(model_dtype))
            sl, ll = logits[..., :-1, :].contiguous(), ids_gpu[..., 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1), ignore_index=-100)
            total_loss += loss.item()
            del ha, hb, ha_gpu, hb_gpu, ids_gpu, hf, logits, sl, ll
            clean()
    lm_head.to("cpu")
    avg = total_loss / max(1, (n + batch_size - 1) // batch_size)
    return math.exp(avg)


def streamed_train_bridge_v2(
    model_a: nn.Module,
    model_b: nn.Module,
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    device: str = "cuda",
    steps: int = 10,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    max_len: int = 128,
    batch_size: int = 2,
    eval_texts: Optional[Sequence[str]] = None,
    bridge_type: str = "linear",
) -> Tuple[nn.Module, float]:
    if eval_texts is not None:
        train_texts = texts
    else:
        train_texts, eval_texts = _holdout_split(texts)
    if not train_texts:
        raise ValueError("texts cannot be empty")

    logger.info("=" * 60)
    logger.info("  APPROACH 2: Bridge v2 (streamed forward each step)")
    logger.info("=" * 60)

    d_a = utils.hidden_dim(model_a.config)
    d_b = utils.hidden_dim(model_b.config)
    if bridge_type == "mlp":
        bridge = merge_prod.MLPBridge(d_a, d_b).to(device)
    else:
        bridge = merge_prod.OptimalBridge(d_a, d_b).to(device)
    lm_head = _copy_lm_head(model_a, device)
    model_dtype = next(model_a.parameters()).dtype

    enc = tokenizer(list(train_texts), truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    ids, _ = enc.input_ids, enc.attention_mask

    opt = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    best: Optional[Dict[str, torch.Tensor]] = None
    best_loss = float("inf")
    n = ids.shape[0]

    torch.set_grad_enabled(True)
    bridge.train()

    for s in range(steps):
        opt.zero_grad()
        total_loss = 0.0
        for b in range(0, n, batch_size):
            batch_ids = ids[b:b + batch_size]
            ha = StreamedForward(model_a, device)(batch_ids)
            hb = StreamedForward(model_b, device)(batch_ids)
            k = min(ha.shape[1], hb.shape[1])
            ha_gpu = ha[:, :k].float().to(device)
            hb_gpu = hb[:, :k].float().to(device)
            ids_gpu = batch_ids[:, :k].to(device)
            hf = bridge(ha_gpu, hb_gpu)
            logits = lm_head(hf.to(model_dtype))
            sl, ll = logits[..., :-1, :].contiguous(), ids_gpu[..., 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1), ignore_index=-100)
            loss.backward()
            total_loss += loss.item()
            del ha, hb, ha_gpu, hb_gpu, ids_gpu, hf, logits, sl, ll
            clean()

        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        opt.step()
        sched.step()

        avg_loss = total_loss / max(1, (n + batch_size - 1) // batch_size)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best = copy.deepcopy(bridge.state_dict())

        logger.info("    Step %d/%d  loss=%.4f", s + 1, steps, avg_loss)

    bridge.load_state_dict(best if best else bridge.state_dict())
    bridge.eval()
    torch.set_grad_enabled(False)

    eval_ppl = _streamed_eval_ppl(model_a, model_b, bridge, tokenizer, eval_texts, device, max_len, batch_size)
    if math.isfinite(eval_ppl):
        logger.info("    Eval PPL (held-out): %.1f", eval_ppl)
    else:
        logger.info("    (no held-out eval texts)")

    lm_head.to("cpu")
    clean()
    return bridge, eval_ppl


# ═══════════════════════════════════════════════════════════════════════════
# APPROACH 3 — Bridge cached
# ═══════════════════════════════════════════════════════════════════════════

def streamed_train_bridge_cached(
    model_a: nn.Module,
    model_b: nn.Module,
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    device: str = "cuda",
    steps: int = 20,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    max_len: int = 128,
    batch_size: int = 2,
    eval_texts: Optional[Sequence[str]] = None,
    bridge_type: str = "linear",
) -> Tuple[nn.Module, float]:
    if eval_texts is not None:
        train_texts = texts
    else:
        train_texts, eval_texts = _holdout_split(texts)
    if not train_texts:
        raise ValueError("texts cannot be empty")

    logger.info("=" * 60)
    logger.info("  APPROACH 3: Bridge Cached (stream cache once, GPU train)")
    logger.info("=" * 60)

    d_a = utils.hidden_dim(model_a.config)
    d_b = utils.hidden_dim(model_b.config)
    if bridge_type == "mlp":
        bridge = merge_prod.MLPBridge(d_a, d_b).to(device)
    else:
        bridge = merge_prod.OptimalBridge(d_a, d_b).to(device)

    logger.info("  Caching hidden states (stream both models)...")
    enc = tokenizer(list(train_texts), truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    ids = enc.input_ids
    n = ids.shape[0]

    all_ha: List[torch.Tensor] = []
    all_hb: List[torch.Tensor] = []
    all_ids: List[torch.Tensor] = []
    for b in range(0, n, batch_size):
        batch_ids = ids[b:b + batch_size]
        logger.info("    Forward model A batch %d...", b // batch_size + 1)
        ha = StreamedForward(model_a, device)(batch_ids)
        logger.info("    Forward model B batch %d...", b // batch_size + 1)
        hb = StreamedForward(model_b, device)(batch_ids)
        k = min(ha.shape[1], hb.shape[1])
        all_ha.append(ha[:, :k].float())
        all_hb.append(hb[:, :k].float())
        all_ids.append(batch_ids[:, :k])

    ha_full = torch.cat(all_ha, dim=0)
    hb_full = torch.cat(all_hb, dim=0)
    ids_full = torch.cat(all_ids, dim=0)
    logger.info("  Cached: ha=%s, hb=%s", list(ha_full.shape), list(hb_full.shape))

    ha_gpu = ha_full.to(device)
    hb_gpu = hb_full.to(device)
    ids_gpu = ids_full.to(device)
    lm_head = _copy_lm_head(model_a, device)
    model_dtype = next(model_a.parameters()).dtype

    opt = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    best: Optional[Dict[str, torch.Tensor]] = None
    best_loss = float("inf")

    torch.set_grad_enabled(True)
    bridge.train()

    for s in range(steps):
        opt.zero_grad()
        hf = bridge(ha_gpu, hb_gpu)
        logits = lm_head(hf.to(model_dtype))
        sl, ll = logits[..., :-1, :].contiguous(), ids_gpu[..., 1:].contiguous()
        loss = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        opt.step()
        sched.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best = copy.deepcopy(bridge.state_dict())

        if (s + 1) % 5 == 0:
            logger.info("    Step %d/%d  loss=%.4f", s + 1, steps, loss.item())

    bridge.load_state_dict(best if best else bridge.state_dict())
    bridge.eval()
    torch.set_grad_enabled(False)

    eval_ppl = _streamed_eval_ppl(model_a, model_b, bridge, tokenizer, eval_texts, device, max_len, batch_size)
    if math.isfinite(eval_ppl):
        logger.info("    Eval PPL (held-out): %.1f", eval_ppl)
    else:
        logger.info("    (no held-out eval texts)")

    del ha_gpu, hb_gpu, ids_gpu, lm_head, ha_full, hb_full, ids_full
    clean()
    return bridge, eval_ppl


# ═══════════════════════════════════════════════════════════════════════════
# APPROACH 4 — Full pipeline
# ═══════════════════════════════════════════════════════════════════════════

def streamed_merge_diff_arch(
    model_a: nn.Module,
    model_b: nn.Module,
    calib_texts: Optional[List[str]] = None,
    save_name: str = "streamed_merge",
    device: str = "cuda",
    tokenizer: Optional[AutoTokenizer] = None,
    steps: int = 10,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    max_len: int = 128,
    bridge_type: str = "linear",
) -> Tuple[nn.Module, float]:
    logger.info("=" * 60)
    logger.info("  APPROACH 4: Full Pipeline (cached + save + eval)")
    logger.info("=" * 60)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_a.config._name_or_path or "distilgpt2"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if calib_texts is None:
        calib_texts = list(merge_prod.load_texts(16)) if hasattr(merge_prod, "load_texts") else ["default text"] * 16

    enc = tokenizer(calib_texts[:4], truncation=True, padding=True, max_length=64, return_tensors="pt")
    ids = enc.input_ids
    logger.info("  Computing model A baseline PPL...")
    ha = StreamedForward(model_a, device)(ids)
    lm_head_copy = _copy_lm_head(model_a, device)
    ppl_a = float("inf")
    if lm_head_copy is not None:
        logits = lm_head_copy(ha.to(dtype=next(model_a.parameters()).dtype).to(device))
        sl, ll = logits[..., :-1, :].contiguous(), ids[..., 1:].to(device).contiguous()
        loss_a = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1), ignore_index=-100)
        ppl_a = math.exp(loss_a.item())

    bridge, ppl_val = streamed_train_bridge_cached(
        model_a, model_b, tokenizer, calib_texts, device=device,
        steps=steps, lr=lr, weight_decay=weight_decay, max_len=max_len,
        bridge_type=bridge_type,
    )

    if save_name:
        save_dir = os.path.join(merge_prod.SAVE_DIR, save_name)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(bridge.state_dict(), os.path.join(save_dir, "bridge.pt"))
        tokenizer.save_pretrained(save_dir)
        with open(os.path.join(save_dir, "bridge_config.json"), "w") as f:
            json.dump({
                "d_a": utils.hidden_dim(model_a.config),
                "d_b": utils.hidden_dim(model_b.config),
                "model_a": model_a.config._name_or_path or "unknown",
                "model_b": model_b.config._name_or_path or "unknown",
                "ppl_a": round(ppl_a, 1),
                "ppl_bridge": round(ppl_val, 1),
                "type": "diff_arch_streamed_bridge",
                "steps": steps,
                "lr": lr,
            }, f, indent=2)
        logger.info("  Saved to %s/", save_dir)

    clean()
    return bridge, ppl_val


# ═══════════════════════════════════════════════════════════════════════════
# GENERATION
# ═══════════════════════════════════════════════════════════════════════════

class StreamedGenerator:
    """Generate text via streamed forward passes."""

    def __init__(
        self,
        model_a: nn.Module,
        model_b: nn.Module,
        bridge: nn.Module,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
        mix_alpha: float = 0.3,
        temp: float = 0.9,
    ) -> None:
        self.model_a = model_a
        self.model_b = model_b
        self.bridge = bridge.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = next(model_a.parameters()).dtype
        self.lm_head = _copy_lm_head(model_a, device)
        self.mix_alpha = mix_alpha
        self.temp = temp

    @torch.no_grad()
    def generate(self, prompt: str, max_new: int = 20, method: str = "bridge") -> str:
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        for _ in range(max_new):
            ha = StreamedForward(self.model_a, self.device)(ids)
            hb = StreamedForward(self.model_b, self.device)(ids)
            k = min(ha.shape[1], hb.shape[1])
            ha_gpu = ha[:, :k].float().to(self.device)
            hb_gpu = hb[:, :k].float().to(self.device)

            if method == "stitch":
                hf = self.bridge(ha_gpu, hb_gpu)
            else:
                h_bridge = self.bridge(ha_gpu, hb_gpu)
                hf = ha_gpu + self.mix_alpha * (h_bridge - ha_gpu)

            logits = self.lm_head(hf.to(self.dtype))[:, -1, :] / self.temp
            next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1).cpu()
            ids = torch.cat([ids, next_tok], dim=-1)
            del ha, hb, ha_gpu, hb_gpu, hf, logits
            clean()
        return self.tokenizer.decode(ids[0], skip_special_tokens=True)
