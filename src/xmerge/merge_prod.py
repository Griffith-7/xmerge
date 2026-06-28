# -*- coding: utf-8 -*-
"""
LLM Merge PROD — Production-ready model merging.

Same-architecture merge (activation-similarity-guided per-layer alpha blending):
    merge_same_arch(model_a, model_b, calib_texts, save_name) -> (model, tokenizer)

Same-architecture bridge (representation-level, works for different sizes):
    merge_same_arch_bridge(model_a, model_b, tok, calib_texts, steps, save_name) -> (bridge, tokenizer)

Cross-architecture bridge (zero-init linear projection + fine-tune):
    build_bridge(ma, mb, tok, texts)
    train_bridge_v2(ma, mb, tok, texts, steps)
    merge_diff_arch(ma, mb, calib_texts, token_map, save_name)

Generation:
    generate_bridge(ma, mb, bridge, tok, prompt)
    stitch_generate(ma, mb, bridge, tok, prompt)
"""

import torch, gc, copy, math, os, json, numpy as np
import torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from . import utils

__all__ = [
    "OptimalBridge", "MLPBridge", "CkaComputer", "activation_similarity",
    "merge_same_arch", "merge_same_arch_bridge", "build_bridge",
    "train_bridge_v2", "train_bridge_cached",
    "merge_diff_arch", "generate_bridge", "stitch_generate",
    "verify_generations", "load_merged",
    "ppl", "clean", "proportional_map", "svd_project", "build_token_map",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
SAVE_DIR = "merged_models"


# ─── ARCHITECTURE DETECTION ──────────────────────────────────────────────

def _get_n_layers(config):
    return getattr(config, 'num_hidden_layers',
           getattr(config, 'n_layer',
           getattr(config, 'n_layers', None)))

def _get_layer_list(model):
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    if hasattr(model, 'model'):
        if hasattr(model.model, 'layers'):
            return model.model.layers
        if hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
            return model.model.decoder.layers
    raise AttributeError(f"Cannot find transformer layer list in {type(model).__name__}. "
                         f"Supported: GPT-2, Llama, OPT, Mistral, Falcon, CodeGen, etc.")

def _get_layer_prefix(model):
    mt = getattr(model.config, 'model_type', '').lower()
    prefixes = {
        'gpt2': 'transformer.h.', 'gpt_neo': 'transformer.h.', 'gptj': 'transformer.h.',
        'codegen': 'transformer.h.', 'falcon': 'transformer.h.',
        'llama': 'model.layers.', 'mistral': 'model.layers.', 'gemma': 'model.layers.',
        'qwen2': 'model.layers.', 'phi': 'model.layers.', 'phi3': 'model.layers.',
        'smollm2': 'model.layers.', 'stablelm': 'model.layers.', 'cohere': 'model.layers.',
        'opt': 'model.decoder.layers.',
    }
    if mt in prefixes:
        return prefixes[mt]
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return 'transformer.h.'
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return 'model.layers.'
    if hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        return 'model.decoder.layers.'
    return 'transformer.h.'

def _get_lm_head(model):
    if hasattr(model, 'lm_head'):
        return model.lm_head
    if hasattr(model, 'embed_out'):
        return model.embed_out
    if hasattr(model, 'output_projection'):
        return model.output_projection

def _get_final_norm(model):
    mt = getattr(model.config, 'model_type', '').lower()
    if mt == 'gpt2' and hasattr(model.transformer, 'ln_f'):
        return model.transformer.ln_f
    if mt in ('llama', 'mistral', 'qwen2', 'phi', 'phi3', 'gemma', 'smollm2', 'cohere'):
        if hasattr(model.model, 'norm'):
            return model.model.norm
    if mt == 'opt' and hasattr(model.model.decoder, 'final_layer_norm'):
        return model.model.decoder.final_layer_norm
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
        return model.transformer.ln_f
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        return model.model.norm
    if hasattr(model, 'model') and hasattr(model.model, 'final_layer_norm'):
        return model.model.final_layer_norm
    return None


# ─── UTILITIES ───────────────────────────────────────────────────────────

def load_texts(n=64):
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation", trust_remote_code=True)
        return [t["text"].strip() for t in ds if len(t["text"].strip()) > 10][:n]
    except Exception:
        return ["The quick brown fox jumps over the lazy dog."] * max(n, 4)

def ppl(model, ids, mask=None):
    model.eval()
    if mask is None: mask = (ids > 0).long()
    loss = model(input_ids=ids, attention_mask=mask, labels=ids).loss
    return math.exp(loss.item())

def clean():
    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

proportional_map = utils.proportional_map
svd_project = utils.svd_project
build_token_map = utils.build_token_map


# ─── SVD PROJECT (keep local reference) ─────────────────────────────────

def svd_project(W, out_t, in_t):
    if W.shape == (out_t, in_t): return W
    W = W.float(); U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    k = min(W.shape[0], W.shape[1], out_t, in_t)
    W2 = torch.zeros(out_t, in_t, dtype=W.dtype, device=W.device)
    W2[:U[:,:k].shape[0], :Vh[:k].shape[1]] = (U[:,:k] @ torch.diag(S[:k]) @ Vh[:k])[:out_t, :in_t]
    return W2.to(dtype=W.dtype)

def build_token_map(tok_a, tok_b):
    id_map = {}
    for i in range(tok_a.vocab_size):
        s = tok_a.decode([i]).strip()
        if not s: id_map[i] = 0; continue
        bid = tok_b.encode(s, add_special_tokens=False)
        id_map[i] = bid[0] if bid else 0
    return id_map


# ═══════════════════════════════════════════════════════════════════════════
# SOLUTION 1 — Activation-similarity-guided merge (same arch, diff sizes)
# ═══════════════════════════════════════════════════════════════════════════

def hsic_cka(h_a, h_b):
    """Proper HSIC Centered Kernel Alignment.
    h_a, h_b: [batch*seq, d_a], [batch*seq, d_b] — the hidden states.
    Returns scalar similarity in [0, 1].
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
    def __init__(self, model, n_layers):
        self.hiddens = {}
        self.handles = []
        layer_list = _get_layer_list(model)
        for i in range(n_layers):
            self.handles.append(
                layer_list[i].register_forward_hook(self._hook(i))
            )

    def _hook(self, i):
        def fn(_, inp, out):
            h = (out[0] if isinstance(out, tuple) else out).float()
            self.hiddens[i] = h.cpu()
        return fn

    def collect(self, model, ids, mask):
        self.hiddens = {}
        model(ids, attention_mask=mask)
        return self.hiddens

    def close(self):
        for h in self.handles: h.remove()


def activation_similarity(h_a, h_b):
    """CKA between two hidden states from different models.
    h_a, h_b: [batch, seq, d_a], [batch, seq, d_b] or [*, d].
    Reshapes to 2D and computes HSIC CKA.
    """
    ha_2d = h_a.reshape(-1, h_a.shape[-1])
    hb_2d = h_b.reshape(-1, h_b.shape[-1])
    min_n = min(ha_2d.shape[0], hb_2d.shape[0])
    return hsic_cka(ha_2d[:min_n], hb_2d[:min_n])


def _compute_procrustes_rotation(model_a, model_b, mapping, ha, hb):
    """Compute orthogonal rotation R aligning B's representation space to A's.

    Uses hidden states (activations) with Tikhonov regularization to handle
    rank-deficiency when the number of tokens < hidden dimension.
    """
    last_a = max(ha.keys()) if ha else None
    last_b = max(hb.keys()) if hb else None
    if last_a is None or last_b is None:
        return None

    h_a = ha[last_a].float().reshape(-1, ha[last_a].shape[-1])
    h_b = hb[last_b].float().reshape(-1, hb[last_b].shape[-1])

    d_h = h_a.shape[1]
    if h_b.shape[1] != d_h:
        return None

    C = h_b.T @ h_a
    trace_C = torch.trace(C).abs().item()
    lambda_reg = max(1e-4 * trace_C / d_h, 1e-6)
    C_reg = C + lambda_reg * torch.eye(d_h, device=C.device)

    U, _, Vt = torch.linalg.svd(C_reg, full_matrices=False)
    R = U @ Vt
    if torch.det(R) < 0:
        Vt[-1] *= -1
        R = U @ Vt
    return R


def _apply_procrustes_alignment(model_a, model_b, mapping, b_proj, ha, hb):
    """Align B's weights to A's coordinate system via orthogonal Procrustes.

    Computes rotation R from hidden states (with Tikhonov regularization),
    then rotates all B weight matrices to match A's coordinate system.
    """
    d_h = utils.hidden_dim(model_a.config)
    R = _compute_procrustes_rotation(model_a, model_b, mapping, ha, hb)
    if R is None:
        return b_proj

    aligned = {}
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

    return aligned


def merge_same_arch(model_a, model_b, calib_texts=None, save_name="merged_same_arch", use_procrustes=False):
    tok = AutoTokenizer.from_pretrained(model_a.config._name_or_path if hasattr(model_a.config, '_name_or_path') else "gpt2")
    tok.pad_token = tok.eos_token
    if calib_texts is None: calib_texts = load_texts(32)

    enc = tok(calib_texts[:24], truncation=True, padding=True, max_length=128, return_tensors="pt")
    ids, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)

    n_a = _get_n_layers(model_a.config)
    n_b = _get_n_layers(model_b.config)

    print("  Computing layer activation similarities...")
    cka_a = CkaComputer(model_a, n_a)
    cka_b = CkaComputer(model_b, n_b)
    ha = cka_a.collect(model_a, ids, mask)
    hb = cka_b.collect(model_b, ids, mask)
    cka_a.close(); cka_b.close()

    sim_matrix = {}
    for i_a in range(n_a):
        for i_b in range(n_b):
            sim_matrix[(i_a, i_b)] = activation_similarity(
                ha.get(i_a, torch.zeros(1)), hb.get(i_b, torch.zeros(1))
            ).item()

    mapping = {}
    sim_vals = {}
    used_b = set()
    for i_a in range(n_a):
        candidates = [(i_b, sim_matrix[(i_a, i_b)]) for i_b in range(n_b) if i_b not in used_b]
        if not candidates:
            candidates = [(i_b, sim_matrix[(i_a, i_b)]) for i_b in range(n_b)]
        i_b, _ = max(candidates, key=lambda x: x[1])
        mapping[i_a] = i_b
        sim_vals[i_a] = sim_matrix[(i_a, i_b)]
        used_b.add(i_b)

    avg_sim = np.mean(list(sim_vals.values()))
    print(f"  Avg similarity: {avg_sim:.3f}")

    print("  Aligning representations via orthogonal Procrustes..." if use_procrustes else "  Projecting B weights...")
    b_proj = _project_b_weights(model_a, model_b, mapping)
    if use_procrustes:
        b_proj = _apply_procrustes_alignment(model_a, model_b, mapping, b_proj, ha, hb)

    alphas = {}
    for i in range(n_a):
        c = sim_vals.get(i, 0.5)
        alphas[i] = max(0.3, min(0.9, 0.8 - 0.4 * c))

    merged_sd = _apply_merge(model_a, model_b, mapping, alphas, b_proj)

    m = AutoModelForCausalLM.from_config(model_a.config).to(DEVICE)
    m.load_state_dict(merged_sd, strict=False)
    best_ppl = ppl(m, ids, mask)
    best_alphas = dict(alphas)

    ppl_pure_a = ppl(model_a, ids, mask)
    print(f"  Init PPL: {best_ppl:.1f} | Pure-A PPL: {ppl_pure_a:.1f} | Refining...")
    if ppl_pure_a < best_ppl:
        best_ppl = ppl_pure_a
        best_alphas = {i: 1.0 for i in range(n_a)}
        merged_sd = _apply_merge(model_a, model_b, mapping, best_alphas, b_proj)
        m.load_state_dict(merged_sd, strict=False)

    for phase, steps in [("coarse", [0.25, -0.25]), ("fine", [0.1, -0.1]), ("ultra", [0.5, -0.5, 0.05, -0.05])]:
        for layer in range(n_a):
            orig = best_alphas[layer]
            best_a, best_p = orig, best_ppl
            for delta in steps:
                cand = max(0.0, min(1.0, orig + delta))
                test = dict(best_alphas); test[layer] = cand
                sd = _apply_merge(model_a, model_b, mapping, test, b_proj)
                m.load_state_dict(sd, strict=False)
                p = ppl(m, ids, mask)
                if p < best_p:
                    best_p = p; best_a = cand
            if best_a != orig: best_alphas[layer] = best_a; best_ppl = best_p

    merged_sd = _apply_merge(model_a, model_b, mapping, best_alphas, b_proj)
    merged = AutoModelForCausalLM.from_config(model_a.config).to(DEVICE)
    merged.load_state_dict(merged_sd, strict=False)
    final_ppl = ppl(merged, ids, mask)

    if save_name is not None:
        save_dir = os.path.join(SAVE_DIR, save_name)
        os.makedirs(save_dir, exist_ok=True)
        merged.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        with open(os.path.join(save_dir, "merge_info.json"), "w") as f:
            json.dump({"alphas": {str(k): round(v, 3) for k, v in best_alphas.items()},
                        "final_ppl": round(final_ppl, 1),
                        "avg_similarity": round(avg_sim, 3),
                        "type": "same_arch_different_size"}, f, indent=2)
        print(f"  [OK] Saved to {save_dir}/")

    print(f"  [OK] Final PPL: {final_ppl:.1f}")
    return merged, tok


def _spectral_repair(merged_sd, sd_a, strength=0.3):
    repaired = {}
    for k, v in merged_sd.items():
        if k in sd_a and v.ndim == 2 and v.shape == sd_a[k].shape:
            v_f, a_f = v.float(), sd_a[k].float()
            U_m, S_m, Vh_m = torch.linalg.svd(v_f, full_matrices=False)
            U_a, S_a, Vh_a = torch.linalg.svd(a_f, full_matrices=False)
            S_r = strength * S_a.to(S_m.device) + (1 - strength) * S_m
            repaired[k] = (U_m @ torch.diag(S_r) @ Vh_m).to(v.dtype)
        else:
            repaired[k] = v.clone()
    return repaired


def _project_b_weights(model_a, model_b, mapping):
    prefix_a = _get_layer_prefix(model_a)
    prefix_b = _get_layer_prefix(model_b)
    sd_a, sd_b = model_a.state_dict(), model_b.state_dict()
    layers_a = {}
    for k, v in sd_a.items():
        if k.startswith(prefix_a):
            parts = k[len(prefix_a):].split(".")
            idx, local = int(parts[0]), ".".join(parts[1:])
            layers_a.setdefault(idx, {})[local] = v
    proj = {}
    for i_a, i_b in mapping.items():
        for local, w_a in layers_a[i_a].items():
            bk = f"{prefix_b}{i_b}.{local}"
            if bk in sd_b:
                w_b = sd_b[bk].float()
                if w_a.shape == w_b.shape:
                    proj[(i_a, local)] = w_b
                elif w_a.dim() == 2:
                    proj[(i_a, local)] = svd_project(w_b, *w_a.shape)
                elif w_a.dim() == 1:
                    proj[(i_a, local)] = F.interpolate(
                        w_b.view(1, 1, -1), size=w_a.shape[0], mode='linear'
                    ).view(-1)
    for k in sd_a:
        if not k.startswith(prefix_a) and k in sd_b:
            if sd_a[k].shape == sd_b[k].shape:
                proj[k] = sd_b[k].float()
    return proj


_EMBED_OUTPUT_TOKENS = {"wte", "wpe", "embed_tokens", "embed_positions", "lm_head", "output_projection"}

def _is_embed_or_output_key(key):
    stem = key.replace(".weight", "").replace(".bias", "")
    parts = set(stem.split("."))
    return bool(parts & _EMBED_OUTPUT_TOKENS)

def _apply_merge(model_a, model_b, mapping, alphas, b_proj):
    prefix = _get_layer_prefix(model_a)
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
                key = (i_a, local)
                if key in b_proj:
                    sd[k] = (a * v.float() + (1 - a) * b_proj[key]).to(v.dtype)
    return sd


# ═══════════════════════════════════════════════════════════════════════════
# SOLUTION 2 — Zero-init bridge (diff arch)
# ═══════════════════════════════════════════════════════════════════════════

class OptimalBridge(nn.Module):
    def __init__(self, d_a, d_b):
        super().__init__()
        self.proj = nn.Linear(d_b, d_a, bias=False)
        nn.init.zeros_(self.proj.weight)
    def forward(self, h_a, h_b):
        return h_a + self.proj(h_b)


class MLPBridge(nn.Module):
    """Non-linear residual bridge for cross-architecture merging.
    Adds a small MLP on top of the linear projection for more capacity.
    All output paths zero-init so bridge starts as identity (h_a unchanged).
    """
    def __init__(self, d_a, d_b, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or min(d_a, 256)
        self.linear = nn.Linear(d_b, d_a, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_b, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_a, bias=False),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.mlp[-1].weight)

    def forward(self, h_a, h_b):
        return h_a + self.linear(h_b) + self.mlp(h_b)


def build_bridge(ma, mb, tok, texts, token_map=None, bridge_type="linear"):
    d_a, d_b = utils.hidden_dim(ma.config), utils.hidden_dim(mb.config)
    if bridge_type == "mlp":
        bridge = MLPBridge(d_a, d_b)
    else:
        bridge = OptimalBridge(d_a, d_b)
    return bridge.to(DEVICE)


def _stitch_forward(ma, mb, bridge, ids, mask, labels, dtype, token_map=None):
    lm_head = _get_lm_head(ma)
    assert lm_head is not None, f"Cannot find lm_head in {type(ma).__name__}"
    with torch.no_grad():
        ids_b = torch.tensor([[token_map.get(i.item(), 0) for i in row] for row in ids],
                              device=DEVICE) if token_map else ids
        oa, ob = ma(ids, attention_mask=mask, output_hidden_states=True), mb(ids_b, attention_mask=mask, output_hidden_states=True)
        ha, hb = oa.hidden_states[-1].float(), ob.hidden_states[-1].float()
        k = min(ha.shape[1], hb.shape[1])
    hf = bridge(ha[:, :k], hb[:, :k])
    logits = lm_head(hf.to(dtype))
    sl, ll = logits[..., :-1, :].contiguous(), labels[:, :k][..., 1:].contiguous()
    loss = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1), ignore_index=-100)
    return loss, logits


def _holdout_split(texts, eval_pct=0.2):
    n = len(texts)
    n_eval = max(min(int(n * eval_pct), n // 2), min(4, n // 2))
    if n_eval < 1:
        return texts, []
    return texts[:-n_eval], texts[-n_eval:]


def _eval_bridge_ppl(ma, mb, bridge, tok, eval_texts, token_map=None, max_len=128):
    if not eval_texts:
        return float("inf")
    dtype = next(ma.parameters()).dtype
    enc = tok(eval_texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    ids, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)
    loss, _ = _stitch_forward(ma, mb, bridge, ids, mask, ids, dtype, token_map)
    return math.exp(loss.item())


def train_bridge_v2(ma, mb, tok, texts, token_map=None, steps=10, lr=3e-4,
                    weight_decay=0.01, max_len=128, eval_texts=None,
                    bridge_type="linear"):
    d_a, d_b = utils.hidden_dim(ma.config), utils.hidden_dim(mb.config)
    if bridge_type == "mlp":
        bridge = MLPBridge(d_a, d_b)
    else:
        bridge = OptimalBridge(d_a, d_b)
    bridge.to(DEVICE)

    if eval_texts is not None:
        train_texts = texts
    else:
        train_texts, eval_texts = texts, None

    if not train_texts:
        train_texts = texts

    enc = tok(train_texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    ids, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)

    opt = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    best = None; best_loss = float("inf")
    model_dtype = next(ma.parameters()).dtype

    torch.set_grad_enabled(True)
    bridge.train()
    for s in range(steps):
        opt.zero_grad()
        loss, _ = _stitch_forward(ma, mb, bridge, ids, mask, ids, model_dtype, token_map)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        opt.step()
        sched.step()
        if loss.item() < best_loss:
            best_loss = loss.item(); best = copy.deepcopy(bridge.state_dict())
        if (s+1) % 5 == 0:
            cur_lr = sched.get_last_lr()[0]
            print(f"    Step {s+1}/{steps} loss={loss.item():.4f} lr={cur_lr:.2e}")

    bridge.load_state_dict(best if best else bridge.state_dict())
    bridge.eval()
    torch.set_grad_enabled(False)

    eval_ppl = _eval_bridge_ppl(ma, mb, bridge, tok, eval_texts, token_map, max_len)
    if math.isfinite(eval_ppl):
        print(f"    Eval PPL (held-out): {eval_ppl:.1f}")
    else:
        print(f"    (no held-out eval texts)")
    bridge.eval_ppl = eval_ppl
    return bridge


def _cache_hidden_states(ma, mb, tok, texts, token_map=None, max_len=128):
    lm_head = _get_lm_head(ma)
    assert lm_head is not None
    enc = tok(texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    ids, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)

    ids_b = torch.tensor(
        [[token_map.get(i.item(), 0) for i in row] for row in ids],
        device=DEVICE
    ) if token_map else ids

    with torch.no_grad():
        oa = ma(ids, attention_mask=mask, output_hidden_states=True)
        ob = mb(ids_b, attention_mask=mask, output_hidden_states=True)
        ha = oa.hidden_states[-1].float()
        hb = ob.hidden_states[-1].float()
        k = min(ha.shape[1], hb.shape[1])

    return {
        "ha": ha[:, :k], "hb": hb[:, :k],
        "ids": ids[:, :k], "lm_head": lm_head,
        "dtype": next(ma.parameters()).dtype,
    }


def train_bridge_cached(ma, mb, tok, texts, token_map=None, steps=20, lr=3e-4,
                         weight_decay=0.01, max_len=128, eval_texts=None,
                         bridge_type="linear"):
    if eval_texts is not None:
        train_texts = texts
    else:
        train_texts, eval_texts = texts, None
    if not train_texts:
        train_texts = texts

    d_a, d_b = utils.hidden_dim(ma.config), utils.hidden_dim(mb.config)
    if bridge_type == "mlp":
        bridge = MLPBridge(d_a, d_b).to(DEVICE)
    else:
        bridge = OptimalBridge(d_a, d_b).to(DEVICE)
    cache = _cache_hidden_states(ma, mb, tok, train_texts, token_map, max_len)
    ha, hb, ids = cache["ha"], cache["hb"], cache["ids"]
    lm_head, dtype = cache["lm_head"], cache["dtype"]

    opt = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    best = None; best_loss = float("inf")

    torch.set_grad_enabled(True)
    bridge.train()
    for s in range(steps):
        opt.zero_grad()
        hf = bridge(ha, hb)
        logits = lm_head(hf.to(dtype))
        sl, ll = logits[..., :-1, :].contiguous(), ids[..., 1:].contiguous()
        loss = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        opt.step()
        sched.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best = copy.deepcopy(bridge.state_dict())
        if (s + 1) % 5 == 0:
            cur_lr = sched.get_last_lr()[0]
            print(f"    Step {s+1}/{steps} loss={loss.item():.4f} lr={cur_lr:.2e}")

    bridge.load_state_dict(best if best else bridge.state_dict())
    bridge.eval()
    torch.set_grad_enabled(False)

    eval_ppl = _eval_bridge_ppl(ma, mb, bridge, tok, eval_texts, token_map, max_len)
    if math.isfinite(eval_ppl):
        print(f"    Eval PPL (held-out): {eval_ppl:.1f}")
    else:
        print(f"    (no held-out eval texts)")
    bridge.eval_ppl = eval_ppl
    return bridge


def merge_same_arch_bridge(model_a, model_b, tok, calib_texts, steps=10, lr=3e-4, save_name=None, bridge_type="linear"):
    n_a = _get_n_layers(model_a.config)
    n_b = _get_n_layers(model_b.config)
    print(f"  Same-arch bridge: {n_a} layers (A) + {n_b} layers (B), {steps} steps")

    bridge = train_bridge_v2(model_a, model_b, tok, calib_texts, steps=steps, lr=lr, bridge_type=bridge_type)

    ppl_val = getattr(bridge, "eval_ppl", float("inf"))
    if math.isfinite(ppl_val):
        print(f"  Bridge PPL: {ppl_val:.1f}")

    if save_name is not None:
        save_dir = os.path.join(SAVE_DIR, save_name)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(bridge.state_dict(), os.path.join(save_dir, "bridge.pt"))
        tok.save_pretrained(save_dir)
        json.dump({
            "type": "same_arch_bridge",
            "d_a": utils.hidden_dim(model_a.config), "d_b": utils.hidden_dim(model_b.config),
            "n_layers_a": n_a, "n_layers_b": n_b,
            "final_ppl": round(ppl_val, 1),
        }, open(os.path.join(save_dir, "bridge_config.json"), "w"), indent=2)
        print(f"  [OK] Saved to {save_dir}/")

    return bridge, tok


@torch.no_grad()
def stitch_generate(ma, mb, bridge, tok, prompt, max_new=30, token_map=None):
    ma.eval(); mb.eval(); bridge.eval()
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    for _ in range(max_new):
        ids_b = torch.tensor([[token_map.get(i.item(), 0) for i in row] for row in ids],
                              device=DEVICE) if token_map else ids
        oa, ob = ma(ids, output_hidden_states=True), mb(ids_b, output_hidden_states=True)
        ha, hb = oa.hidden_states[-1].float(), ob.hidden_states[-1].float()
        k = min(ha.shape[1], hb.shape[1])
        hf = bridge(ha[:, :k], hb[:, :k])
        dtype = next(ma.parameters()).dtype
        lm_head = _get_lm_head(ma)
        logits = lm_head(hf.to(dtype))[:, -1, :] / 0.8
        ids = torch.cat([ids, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=-1)
    return tok.decode(ids[0], skip_special_tokens=True)


@torch.no_grad()
def generate_bridge(model_a, model_b, bridge, tok, prompt, max_new=50,
                    token_map=None, mix_alpha=0.3, temp=0.9):
    model_a.eval(); model_b.eval(); bridge.eval()
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    for _ in range(max_new):
        ids_mapped = torch.tensor([[token_map.get(i.item(), 0) for i in row] for row in ids],
                                   device=DEVICE) if token_map else ids
        oa = model_a(ids, output_hidden_states=True)
        ob = model_b(ids_mapped, output_hidden_states=True)
        ha, hb = oa.hidden_states[-1].float(), ob.hidden_states[-1].float()
        k = min(ha.shape[1], hb.shape[1])
        h_bridge = bridge(ha[:, :k], hb[:, :k])
        hf = ha[:, :k] + mix_alpha * (h_bridge - ha[:, :k])
        dtype = next(model_a.parameters()).dtype
        lm_head = _get_lm_head(model_a)
        logits = lm_head(hf.to(dtype))[:, -1, :] / temp
        probs = F.softmax(logits, dim=-1)
        ids = torch.cat([ids, torch.multinomial(probs, 1)], dim=-1)

    return tok.decode(ids[0], skip_special_tokens=True)


def merge_diff_arch(model_a, model_b, calib_texts=None, token_map=None,
                    save_name="merged_diff_arch", tok=None,
                    steps=10, lr=3e-4, weight_decay=0.01, max_len=128,
                    bridge_type="linear"):
    if tok is None:
        tok = AutoTokenizer.from_pretrained(
            model_a.config._name_or_path if hasattr(model_a.config, '_name_or_path') else "distilgpt2"
        )
        tok.pad_token = tok.eos_token

    if calib_texts is None: calib_texts = load_texts(48)

    enc = tok(calib_texts[:16], truncation=True, padding=True, max_length=64, return_tensors="pt")
    ids, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)
    pp_a = ppl(model_a, ids, mask)
    print(f"  A PPL: {pp_a:.1f}")

    print(f"  Training bridge ({steps} steps, lr={lr:.0e}, wd={weight_decay})...")
    bridge = train_bridge_v2(model_a, model_b, tok, calib_texts,
                             token_map=token_map, steps=steps, lr=lr,
                             weight_decay=weight_decay, max_len=max_len,
                             bridge_type=bridge_type)

    dtype = next(model_a.parameters()).dtype
    with torch.no_grad():
        ids_mapped = torch.tensor([[token_map.get(i.item(), 0) for i in row] for row in ids],
                                   device=DEVICE) if token_map else ids
        oa = model_a(ids, attention_mask=mask, output_hidden_states=True)
        ob = model_b(ids_mapped, attention_mask=mask, output_hidden_states=True)
        ha, hb = oa.hidden_states[-1].float(), ob.hidden_states[-1].float()
        k = min(ha.shape[1], hb.shape[1])
        hf = bridge(ha[:, :k], hb[:, :k])
        lm_head = _get_lm_head(model_a)
        logits = lm_head(hf.to(dtype))
        sl, ll = logits[..., :-1, :].contiguous(), ids[:, :k][..., 1:].contiguous()
        loss = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1), ignore_index=-100)
        b_ppl = math.exp(loss.item())
    print(f"  Bridge PPL: {b_ppl:.1f} {'<-- BETTER' if b_ppl < pp_a else ''}")

    bridge_dir = os.path.join(SAVE_DIR, save_name)
    os.makedirs(bridge_dir, exist_ok=True)
    torch.save(bridge.state_dict(), os.path.join(bridge_dir, "bridge.pt"))
    tok.save_pretrained(bridge_dir)
    with open(os.path.join(bridge_dir, "bridge_config.json"), "w") as f:
        json.dump({
            "d_a": utils.hidden_dim(model_a.config),
            "d_b": utils.hidden_dim(model_b.config),
            "model_a": model_a.config._name_or_path if hasattr(model_a.config, '_name_or_path') else "unknown",
            "model_b": model_b.config._name_or_path if hasattr(model_b.config, '_name_or_path') else "unknown",
            "ppl_a": round(pp_a, 1),
            "ppl_bridge": round(b_ppl, 1),
            "type": "diff_arch_bridge",
            "steps": steps, "lr": lr, "bridge_type": bridge_type,
        }, f, indent=2)
    print(f"  [OK] Saved to {bridge_dir}/")
    clean()
    return bridge


# ─── EVALUATION ──────────────────────────────────────────────────────────

def verify_generations(ma, mb, bridge, tok, prompts=None, token_map=None):
    if prompts is None:
        prompts = ["The future of AI is", "In the beginning,"]
    for p in prompts:
        g = stitch_generate(ma, mb, bridge, tok, p, token_map=token_map)
        print(f"  [{p}] -> {g[:100]}")

def load_merged(bridge_dir, model_a, model_b, tok=None):
    d_a = utils.hidden_dim(model_a.config)
    d_b = utils.hidden_dim(model_b.config)
    config_path = os.path.join(bridge_dir, "bridge_config.json")
    bridge_type = "linear"
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
            bridge_type = cfg.get("bridge_type", "linear")
    if bridge_type == "mlp":
        bridge = MLPBridge(d_a, d_b).to(DEVICE)
    else:
        bridge = OptimalBridge(d_a, d_b).to(DEVICE)
    bridge.load_state_dict(torch.load(os.path.join(bridge_dir, "bridge.pt"), map_location=DEVICE))
    bridge.eval()
    if tok is None:
        tok = AutoTokenizer.from_pretrained(bridge_dir)
    return bridge, tok
