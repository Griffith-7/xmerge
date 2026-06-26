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
    "OptimalBridge", "CkaComputer", "activation_similarity",
    "merge_same_arch", "merge_same_arch_bridge", "build_bridge", "train_bridge_v2",
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
    for name, mod in model.named_modules():
        if 'lm_head' in name.lower() or 'embed_out' in name.lower():
            if isinstance(mod, nn.Linear):
                return mod
    return None


# ─── HELPERS ───────────────────────────────────────────────────────────────

def load_texts(n=64):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t["text"] for t in ds if len(t["text"].strip()) > 50][:n*2]
    return texts[:n] or ["The quick brown fox jumps over the lazy dog."] * n

@torch.no_grad()
def ppl(model, ids, mask=None):
    return math.exp(model(input_ids=ids, attention_mask=mask, labels=ids).loss.item())

def clean():
    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

def _get_final_norm(model):
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
        return model.transformer.ln_f
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        return model.model.norm
    if hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'final_layer_norm'):
        return model.model.decoder.final_layer_norm
    if hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'norm'):
        return model.model.decoder.norm
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'norm'):
        return model.base_model.norm
    return None

def proportional_map(n_a, n_b):
    return {i: min(int((i + 0.5) * n_b / n_a), n_b - 1) for i in range(n_a)}

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
            self.hiddens[i] = (out[0] if isinstance(out, tuple) else out).float().reshape(-1).cpu()
        return fn

    def collect(self, model, ids, mask):
        self.hiddens = {}
        model(ids, attention_mask=mask)
        return self.hiddens

    def close(self):
        for h in self.handles: h.remove()


def activation_similarity(x, y):
    x = x - x.mean(); y = y - y.mean()
    xx = x @ x; yy = y @ y; xy = x @ y
    return xy / (xx * yy).sqrt().clamp(min=1e-10)


def merge_same_arch(model_a, model_b, calib_texts=None, save_name="merged_same_arch"):
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

    alphas = {}
    for i in range(n_a):
        c = sim_vals.get(i, 0.5)
        alphas[i] = max(0.3, min(0.9, 0.8 - 0.4 * c))

    b_proj = _project_b_weights(model_a, model_b, mapping)
    merged_sd = _apply_merge(model_a, model_b, mapping, alphas, b_proj)

    m = AutoModelForCausalLM.from_config(model_a.config).to(DEVICE)
    m.load_state_dict(merged_sd, strict=False)
    best_ppl = ppl(m, ids, mask)
    best_alphas = dict(alphas)

    # Check pure A baseline — if blending a bad parent only makes things worse
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
                proj[(i_a, local)] = w_b if w_a.shape == w_b.shape else svd_project(w_b, *w_a.shape)
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
    sd = {}
    for k, v in sd_a.items():
        if _is_embed_or_output_key(k):
            sd[k] = v.clone()
        elif k.startswith(prefix):
            sd[k] = v.clone()
        elif k in b_proj:
            avg_alpha = sum(alphas.values()) / max(len(alphas), 1)
            sd[k] = (avg_alpha * v.float() + (1 - avg_alpha) * b_proj[k])
        else:
            sd[k] = v.clone()
    for k in sd_a:
        if k.startswith(prefix):
            rest = k[len(prefix):]
            parts = rest.split(".")
            i_a = int(parts[0])
            local = ".".join(parts[1:])
            a = alphas.get(i_a, 0.5)
            key = (i_a, local)
            if key in b_proj:
                sd[k] = (a * sd_a[k].float() + (1 - a) * b_proj[key])
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


def build_bridge(ma, mb, tok, texts, token_map=None):
    d_a, d_b = utils.hidden_dim(ma.config), utils.hidden_dim(mb.config)
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


def train_bridge_v2(ma, mb, tok, texts, token_map=None, steps=10, lr=3e-4, weight_decay=0.01, max_len=128):
    d_a, d_b = utils.hidden_dim(ma.config), utils.hidden_dim(mb.config)
    bridge = OptimalBridge(d_a, d_b)
    bridge.to(DEVICE)

    enc = tok(texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
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
    return bridge


def merge_same_arch_bridge(model_a, model_b, tok, calib_texts, steps=10, lr=3e-4, save_name=None):
    n_a = _get_n_layers(model_a.config)
    n_b = _get_n_layers(model_b.config)
    print(f"  Same-arch bridge: {n_a} layers (A) + {n_b} layers (B), {steps} steps")

    bridge = train_bridge_v2(model_a, model_b, tok, calib_texts, steps=steps, lr=lr)

    enc = tok(calib_texts[:16], truncation=True, padding=True, max_length=64, return_tensors="pt")
    ids = enc.input_ids.to(DEVICE)
    mask = enc.attention_mask.to(DEVICE)
    dtype = next(model_a.parameters()).dtype
    loss, _ = _stitch_forward(model_a, model_b, bridge, ids, mask, ids, dtype)
    ppl_val = math.exp(loss.item())
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
        lm_head = _get_lm_head(ma)
        logits = lm_head(hf.to(dtype=next(ma.parameters()).dtype))[:, -1, :] / 0.8
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
        lm_head = _get_lm_head(model_a)
        logits = lm_head(hf.to(dtype=next(model_a.parameters()).dtype))[:, -1, :] / temp
        probs = F.softmax(logits, dim=-1)
        ids = torch.cat([ids, torch.multinomial(probs, 1)], dim=-1)

    return tok.decode(ids[0], skip_special_tokens=True)


def merge_diff_arch(model_a, model_b, calib_texts=None, token_map=None,
                    save_name="merged_diff_arch", tok=None,
                    steps=10, lr=3e-4, weight_decay=0.01, max_len=128):
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
                             weight_decay=weight_decay, max_len=max_len)

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
    config = {
        "d_a": utils.hidden_dim(model_a.config), "d_b": utils.hidden_dim(model_b.config),
        "model_a": model_a.config._name_or_path if hasattr(model_a.config, '_name_or_path') else "distilgpt2",
        "model_b": model_b.config._name_or_path if hasattr(model_b.config, '_name_or_path') else "unknown",
        "ppl_a": round(pp_a, 1), "ppl_bridge": round(b_ppl, 1),
        "type": "diff_arch_trained_bridge",
        "steps": steps, "lr": lr, "weight_decay": weight_decay,
        "has_token_map": token_map is not None,
        "generation_mix_alpha": 0.3,
    }
    with open(os.path.join(bridge_dir, "bridge_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    tok.save_pretrained(bridge_dir)
    print(f"  [OK] Saved to {bridge_dir}/")

    return bridge


# ═══════════════════════════════════════════════════════════════════════════
# LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_merged(save_name):
    """Load a previously saved merged model or bridge.

    For same-arch merges: returns (model, tokenizer, config_dict)
    For diff-arch bridges: returns (bridge, tokenizer, config_dict)
    """
    save_dir = os.path.join(SAVE_DIR, save_name)
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"Saved model not found at {save_dir}")

    if os.path.exists(os.path.join(save_dir, "merge_info.json")):
        # Same-arch merge
        with open(os.path.join(save_dir, "merge_info.json")) as f:
            info = json.load(f)
        model = AutoModelForCausalLM.from_pretrained(save_dir, torch_dtype=DTYPE).to(DEVICE).eval()
        tok = AutoTokenizer.from_pretrained(save_dir)
        return model, tok, info

    if os.path.exists(os.path.join(save_dir, "bridge_config.json")):
        # Diff-arch bridge
        with open(os.path.join(save_dir, "bridge_config.json")) as f:
            info = json.load(f)
        bridge = OptimalBridge(info["d_a"], info["d_b"])  # zero init
        state = torch.load(os.path.join(save_dir, "bridge.pt"), map_location=DEVICE, weights_only=True)
        bridge.load_state_dict(state)
        bridge.to(DEVICE).eval()
        tok = AutoTokenizer.from_pretrained(save_dir)
        return bridge, tok, info

    raise ValueError(f"Unknown format in {save_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════

EVAL_PROMPTS = [
    "The future of artificial intelligence is",
    "In the beginning, there was",
    "The meaning of life is",
    "Once upon a time in a",
    "The most important thing to remember is",
]

def verify_generations(model_or_bridge, model_a, model_b, tok, token_map=None, tag=""):
    print(f"\n  {'='*50}")
    print(f"  Generation samples {tag}")
    print(f"  {'='*50}")
    for prompt in EVAL_PROMPTS:
        try:
            if isinstance(model_or_bridge, nn.Module) and hasattr(model_or_bridge, 'proj'):
                text = generate_bridge(model_a, model_b, model_or_bridge, tok, prompt,
                                       token_map=token_map)
            else:
                inp = tok(prompt, return_tensors="pt").to(DEVICE)
                out = model_or_bridge.generate(**inp, max_new_tokens=50, do_sample=True,
                                                temperature=0.8, top_p=0.9, top_k=40,
                                                repetition_penalty=1.1,
                                                pad_token_id=tok.pad_token_id or tok.eos_token_id)
                text = tok.decode(out[0], skip_special_tokens=True)
            print(f"  [{prompt}]\n    -> {text}\n")
        except Exception as e:
            print(f"  [{prompt}]\n    -> FAILED: {e}\n")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print(f"Device: {DEVICE}  |  Running on {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}")
    run_sol = sys.argv[1] if len(sys.argv) > 1 else "3"

    if run_sol in ("1", "3"):
        print("\n" + "="*65)
        print("  SOLUTION 1: Same architecture, different sizes")
        print("  Method: Activation-similarity-guided merge (zero training)")
        print("="*65)

        ma = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=DTYPE).to(DEVICE).eval()
        mb = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=DTYPE).to(DEVICE).eval()
        tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token

        merged, _ = merge_same_arch(ma, mb, save_name="gpt2_distilgpt2_merged")
        verify_generations(merged, None, None, tok, tag="(Sol 1: GPT-2 + DistilGPT-2)")

        del ma, mb, merged; clean()

    if run_sol in ("2", "3"):
        print("\n" + "="*65)
        print("  SOLUTION 2: Different architectures (LS bridge, zero training)")
        print("="*65)

        print("\n  ── Test A: DistilGPT-2 + OPT-125M (same tokenizer) ──")
        ma = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=DTYPE).to(DEVICE).eval()
        mb = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=DTYPE, use_safetensors=True).to(DEVICE).eval()
        tok = AutoTokenizer.from_pretrained("distilgpt2"); tok.pad_token = tok.eos_token

        bridge_a = merge_diff_arch(ma, mb, save_name="distilgpt2_opt125m_bridge")
        verify_generations(bridge_a, ma, mb, tok, tag="(Sol 2a: GPT-2 + OPT)")
        del bridge_a; clean()

        print("\n  ── Test B: DistilGPT-2 + SmolLM2-135M (cross-tokenizer) ──")
        mb2 = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=DTYPE).to(DEVICE).eval()
        tok2 = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        if tok2.pad_token is None: tok2.pad_token = tok2.eos_token

        print("  Building token map...")
        token_map = build_token_map(tok, tok2)
        match_rate = sum(1 for v in token_map.values() if v > 0) / len(token_map) * 100
        print(f"  Token match rate: {match_rate:.1f}%")

        bridge_b = merge_diff_arch(ma, mb2, token_map=token_map,
                                   save_name="distilgpt2_smollm2_bridge")
        verify_generations(bridge_b, ma, mb2, tok, token_map=token_map,
                          tag="(Sol 2b: GPT-2 + SmolLM2)")
        del ma, mb, mb2, bridge_b; clean()

    print("\n" + "="*65)
    print("  [OK] All merges complete. Models saved to 'merged_models/'")
    print("="*65)
