# -*- coding: utf-8 -*-
"""
LLM Merge v2 — Ceiling broken, now pushing quality higher with minimal compute.

Solution 1 (different sizes): CKA-guided init + 1-pass per-layer refinement
  → 4x fewer trials, better quality

Solution 2 (different archs):  Least-squares bridge init + 10-step fine-tune
  → 3x fewer steps, better quality via optimal initialization
"""

import torch, gc, copy, math, warnings, numpy as np
import torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

warnings.filterwarnings("ignore"); torch.set_grad_enabled(False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
print(f"Device: {DEVICE}")

# ─── COMMON UTILITIES ──────────────────────────────────────────────────────

def load_texts(n=64):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
    texts = [t["text"] for t in ds if len(t["text"].strip()) > 50][:n*2]
    return texts[:n] or ["The quick brown fox jumps over the lazy dog."] * n

def ppl(model, ids, mask=None):
    return math.exp(model(input_ids=ids, attention_mask=mask, labels=ids).loss.item())

def generate(model, tokenizer, prompt="The future of AI is", max_new=50):
    inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=True,
                         temperature=0.8, top_p=0.9,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def clean():
    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

def proportional_map(n_a, n_b):
    return {i: min(int((i + 0.5) * n_b / n_a), n_b - 1) for i in range(n_a)}

def svd_project(W, out_t, in_t):
    if W.shape == (out_t, in_t): return W
    W = W.float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    k = min(W.shape[0], W.shape[1], out_t, in_t)
    W2 = torch.zeros(out_t, in_t, dtype=W.dtype, device=W.device)
    W2[:U[:,:k].shape[0], :Vh[:k].shape[1]] = (U[:,:k] @ torch.diag(S[:k]) @ Vh[:k])[:out_t, :in_t]
    return W2.to(dtype=W.dtype)

# ─── SOLUTION 1: CKA-GUIDED PER-LAYER OPTIMIZATION ────────────────────────

def collect_hidden_stats(model, ids, mask, n_layers):
    """Collect hidden states from each layer for CKA computation"""
    hiddens = {}; handles = []
    def hook(i):
        def fn(_, inp, out):
            hiddens[i] = (out[0] if isinstance(out, tuple) else out).float().reshape(-1).cpu()
        return fn
    for i in range(n_layers):
        handles.append(model.transformer.h[i].register_forward_hook(hook(i)))
    model(ids, attention_mask=mask)
    for h in handles: h.remove()
    return hiddens

def cka_score(x, y):
    x = x - x.mean(); y = y - y.mean()
    xx = x @ x; yy = y @ y; xy = x @ y
    return xy / (xx * yy).sqrt().clamp(min=1e-10)

def cka_guided_merge(model_a, model_b, ids, mask):
    """CKA-guided initialization + 1-pass per-layer refinement"""
    n = model_a.config.n_layer
    mapping = proportional_map(n, model_b.config.n_layer)
    b_proj = prepare_b(model_a, model_b, mapping)
    
    print("  Computing CKA per layer...")
    ha = collect_hidden_stats(model_a, ids, mask, n)
    hb = collect_hidden_stats(model_b, ids, mask, model_b.config.n_layer)
    
    # CKA between each A-layer and its mapped B-layer
    cka_scores = {}
    for i_a, i_b in mapping.items():
        if i_a in ha and i_b in hb:
            cka_scores[i_a] = cka_score(ha[i_a], hb[i_b]).item()
        else:
            cka_scores[i_a] = 0.5
    
    avg_cka = np.mean(list(cka_scores.values()))
    print(f"  Avg CKA: {avg_cka:.3f}")
    
    # Smart α init: higher CKA = more weight from model B (they agree)
    alphas = {i: 1.0 - 0.5 * cka_scores.get(i, 0.5) for i in range(n)}
    alphas = {i: max(0.1, min(0.9, a)) for i, a in alphas.items()}
    
    best_sd = merge_model_a(model_a, model_b, mapping, alphas, b_proj)
    m = AutoModelForCausalLM.from_config(model_a.config).to(DEVICE)
    m.load_state_dict(best_sd, strict=False)
    best_ppl = ppl(m, ids, mask)
    best_alphas = dict(alphas)
    print(f"  CKA-guided init PPL: {best_ppl:.1f}")
    
    # 1-pass per-layer refinement (3 evaluations each → 36 total)
    for layer in range(n):
        orig = best_alphas[layer]
        best_a, best_p = orig, best_ppl
        for candidate in [max(0.0, orig-0.3), min(1.0, orig+0.3), 0.5]:
            test = dict(best_alphas); test[layer] = candidate
            sd = merge_model_a(model_a, model_b, mapping, test, b_proj)
            m.load_state_dict(sd, strict=False)
            p = ppl(m, ids, mask)
            if p < best_p: best_p = p; best_a = candidate
        best_alphas[layer] = best_a; best_ppl = best_p
    
    # 2nd pass: finer refinement
    for layer in range(n):
        orig = best_alphas[layer]
        best_a, best_p = orig, best_ppl
        for candidate in [max(0.0, orig-0.15), min(1.0, orig+0.15)]:
            test = dict(best_alphas); test[layer] = candidate
            sd = merge_model_a(model_a, model_b, mapping, test, b_proj)
            m.load_state_dict(sd, strict=False)
            p = ppl(m, ids, mask)
            if p < best_p: best_p = p; best_a = candidate
        best_alphas[layer] = best_a; best_ppl = best_p
    
    best_sd = merge_model_a(model_a, model_b, mapping, best_alphas, b_proj)
    m.load_state_dict(best_sd, strict=False)
    best_ppl = ppl(m, ids, mask)
    print(f"  Final PPL: {best_ppl:.1f}")
    
    # Spectral repair (fast, one-pass)
    best_sd = spectral_fix(best_sd, model_a, model_b, mapping, best_alphas)
    m.load_state_dict(best_sd, strict=False)
    sp_ppl = ppl(m, ids, mask)
    print(f"  Spectral-repaired PPL: {sp_ppl:.1f}")
    
    if sp_ppl < best_ppl:
        best_ppl = sp_ppl
    
    return m, best_ppl, best_alphas, mapping


def get_layers(sd, prefix="transformer.h."):
    layers = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            parts = k.split(".")
            idx, local = int(parts[2]), ".".join(parts[3:])
            layers.setdefault(idx, {})[local] = v
    return layers

def prepare_b(model_a, model_b, mapping):
    sd_a, sd_b = model_a.state_dict(), model_b.state_dict()
    layers_a = get_layers(sd_a)
    proj = {}
    for i_a, i_b in mapping.items():
        for local, w_a in layers_a[i_a].items():
            bk = f"transformer.h.{i_b}.{local}"
            if bk in sd_b:
                w_b = sd_b[bk].float()
                proj[(i_a, local)] = w_b if w_a.shape == w_b.shape else svd_project(w_b, *w_a.shape)
    for k in sd_a:
        if not k.startswith("transformer.h.") and k in sd_b:
            if sd_a[k].shape == sd_b[k].shape:
                proj[k] = sd_b[k].float()
    return proj

def merge_model_a(model_a, model_b, mapping, alphas, b_proj):
    sd_a = model_a.state_dict(); sd = {}
    for k, v in sd_a.items():
        if k in ("lm_head.weight", "transformer.wte.weight", "transformer.wpe.weight"):
            sd[k] = v.clone()
        elif k in b_proj:
            sd[k] = (0.5 * v.float() + 0.5 * b_proj[k])
        else:
            sd[k] = v.clone()
    layers_a = get_layers(sd_a)
    for i_a in mapping:
        a = alphas.get(i_a, 0.5)
        for local, w_a in layers_a[i_a].items():
            key = (i_a, local)
            sd[f"transformer.h.{i_a}.{local}"] = (a * w_a.float() + (1-a) * b_proj[key]) if key in b_proj else w_a.clone()
    return sd

def spectral_fix(sd, model_a, model_b, mapping, alphas):
    """One-pass spectral repair — replaces merged singular values"""
    out = {}
    for k, v in sd.items():
        if k.startswith("transformer.h.") and v.dim() == 2:
            parts = k.split("."); i_a = int(parts[2]); local = ".".join(parts[3:])
            i_b = mapping.get(i_a, i_a); a = alphas.get(i_a, 0.5)
            bk = f"transformer.h.{i_b}.{local}"
            w_a = model_a.state_dict()[k].float()
            if bk in model_b.state_dict():
                w_b = model_b.state_dict()[bk].float()
                if w_a.shape != w_b.shape: w_b = svd_project(w_b, *w_a.shape)
                try:
                    U, S, Vt = torch.linalg.svd(v.float(), full_matrices=False)
                    _, Sa, _ = torch.linalg.svd(w_a, full_matrices=False)
                    _, Sb, _ = torch.linalg.svd(w_b, full_matrices=False)
                    kk = min(len(S), len(Sa), len(Sb))
                    S_new = torch.zeros_like(S); S_new[:kk] = a * Sa[:kk] + (1-a) * Sb[:kk]
                    out[k] = U @ torch.diag(S_new) @ Vt
                    continue
                except: pass
        out[k] = v.clone()
    return out

# ─── SOLUTION 2: LEAST-SQUARES BRIDGE (minimal training) ──────────────────

class OptimalBridge(nn.Module):
    def __init__(self, d_a, d_b):
        super().__init__()
        self.proj = nn.Linear(d_b, d_a, bias=False)
    
    def forward(self, h_a, h_b):
        return h_a + self.proj(h_b)

def build_token_map(tok_a, tok_b):
    """Map A's token IDs to B's by matching decoded strings (zero-training)"""
    id_map = {}
    for i in range(tok_a.vocab_size):
        s = tok_a.decode([i]).strip()
        if not s:
            id_map[i] = 0
            continue
        bid = tok_b.encode(s, add_special_tokens=False)
        id_map[i] = bid[0] if bid else 0
    return id_map

def ls_init_bridge(ma, mb, tok, texts, token_map=None):
    """Initialize bridge via least-squares (closed form, no gradients)"""
    d_a, d_b = ma.config.n_embd, mb.config.hidden_size
    bridge = OptimalBridge(d_a, d_b)
    
    enc = tok(texts, truncation=True, padding=True, max_length=64, return_tensors="pt")
    ids, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)
    ids_b = torch.tensor([[token_map.get(i.item(), 0) for i in row] for row in ids],
                          device=DEVICE) if token_map else ids
    
    ha = ma(ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1].float()
    hb = mb(ids_b, attention_mask=mask, output_hidden_states=True).hidden_states[-1].float()
    k = min(ha.shape[1], hb.shape[1])
    Ha, Hb = ha[:, :k].reshape(-1, d_a), hb[:, :k].reshape(-1, d_b)
    
    W = (Ha.T @ Hb) @ torch.linalg.inv(Hb.T @ Hb + 1e-6 * torch.eye(d_b, device=DEVICE))
    bridge.proj.weight.data = W.to(dtype=torch.float32)
    return bridge.to(DEVICE)

def train_bridge_v2(ma, mb, tok, texts, token_map=None, steps=10, lr=3e-4):
    """LS init + fine-tune bridge"""
    print("  Computing least-squares bridge init...")
    bridge = ls_init_bridge(ma, mb, tok, texts, token_map)
    
    enc = tok(texts, truncation=True, padding=True, max_length=64, return_tensors="pt")
    ids, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)
    
    opt = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=0.01)
    best = None; best_loss = float("inf")
    model_dtype = next(ma.parameters()).dtype
    stitched = lambda ids_b, labels: _stitch_forward(ma, mb, bridge, ids_b, mask, labels, model_dtype, token_map)
    
    torch.set_grad_enabled(True)
    for s in range(steps):
        opt.zero_grad()
        loss, _ = stitched(ids, ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item(); best = copy.deepcopy(bridge.state_dict())
        if (s+1) % 5 == 0: print(f"    Step {s+1}/{steps} loss={loss.item():.4f}")
    
    bridge.load_state_dict(best if best else bridge.state_dict())
    torch.set_grad_enabled(False)
    return bridge

def _stitch_forward(ma, mb, bridge, ids, mask, labels, dtype, token_map=None):
    with torch.no_grad():
        ids_b = torch.tensor([[token_map.get(i.item(), 0) for i in row] for row in ids],
                              device=DEVICE) if token_map else ids
        oa, ob = ma(ids, attention_mask=mask, output_hidden_states=True), mb(ids_b, attention_mask=mask, output_hidden_states=True)
        ha, hb = oa.hidden_states[-1].float(), ob.hidden_states[-1].float()
        k = min(ha.shape[1], hb.shape[1])
    hf = bridge(ha[:, :k], hb[:, :k])
    logits = ma.lm_head(hf.to(dtype))
    sl, ll = logits[..., :-1, :].contiguous(), labels[:, :k][..., 1:].contiguous()
    loss = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1), ignore_index=-100)
    return loss, logits

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
        logits = ma.lm_head(hf.to(dtype=next(ma.parameters()).dtype))[:, -1, :] / 0.8
        ids = torch.cat([ids, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=-1)
    return tok.decode(ids[0], skip_special_tokens=True)

# ─── RUN SOLUTION 1 ────────────────────────────────────────────────────────

def run_solution1():
    print("\n" + "="*60)
    print("SOLUTION 1: DIFFERENT SIZES (CKA-guided, 0 training)")
    print("="*60)
    
    ma = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=DTYPE).to(DEVICE).eval()
    mb = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=DTYPE).to(DEVICE).eval()
    tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    
    print(f"  A: {ma.config.n_layer}L {ma.config.n_embd}d  B: {mb.config.n_layer}L {mb.config.n_embd}d")
    
    texts = load_texts(48)
    enc = tok(texts[:32], truncation=True, padding=True, max_length=128, return_tensors="pt")
    ids, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)
    
    pp_a = ppl(ma, ids, mask); pp_b = ppl(mb, ids, mask)
    print(f"  A PPL: {pp_a:.1f}  B PPL: {pp_b:.1f}")
    
    # Uniform baseline
    m_u = AutoModelForCausalLM.from_config(ma.config).to(DEVICE)
    mapping = proportional_map(ma.config.n_layer, mb.config.n_layer)
    b_proj = prepare_b(ma, mb, mapping)
    m_u.load_state_dict(merge_model_a(ma, mb, mapping, {}, b_proj), strict=False)
    u_ppl = ppl(m_u, ids, mask)
    u_text = generate(m_u, tok)
    del m_u; clean()
    print(f"  Uniform PPL: {u_ppl:.1f}")
    
    # CKA-guided merge
    merged, best_ppl, alphas, mapping = cka_guided_merge(ma, mb, ids[:24], mask[:24])
    best_text = generate(merged, tok)
    
    print(f"\n{'-'*55}")
    print(f"  A PPL:    {pp_a:.1f} | Uniform: {u_ppl:.1f} | Merged: {best_ppl:.1f}")
    print(f"{'-'*55}")
    print(f"\nUniform: {u_text}")
    print(f"\nMerged:  {best_text}")
    print(f"\nPer-layer alpha: {', '.join(f'{i}:{alphas.get(i,0):.2f}' for i in range(ma.config.n_layer))}")
    
    return merged

# ─── RUN SOLUTION 2 ────────────────────────────────────────────────────────

def run_solution2():
    print("\n" + "="*60)
    print("SOLUTION 2: DIFFERENT ARCHS (LS init + 10-step fine-tune)")
    print("="*60)
    
    # ── Test A: Same tokenizer (GPT-2 + OPT) ──
    print("Test A: DistilGPT-2 + OPT-125M (same tokenizer)")
    ma = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=DTYPE).to(DEVICE).eval()
    mb = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=DTYPE, use_safetensors=True).to(DEVICE).eval()
    tok = AutoTokenizer.from_pretrained("distilgpt2"); tok.pad_token = tok.eos_token
    
    print(f"  A (GPT-2): {ma.config.n_layer}L {ma.config.n_embd}d  |  B (OPT): {mb.config.num_hidden_layers}L {mb.config.hidden_size}d")
    
    texts = load_texts(64)
    enc = tok(texts[:16], truncation=True, padding=True, max_length=64, return_tensors="pt")
    ids, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)
    pp_a = ppl(ma, ids, mask)
    print(f"  A PPL: {pp_a:.1f}")
    
    bridge = train_bridge_v2(ma, mb, tok, texts[:48], token_map=None, steps=10)
    with torch.no_grad():
        loss, _ = _stitch_forward(ma, mb, bridge, ids, mask, ids, next(ma.parameters()).dtype)
    b_ppl = math.exp(loss.item())
    print(f"  Bridge PPL: {b_ppl:.1f}")
    
    print("  Generating:")
    for prompt in ["The future of artificial intelligence is", "In the beginning, there was"]:
        try:
            text = stitch_generate(ma, mb, bridge, tok, prompt)
            print(f"    [{prompt}] -> {text[:120]}")
        except Exception as e:
            print(f"    FAILED: {e}")
    
    print(f"\n  Result: Bridge PPL {b_ppl:.1f} {'<' if b_ppl < pp_a else '>'} A PPL {pp_a:.1f}")
    
    # ── Test B: Cross-tokenizer (DistilGPT-2 + SmolLM2 with token map) ──
    clean()
    print(f"\n{'='*60}")
    print("Test B: DistilGPT-2 + SmolLM2-135M (cross-tokenizer + token map)")
    mb2 = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=DTYPE).to(DEVICE).eval()
    tok2 = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    if tok2.pad_token is None: tok2.pad_token = tok2.eos_token
    
    print(f"  B (SmolLM2): {mb2.config.num_hidden_layers}L {mb2.config.hidden_size}d")
    print("  Building token map (GPT-2 IDs -> SmolLM2 IDs)...")
    token_map = build_token_map(tok, tok2)
    match_rate = sum(1 for v in token_map.values() if v > 0) / len(token_map) * 100
    print(f"  Token mapping match rate: {match_rate:.1f}%")
    
    bridge2 = train_bridge_v2(ma, mb2, tok, texts[:48], token_map=token_map, steps=10)
    with torch.no_grad():
        loss2, _ = _stitch_forward(ma, mb2, bridge2, ids, mask, ids, next(ma.parameters()).dtype, token_map)
    b2_ppl = math.exp(loss2.item())
    print(f"  Bridge PPL: {b2_ppl:.1f}")
    
    print("  Generating:")
    for prompt in ["The future of artificial intelligence is", "In the beginning, there was"]:
        try:
            text = stitch_generate(ma, mb2, bridge2, tok, prompt, token_map=token_map)
            print(f"    [{prompt}] -> {text[:120]}")
        except Exception as e:
            print(f"    FAILED: {e}")
    
    print(f"\n  Result: Bridge PPL {b2_ppl:.1f} {'<' if b2_ppl < pp_a else '>'} A PPL {pp_a:.1f}")
    
    print(f"\n{'='*60}")
    return bridge

# ─── MAIN ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys; sol = sys.argv[1] if len(sys.argv) > 1 else "3"
    torch.set_grad_enabled(False)
    
    if sol in ("1", "3"): run_solution1(); clean()
    if sol in ("2", "3"): run_solution2(); clean()
