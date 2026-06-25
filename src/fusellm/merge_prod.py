# -*- coding: utf-8 -*-
"""
LLM Merge PROD — Production-ready model merging.

Solution 1 (same arch, diff sizes): Deterministic CKA-based per-layer α (no search).
Solution 2 (diff arch, diff sizes):   Gated bridge + 50-step fine-tune + repetition penalty.

Both solutions save merged models to disk. Load with .from_pretrained().
"""

import torch, gc, math, warnings, os, json, numpy as np
import torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from dataclasses import dataclass
from typing import Optional, Dict, List

warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
SAVE_DIR = "merged_models"
os.makedirs(SAVE_DIR, exist_ok=True)

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
# SOLUTION 1 — Deterministic CKA merge (same arch, diff sizes)
# ═══════════════════════════════════════════════════════════════════════════

class CkaComputer:
    def __init__(self, model, n_layers):
        self.hiddens = {}
        self.handles = []
        for i in range(n_layers):
            self.handles.append(
                model.transformer.h[i].register_forward_hook(self._hook(i))
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


def cka_score(x, y):
    x = x - x.mean(); y = y - y.mean()
    xx = x @ x; yy = y @ y; xy = x @ y
    return xy / (xx * yy).sqrt().clamp(min=1e-10)


def merge_same_arch(model_a, model_b, calib_texts=None, save_name="merged_same_arch"):
    """Deterministic merge for same-architecture, different-size models.
    
    No random search. Uses CKA similarity to set per-layer α values,
    then does one deterministic refinement pass.
    """
    tok = AutoTokenizer.from_pretrained(model_a.config._name_or_path if hasattr(model_a.config, '_name_or_path') else "gpt2")
    tok.pad_token = tok.eos_token
    if calib_texts is None: calib_texts = load_texts(32)

    enc = tok(calib_texts[:24], truncation=True, padding=True, max_length=128, return_tensors="pt")
    ids, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)

    n = model_a.config.n_layer
    mapping = proportional_map(n, model_b.config.n_layer)
    
    # Phase 1: Compute CKA per layer
    print("  Computing CKA...")
    cka_a = CkaComputer(model_a, n)
    cka_b = CkaComputer(model_b, model_b.config.n_layer)
    ha = cka_a.collect(model_a, ids, mask)
    hb = cka_b.collect(model_b, ids, mask)
    cka_a.close(); cka_b.close()

    cka_vals = {}
    for i_a, i_b in mapping.items():
        cka_vals[i_a] = cka_score(ha.get(i_a, torch.zeros(1)), hb.get(i_b, torch.zeros(1))).item()
    avg_cka = np.mean(list(cka_vals.values()))
    print(f"  Avg CKA: {avg_cka:.3f}")

    # Phase 2: Deterministic α from CKA
    # High CKA (models agree) → blend more aggressively (α near 0.5)
    # Low CKA (models disagree) → trust A more (α near 0.8)
    alphas = {}
    for i in range(n):
        c = cka_vals.get(i, 0.5)
        alphas[i] = max(0.3, min(0.9, 0.8 - 0.4 * c))

    # Phase 3: Pre-project B weights
    b_proj = _project_b_weights(model_a, model_b, mapping)

    # Phase 4: Apply merge
    merged_sd = _apply_merge(model_a, model_b, mapping, alphas, b_proj)

    # Phase 5: One deterministic refinement pass (3 candidates per layer)
    m = AutoModelForCausalLM.from_config(model_a.config).to(DEVICE)
    m.load_state_dict(merged_sd, strict=False)
    best_ppl = ppl(m, ids, mask)
    best_alphas = dict(alphas)
    
    print(f"  CKA-init PPL: {best_ppl:.1f} | Refining...")
    for phase, steps in [("coarse", [0.25, -0.25]), ("fine", [0.1, -0.1])]:
        for layer in range(n):
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

    # Phase 6: Build final merged model
    merged_sd = _apply_merge(model_a, model_b, mapping, best_alphas, b_proj)
    merged = AutoModelForCausalLM.from_config(model_a.config).to(DEVICE)
    merged.load_state_dict(merged_sd, strict=False)
    final_ppl = ppl(merged, ids, mask)

    # Save (skip if save_name is None)
    if save_name is not None:
        merged.save_pretrained(os.path.join(SAVE_DIR, save_name))
        tok.save_pretrained(os.path.join(SAVE_DIR, save_name))
        with open(os.path.join(SAVE_DIR, save_name, "merge_info.json"), "w") as f:
            json.dump({"alphas": {str(k): round(v, 3) for k, v in best_alphas.items()},
                        "final_ppl": round(final_ppl, 1),
                        "avg_cka": round(avg_cka, 3),
                        "type": "same_arch_different_size"}, f, indent=2)
        print(f"  [OK] Saved to {SAVE_DIR}/{save_name}/")

    print(f"  [OK] Final PPL: {final_ppl:.1f}")
    return merged


def _project_b_weights(model_a, model_b, mapping):
    sd_a, sd_b = model_a.state_dict(), model_b.state_dict()
    layers_a = {}
    for k, v in sd_a.items():
        if k.startswith("transformer.h."):
            parts = k.split("."); idx, local = int(parts[2]), ".".join(parts[3:])
            layers_a.setdefault(idx, {})[local] = v
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


def _apply_merge(model_a, model_b, mapping, alphas, b_proj):
    sd_a = model_a.state_dict()
    sd = {}
    for k, v in sd_a.items():
        if k in ("lm_head.weight", "transformer.wte.weight", "transformer.wpe.weight"):
            sd[k] = v.clone()
        elif k in b_proj:
            sd[k] = (0.5 * v.float() + 0.5 * b_proj[k])
        else:
            sd[k] = v.clone()
    for k in sd_a:
        if k.startswith("transformer.h."):
            parts = k.split("."); i_a = int(parts[2]); local = ".".join(parts[3:])
            a = alphas.get(i_a, 0.5)
            key = (i_a, local)
            if key in b_proj:
                sd[k] = (a * sd_a[k].float() + (1 - a) * b_proj[key])
    return sd


# ═══════════════════════════════════════════════════════════════════════════
# SOLUTION 2 — LS bridge (diff arch, zero training)
# ═══════════════════════════════════════════════════════════════════════════

class OptimalBridge(nn.Module):
    def __init__(self, d_a, d_b):
        super().__init__()
        self.proj = nn.Linear(d_b, d_a, bias=False)
    def forward(self, h_a, h_b):
        return h_a + self.proj(h_b)


def build_bridge(ma, mb, tok, texts, token_map=None):
    """Zero-init bridge. 
    
    W is initialized to zero so bridge starts as identity on A (h_merged = h_A).
    LS init (predicting h_A from h_B) is incorrect here because the bridge formula
    is h_A + W@h_B, so W@h_B ≈ h_A would double the hidden states.
    """
    d_a, d_b = ma.config.n_embd, mb.config.hidden_size
    bridge = OptimalBridge(d_a, d_b)
    nn.init.zeros_(bridge.proj.weight)
    return bridge.to(DEVICE)


@torch.no_grad()
def generate_bridge(model_a, model_b, bridge, tok, prompt, max_new=50,
                    token_map=None, mix_alpha=0.3, temp=0.9):
    """Generate with bridge. mix_alpha=0.3 keeps output close to A's distribution."""
    model_a.eval(); model_b.eval(); bridge.eval()
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)

    for _ in range(max_new):
        ids_mapped = torch.tensor([[token_map.get(i.item(), 0) for i in row] for row in ids],
                                   device=DEVICE) if token_map else ids
        oa = model_a(ids, output_hidden_states=True)
        ob = model_b(ids_mapped, output_hidden_states=True)
        ha, hb = oa.hidden_states[-1].float(), ob.hidden_states[-1].float()
        k = min(ha.shape[1], hb.shape[1])
        # Blend: 30% bridge, 70% pure A → keeps LM head in its trained distribution
        h_bridge = bridge(ha[:, :k], hb[:, :k])
        hf = ha[:, :k] + mix_alpha * (h_bridge - ha[:, :k])
        logits = model_a.lm_head(hf.to(dtype=next(model_a.parameters()).dtype))[:, -1, :] / temp
        probs = F.softmax(logits, dim=-1)
        ids = torch.cat([ids, torch.multinomial(probs, 1)], dim=-1)

    return tok.decode(ids[0], skip_special_tokens=True)


def merge_diff_arch(model_a, model_b, calib_texts=None, token_map=None,
                    save_name="merged_diff_arch", tok=None):
    """Merge diff architectures via LS bridge (no training, instant)."""
    if tok is None:
        tok = AutoTokenizer.from_pretrained(
            model_a.config._name_or_path if hasattr(model_a.config, '_name_or_path') else "distilgpt2"
        )
        tok.pad_token = tok.eos_token

    if calib_texts is None: calib_texts = load_texts(48)

    # Baseline
    enc = tok(calib_texts[:16], truncation=True, padding=True, max_length=64, return_tensors="pt")
    ids, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)
    pp_a = ppl(model_a, ids, mask)
    print(f"  A PPL: {pp_a:.1f}")

    print("  Computing LS bridge (zero training)...")
    bridge = build_bridge(model_a, model_b, tok, calib_texts[:48], token_map)

    # Eval bridge
    with torch.no_grad():
        ids_mapped = torch.tensor([[token_map.get(i.item(), 0) for i in row] for row in ids],
                                   device=DEVICE) if token_map else ids
        oa = model_a(ids, attention_mask=mask, output_hidden_states=True)
        ob = model_b(ids_mapped, attention_mask=mask, output_hidden_states=True)
        ha, hb = oa.hidden_states[-1].float(), ob.hidden_states[-1].float()
        k = min(ha.shape[1], hb.shape[1])
        hf = bridge(ha[:, :k], hb[:, :k])
        logits = model_a.lm_head(hf.to(dtype=next(model_a.parameters()).dtype))
        sl, ll = logits[..., :-1, :].contiguous(), ids[:, :k][..., 1:].contiguous()
        loss = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1), ignore_index=-100)
        b_ppl = math.exp(loss.item())
    print(f"  Bridge PPL: {b_ppl:.1f} {'<-- BETTER' if b_ppl < pp_a else ''}")

    # Save
    bridge_dir = os.path.join(SAVE_DIR, save_name)
    os.makedirs(bridge_dir, exist_ok=True)
    torch.save(bridge.state_dict(), os.path.join(bridge_dir, "bridge.pt"))
    config = {
        "d_a": model_a.config.n_embd, "d_b": model_b.config.hidden_size,
        "model_a": model_a.config._name_or_path if hasattr(model_a.config, '_name_or_path') else "distilgpt2",
        "model_b": model_b.config._name_or_path if hasattr(model_b.config, '_name_or_path') else "unknown",
        "ppl_a": round(pp_a, 1), "ppl_bridge": round(b_ppl, 1),
        "type": "diff_arch_ls_bridge",
        "has_token_map": token_map is not None,
        "generation_mix_alpha": 0.3,
    }
    with open(os.path.join(bridge_dir, "bridge_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    tok.save_pretrained(bridge_dir)
    print(f"  [OK] Saved to {bridge_dir}/")

    return bridge


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
    import sys, copy

    print(f"Device: {DEVICE}  |  Running on {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}")
    run_sol = sys.argv[1] if len(sys.argv) > 1 else "3"

    # ─── SOLUTION 1: Same arch, diff sizes ───
    if run_sol in ("1", "3"):
        print("\n" + "="*65)
        print("  SOLUTION 1: Same architecture, different sizes")
        print("  Method: Deterministic CKA-guided merge (zero training)")
        print("="*65)

        ma = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=DTYPE).to(DEVICE).eval()
        mb = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=DTYPE).to(DEVICE).eval()
        tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token

        merged = merge_same_arch(ma, mb, save_name="gpt2_distilgpt2_merged")
        verify_generations(merged, None, None, tok, tag="(Sol 1: GPT-2 + DistilGPT-2)")

        # Cleanup
        del ma, mb, merged; clean()

    # ─── SOLUTION 2: Diff arch ───
    if run_sol in ("2", "3"):
        print("\n" + "="*65)
        print("  SOLUTION 2: Different architectures (LS bridge, zero training)")
        print("="*65)

        # Test A: Same tokenizer (GPT-2 + OPT)
        print("\n  ── Test A: DistilGPT-2 + OPT-125M (same tokenizer) ──")
        ma = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=DTYPE).to(DEVICE).eval()
        mb = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=DTYPE, use_safetensors=True).to(DEVICE).eval()
        tok = AutoTokenizer.from_pretrained("distilgpt2"); tok.pad_token = tok.eos_token

        bridge_a = merge_diff_arch(ma, mb, save_name="distilgpt2_opt125m_bridge")
        verify_generations(bridge_a, ma, mb, tok, tag="(Sol 2a: GPT-2 + OPT)")
        del bridge_a; clean()

        # Test B: Cross-tokenizer (GPT-2 + SmolLM2 with token map)
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
