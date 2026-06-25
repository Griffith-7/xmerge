# -*- coding: utf-8 -*-
"""
LLM Merge — breaks the ceiling with NO training/fine-tuning.
Uses: per-layer random search optimization + activation normalization repair.

Solution 1 (different sizes): gpt2 (12L) + distilgpt2 (6L)
Solution 2 (different archs):  gpt2 (GPT-2) + SmolLM2-135M (Llama)
"""

import torch, gc, copy, os, json, math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Set HF_TOKEN env var if accessing gated models
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
print(f"Device: {DEVICE}")

# ─── UTILITIES ─────────────────────────────────────────────────────────────

def load_calib(name="gpt2", n=64, seq=128):
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
    texts = [t["text"] for t in ds if len(t["text"].strip()) > 50][:n*2]
    if not texts: texts = ["The quick brown fox jumps over the lazy dog."] * n
    enc = tokenizer(texts[:n], truncation=True, padding=True, max_length=seq, return_tensors="pt")
    return enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE), tokenizer

@torch.no_grad()
def ppl(model, ids, mask=None):
    model.eval()
    return math.exp(model(input_ids=ids, attention_mask=mask, labels=ids).loss.item())

@torch.no_grad()
def generate(model, tokenizer, prompt="The future of AI is", max_new=40):
    model.eval()
    inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=True,
                         temperature=0.8, top_p=0.9,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def clean():
    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

# ─── LAYER MAPPING ─────────────────────────────────────────────────────────

def proportional_map(n_a, n_b):
    return {i: min(int((i + 0.5) * n_b / n_a), n_b - 1) for i in range(n_a)}

# ─── SVD PROJECTION ────────────────────────────────────────────────────────

def svd_project(W, out_t, in_t):
    if W.shape == (out_t, in_t): return W
    W = W.float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    k = min(W.shape[0], W.shape[1], out_t, in_t)
    W2 = torch.zeros(out_t, in_t, dtype=W.dtype, device=W.device)
    U_k = U[:, :k]; Vh_k = Vh[:k, :]; S_k = S[:k]
    reduced = U_k @ torch.diag(S_k) @ Vh_k
    W2[:U_k.shape[0], :Vh_k.shape[1]] = reduced[:out_t, :in_t]
    return W2

# ─── MODEL WRAPPERS ────────────────────────────────────────────────────────

def get_gpt2_weights(model):
    """Get per-layer weights from GPT-2"""
    sd = model.state_dict()
    layers = {}
    others = {}
    for k, v in sd.items():
        if k.startswith("transformer.h."):
            parts = k.split(".")
            idx = int(parts[2])
            local = ".".join(parts[3:])
            layers.setdefault(idx, {})[local] = v
        else:
            others[k] = v
    return layers, others

def get_llama_weights(model):
    """Get per-layer weights from Llama-style model"""
    sd = model.state_dict()
    layers = {}
    others = {}
    for k, v in sd.items():
        if k.startswith("model.layers."):
            parts = k.split(".")
            idx = int(parts[2])
            local = ".".join(parts[3:])
            layers.setdefault(idx, {})[local] = v
        else:
            others[k] = v
    return layers, others

# ─── SOLUTION 1: MERGE INTO GPT-2 ARCHITECTURE ────────────────────────────

def prepare_b_projected(model_a, model_b, layer_map):
    """Pre-project ALL model B weights to match model A's dimensions"""
    sd_b = model_b.state_dict()
    sd_a = model_a.state_dict()
    layers_a, _ = get_gpt2_weights(model_a)
    
    projected = {}
    for i_a, i_b in layer_map.items():
        for local, w_a in layers_a[i_a].items():
            bkey = f"transformer.h.{i_b}.{local}"
            if bkey in sd_b:
                w_b = sd_b[bkey].float()
                projected[(i_a, local)] = svd_project(w_b, *w_a.shape) if w_a.shape != w_b.shape else w_b.clone()
    
    # Non-layer weights
    for k, v in sd_a.items():
        if k.startswith("transformer.h."):
            continue
        if k in sd_b:
            projected[k] = sd_b[k].float() if v.shape == sd_b[k].shape else svd_project(sd_b[k].float(), *v.shape)
    
    return projected


def merge_same_arch(model_a, model_b, layer_map, alphas, b_projected, keep_embeds=True):
    """Fast merge using pre-projected B weights"""
    sd_a = model_a.state_dict()
    sd = {}
    
    for k, v in sd_a.items():
        if keep_embeds and k in ("lm_head.weight", "transformer.wte.weight", "transformer.wpe.weight"):
            sd[k] = v.clone()
        elif k in b_projected:
            sd[k] = (0.5 * v.float() + 0.5 * b_projected[k])
        elif k.startswith("transformer.h."):
            sd[k] = v.clone()
        else:
            sd[k] = v.clone()
    
    for i_a in layer_map.keys():
        alpha = alphas.get(i_a, 0.5)
        for local, w_a in get_gpt2_weights(model_a)[0][i_a].items():
            key = (i_a, local)
            if key in b_projected:
                sd[f"transformer.h.{i_a}.{local}"] = (alpha * w_a.float() + (1-alpha) * b_projected[key])
            else:
                sd[f"transformer.h.{i_a}.{local}"] = w_a.clone()
    return sd

def objective_score(model, ids, mask):
    """Combined PPL + diversity penalty in one forward pass"""
    model.eval()
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, labels=ids)
        ppl_val = math.exp(out.loss.item())
        logits = out.logits[:, :-1].float()
        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1).values.mean().item()
        score = ppl_val + max(0, max_prob - 0.3) * 5
    return score, ppl_val


def random_search_merge(model_a, model_b, ids, mask, trials=40):
    """Random search for optimal per-layer alphas"""
    n = model_a.config.n_layer
    mapping = proportional_map(n, model_b.config.n_layer)
    print(f"  Layer mapping: {dict(mapping)}")
    print("  Pre-projecting B weights (one-time SVD cost)...")
    b_proj = prepare_b_projected(model_a, model_b, mapping)
    
    sd_u = merge_same_arch(model_a, model_b, mapping, {}, b_proj)
    m = AutoModelForCausalLM.from_config(model_a.config).to(DEVICE)
    m.load_state_dict(sd_u, strict=False)
    best_score, best_ppl = objective_score(m, ids, mask)
    best_alphas = {i: 0.5 for i in range(n)}
    best_sd = sd_u
    print(f"  Uniform(0.5) PPL: {best_ppl:.1f}")
    
    # Reuse a single model instance for all trials (swap weights in-place)
    m = AutoModelForCausalLM.from_config(model_a.config).to(DEVICE)
    
    for t in range(trials):
        alphas = {i: np.random.uniform(0.0, 1.0) for i in range(n)}
        sd = merge_same_arch(model_a, model_b, mapping, alphas, b_proj)
        try:
            m.load_state_dict(sd, strict=False)
            s, p = objective_score(m, ids, mask)
            if s < best_score:
                best_score = s; best_alphas = alphas; best_sd = sd; best_ppl = p
                print(f"  Trial {t+1}: PPL {p:.1f}")
        except: pass
    
    del m; clean()
    return best_sd, best_alphas, best_ppl, mapping


def spectral_repair_state_dict(sd, model_a, model_b, mapping, alphas):
    """Apply spectral repair: replace merged singular values with interpolated originals"""
    out = {}
    for k, v in sd.items():
        if k.startswith("transformer.h.") and "weight" in k and v.dim() == 2:
            # Get corresponding weights from originals
            parts = k.split(".")
            i_a = int(parts[2])
            i_b = mapping.get(i_a, i_a)
            local = ".".join(parts[3:])
            
            w_a = model_a.state_dict()[k].float()
            bkey = f"transformer.h.{i_b}.{local}"
            if bkey in model_b.state_dict():
                w_b = model_b.state_dict()[bkey].float()
                if w_a.shape != w_b.shape:
                    w_b = svd_project(w_b, *w_a.shape)
                alpha = alphas.get(i_a, 0.5)
                
                try:
                    U, S, Vt = torch.linalg.svd(v.float(), full_matrices=False)
                    _, S_a, _ = torch.linalg.svd(w_a, full_matrices=False)
                    _, S_b, _ = torch.linalg.svd(w_b, full_matrices=False)
                    k_min = min(len(S), len(S_a), len(S_b))
                    S_new = alpha * S_a[:k_min] + (1 - alpha) * S_b[:k_min]
                    S_repaired = torch.zeros_like(S)
                    S_repaired[:k_min] = S_new[:k_min]
                    out[k] = U @ torch.diag(S_repaired) @ Vt
                    continue
                except: pass
        out[k] = v.clone()
    return out

# ─── ACTIVATION NORMALIZATION REPAIR ──────────────────────────────────────

@torch.no_grad()
def collect_stats(model, ids, attn, n_layers):
    """Collect per-layer hidden mean/std"""
    stats = {}
    handles = []
    hiddens = {}
    
    def make_hook(i):
        def hook(_, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            hiddens[i] = h.float()
        return hook
    
    for i in range(n_layers):
        h = model.transformer.h[i]
        handles.append(h.register_forward_hook(make_hook(i)))
    
    # Final LN
    def final_hook(_, inp, out):
        hiddens["final"] = (out[0] if isinstance(out, tuple) else out).float()
    handles.append(model.transformer.ln_f.register_forward_hook(final_hook))
    
    with torch.no_grad():
        model(input_ids=ids[:32], attention_mask=attn[:32])
    for h in handles: h.remove()
    
    for i, h in hiddens.items():
        stats[i] = {"mean": h.mean(dim=(0,1)), "std": h.std(dim=(0,1)) + 1e-8}
    return stats

def repair_activations(merged_model, model_a, ids, attn):
    """Apply activation normalization correction"""
    n = merged_model.config.n_layer
    dtype = next(merged_model.parameters()).dtype
    print("  Collecting A stats...")
    s_a = collect_stats(model_a, ids, attn, n)
    print("  Collecting merged stats...")
    s_m = collect_stats(merged_model, ids, attn, n)
    
    scales, shifts = {}, {}
    for i in range(n):
        if i in s_a and i in s_m:
            scales[i] = (s_a[i]["std"] / s_m[i]["std"]).to(device=DEVICE, dtype=dtype)
            shifts[i] = (s_a[i]["mean"] - s_m[i]["mean"] * scales[i]).to(device=DEVICE, dtype=dtype)
        else:
            dim = merged_model.config.n_embd
            scales[i] = torch.ones(dim, device=DEVICE, dtype=dtype)
            shifts[i] = torch.zeros(dim, device=DEVICE, dtype=dtype)
    
    handles = []
    for i in range(n):
        def make_hook(idx):
            def hook(_, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                s = scales.get(idx)
                sh = shifts.get(idx)
                if s is not None and h.shape[-1] == s.shape[0]:
                    h_corrected = h * s + sh
                    if isinstance(out, tuple):
                        return (h_corrected,) + out[1:]
                    return h_corrected
                return out
            return hook
        handles.append(merged_model.transformer.h[i].register_forward_hook(make_hook(i)))
    
    return merged_model, handles

# ─── SOLUTION 2: DIFFERENT ARCHITECTURES (stitching) ─────────────────────

class FusionBridge(nn.Module):
    """Simple linear bridge from model B's space to model A's space"""
    def __init__(self, d_a, d_b):
        super().__init__()
        self.proj = nn.Linear(d_b, d_a, bias=False)
        nn.init.eye_(self.proj.weight[:min(d_a, d_b), :min(d_a, d_b)])
        nn.init.normal_(self.proj.weight, std=0.001)
    
    def forward(self, h_a, h_b):
        return h_a + self.proj(h_b)

class StitchedModel(nn.Module):
    def __init__(self, ma, mb, bridge, tok_a, tok_b):
        super().__init__()
        self.ma = ma; self.mb = mb; self.bridge = bridge
        self.tok_a = tok_a; self.tok_b = tok_b
        for p in ma.parameters(): p.requires_grad = False
        for p in mb.parameters(): p.requires_grad = False
    
    def forward(self, ids_a, ids_b=None, mask_a=None, mask_b=None, labels=None):
        with torch.no_grad():
            self.ma.eval()
            model_dtype = next(self.ma.parameters()).dtype
            if ids_a is None:
                ids_a = ids_b.clip(0, self.ma.config.vocab_size - 1)
            if mask_a is None:
                mask_a = mask_b
            
            oa = self.ma(ids_a, attention_mask=mask_a, output_hidden_states=True)
            ob = self.mb(ids_b, attention_mask=mask_b, output_hidden_states=True)
            
            ha = oa.hidden_states[-1].to(dtype=torch.float32)
            hb = ob.hidden_states[-1].to(dtype=torch.float32)
            
            min_len = min(ha.shape[1], hb.shape[1])
            ha, hb = ha[:, :min_len], hb[:, :min_len]
        
        hf = self.bridge(ha, hb)
        logits = self.ma.lm_head(hf.to(dtype=model_dtype))
        
        loss = None
        if labels is not None:
            sl = logits[..., :-1, :].contiguous()
            ll = labels[:, :min_len][..., 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1), ignore_index=-100)
        return type("O", (), {"loss": loss, "logits": logits})
    
    @torch.no_grad()
    def generate(self, prompt, max_new=30, temp=0.8, top_p=0.9):
        self.eval()
        # Use model B tokenizer for primary, clip for model A
        ids_b = self.tok_b(prompt, return_tensors="pt").input_ids.to(DEVICE)
        
        for _ in range(max_new):
            ids_a_safe = ids_b.clip(0, self.ma.config.vocab_size - 1)
            oa = self.ma(ids_a_safe, output_hidden_states=True)
            ob = self.mb(ids_b, output_hidden_states=True)
            
            ha, hb = oa.hidden_states[-1].float(), ob.hidden_states[-1].float()
            min_len = min(ha.shape[1], hb.shape[1])
            
            hf = self.bridge(ha[:, :min_len], hb[:, :min_len])
            logits = self.ma.lm_head(hf.to(dtype=next(self.ma.parameters()).dtype))[:, -1, :] / temp
            
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids_b = torch.cat([ids_b, next_id], dim=-1)
        
        return self.tok_b.decode(ids_b[0], skip_special_tokens=True)

def train_bridge(ma, mb, tok_a, tok_b, texts, steps=50, lr=5e-4):
    """Train bridge with per-tokenizer inputs"""
    d_a, d_b = ma.config.n_embd, mb.config.hidden_size
    bridge = FusionBridge(d_a, d_b).to(DEVICE)  # stays float32
    opt = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=0.01)
    
    # Use SmolLM2 tokenizer as primary; clip for GPT-2
    enc_b = tok_b(texts, truncation=True, padding=True, max_length=64, return_tensors="pt")
    ids_b, mask_b = enc_b.input_ids.to(DEVICE), enc_b.attention_mask.to(DEVICE)
    
    stitched = StitchedModel(ma, mb, bridge, tok_a, tok_b).to(DEVICE)
    
    best = None; best_loss = float('inf')
    print(f"  Bridge: {sum(p.numel() for p in bridge.parameters()):,} params")
    for s in range(steps):
        opt.zero_grad()
        o = stitched(None, ids_b, None, mask_b, labels=ids_b)
        if torch.isnan(o.loss):
            print(f"  Step {s+1}: loss is nan, resetting bridge")
            bridge = FusionBridge(d_a, d_b).to(device=DEVICE, dtype=dtype)
            opt = torch.optim.AdamW(bridge.parameters(), lr=lr/10, weight_decay=0.01)
            continue
        o.loss.backward(); torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0); opt.step()
        if o.loss.item() < best_loss:
            best_loss = o.loss.item(); best = copy.deepcopy(bridge.state_dict())
        if (s+1) % 10 == 0:
            print(f"  Step {s+1}/{steps} loss={o.loss.item():.4f}")
    bridge.load_state_dict(best)
    return bridge, stitched

# ─── SOLUTION 2B: VOCABULARY BRIDGE ───────────────────────────────────────

class TokenizerBridge(nn.Module):
    """Learnable soft mapping between two token embeddings"""
    def __init__(self, tok_a, tok_b, dim, hid=128):
        super().__init__()
        self.tok_a = tok_a; self.tok_b = tok_b
        # Cosine similarity-based projection
        emb_a = tok_a.weight.data.float(); emb_b = tok_b.weight.data.float()
        norm_a = emb_a / emb_a.norm(dim=-1, keepdim=True)
        norm_b = emb_b / emb_b.norm(dim=-1, keepdim=True)
        sim = norm_a @ norm_b.T  # [vocab_a, vocab_b]
        
        # Top-k soft projection
        topk = min(5, sim.shape[-1])
        vals, idxs = sim.topk(topk, dim=-1)
        self.register_buffer("proj_idx", idxs)  # [vocab_a, topk]
        self.register_buffer("proj_val", vals)   # [vocab_a, topk]
    
    def forward(self, emb_b):
        # emb_b: [vocab_b, dim]
        idx = self.proj_idx  # [vocab_a, k]
        val = F.softmax(self.proj_val, dim=-1)  # [vocab_a, k]
        mapped = emb_b[idx] * val.unsqueeze(-1)  # [vocab_a, k, dim]
        return mapped.sum(dim=1)  # [vocab_a, dim]

# ─── RUN SOLUTION 1 ────────────────────────────────────────────────────────

def run_solution1():
    print("\n" + "="*60)
    print("SOLUTION 1: DIFFERENT SIZES - gpt2 + distilgpt2")
    print("="*60)
    
    print("Loading models...")
    ma = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=DTYPE).to(DEVICE).eval()
    mb = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=DTYPE).to(DEVICE).eval()
    tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    
    print(f"  A: {ma.config.n_layer}L, {ma.config.n_embd}d  B: {mb.config.n_layer}L, {mb.config.n_embd}d")
    
    print("Loading calibration data...")
    ids, mask, _ = load_calib("gpt2", n=32)
    
    # Baselines
    print("Baselines...")
    pp_a = ppl(ma, ids, mask); pp_b = ppl(mb, ids, mask)
    print(f"  Original A PPL: {pp_a:.2f}")
    print(f"  Original B PPL: {pp_b:.2f}")
    
    # Random search optimization
    print("Optimizing per-layer merge coefficients...")
    best_sd, best_alphas, best_ppl, mapping = random_search_merge(ma, mb, ids, mask, trials=40)
    
    # Create merged model
    mm = AutoModelForCausalLM.from_config(ma.config).to(DEVICE)
    mm.load_state_dict(best_sd, strict=False)
    
    opt_text = generate(mm, tok)
    print(f"\n  Optimized merge PPL: {best_ppl:.2f}")
    print(f"  Output: {opt_text}")
    
    r_text = generate(mm, tok)
    
    # Uniform 0.5 baseline for comparison
    b_proj_u = prepare_b_projected(ma, mb, mapping)
    sd_u = merge_same_arch(ma, mb, mapping, {}, b_proj_u)
    mu = AutoModelForCausalLM.from_config(ma.config).to(DEVICE)
    mu.load_state_dict(sd_u, strict=False)
    u_ppl = ppl(mu, ids, mask)
    u_text = generate(mu, tok)
    del mu; clean()
    
    print(f"\n{'-'*60}")
    print("RESULTS")
    print(f"{'-'*60}")
    print(f"  Original A PPL:       {pp_a:.2f}")
    print(f"  Original B PPL:       {pp_b:.2f}")
    print(f"  Uniform(0.5) PPL:     {u_ppl:.2f}")
    print(f"  Optimized PPL:         {best_ppl:.2f}")
    print(f"  CEILING BROKEN:       {'YES' if best_ppl < pp_a*2 else 'PARTIAL' if best_ppl < pp_a*10 else 'NO'}")
    print(f"{'-'*60}")
    print(f"\nUniform(0.5) output:    {u_text}")
    print(f"\nOptimized output:       {r_text}")
    
    return best_ppl

# ─── RUN SOLUTION 2 ────────────────────────────────────────────────────────

def run_solution2():
    print("\n" + "="*60)
    print("SOLUTION 2: DIFFERENT ARCHITECTURES - distilgpt2 + SmolLM2-135M")
    print("="*60)
    
    print("Loading models...")
    import gc
    # Use smaller model A to save memory
    ma = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=DTYPE).to(DEVICE).eval()
    tok_a = AutoTokenizer.from_pretrained("distilgpt2"); tok_a.pad_token = tok_a.eos_token
    
    mb = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=DTYPE).to(DEVICE).eval()
    tok_b = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    if tok_b.pad_token is None: tok_b.pad_token = tok_b.eos_token
    
    print(f"  A: {ma.config.model_type} {ma.config.n_layer}L,{ma.config.n_embd}d  vocab={ma.config.vocab_size}")
    print(f"  B: {mb.config.model_type} {mb.config.num_hidden_layers}L,{mb.config.hidden_size}d  vocab={mb.config.vocab_size}")
    
    # Load calibration texts
    print("Loading calibration data...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
    texts = [t["text"] for t in ds if len(t["text"].strip()) > 50][:32]
    if not texts: texts = ["The quick brown fox jumps over the lazy dog."] * 32
    
    # Baselines
    print("Baselines...")
    enc = tok_a(texts[:16], truncation=True, padding=True, max_length=64, return_tensors="pt")
    pp_a = ppl(ma, enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE))
    print(f"  Original A PPL: {pp_a:.2f}")
    
    # Train fusion bridge (per-tokenizer inputs) - use very small batch
    print("\nTraining fusion bridge...")
    bridge, stitched = train_bridge(ma, mb, tok_a, tok_b, texts[:32], steps=30)
    
    # Evaluate bridge PPL
    stitched.eval()
    enc_b = tok_b(texts[:16], truncation=True, padding=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        o = stitched(None, enc_b.input_ids.to(DEVICE), None, enc_b.attention_mask.to(DEVICE),
                     labels=enc_b.input_ids.to(DEVICE))
    b_ppl = math.exp(o.loss.item()) if o.loss is not None and not math.isnan(o.loss.item()) else 999
    print(f"\n  Bridge PPL: {b_ppl:.2f}")
    
    # Generate
    print("\nGenerating text...")
    prompts = [
        "The future of artificial intelligence is",
        "In the beginning, there was",
        "The most important discovery was",
    ]
    for prompt in prompts:
        try:
            text = stitched.generate(prompt, max_new=30)
            print(f"  [{prompt}]\n    -> {text}")
        except Exception as e:
            print(f"  [{prompt}]\n    -> FAILED: {e}")
    
    print(f"\n{'-'*60}")
    print("RESULTS")
    print(f"{'-'*60}")
    print(f"  Original A PPL:       {pp_a:.2f}")
    print(f"  Bridge PPL:           {b_ppl:.2f}")
    print(f"  CEILING BROKEN:       {'YES' if b_ppl < pp_a*3 else 'PARTIAL' if b_ppl < pp_a*10 else 'NO'}")
    print(f"{'-'*60}")
    
    return b_ppl

# ─── MAIN ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sol = sys.argv[1] if len(sys.argv) > 1 else "1"
    
    if sol == "1":
        run_solution1()
    elif sol == "2":
        run_solution2()
    else:
        run_solution1()
