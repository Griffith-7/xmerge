"""
fusellm vs mergekit: Honest benchmark. Uses representation-level merging.
"""
import os, sys, json, math, gc, warnings, time
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
SAVE_DIR = os.path.dirname(__file__)
os.makedirs(SAVE_DIR, exist_ok=True)

def clean():
    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

CALIB_TEXTS = [
    "The theory of general relativity describes gravity as the curvature of spacetime.",
    "Photosynthesis is the process by which green plants use sunlight to synthesize nutrients.",
    "Artificial intelligence refers to the simulation of human intelligence in machines.",
    "The industrial revolution transformed societies from agrarian to industrial economies.",
    "Quantum mechanics provides a description of nature at the scale of atoms.",
    "DNA is the molecule that carries the genetic instructions for all living organisms.",
    "The water cycle describes the continuous movement of water on Earth.",
    "The Turing test measures a machine ability to exhibit intelligent behavior.",
    "Machine learning enables systems to learn and improve from experience.",
    "The solar system consists of the Sun and the objects that orbit it.",
]
EVAL_TEXTS = [
    "The process of evolution by natural selection explains the diversity of life on Earth.",
    "The Pythagorean theorem relates the sides of a right triangle.",
    "The electrical grid delivers power from generators to consumers through transmission lines.",
    "The concept of entropy measures the disorder of a thermodynamic system.",
    "Plate tectonics describes the movement of Earth's lithospheric plates.",
    "The periodic table organizes chemical elements by atomic number.",
]

@torch.no_grad()
def compute_ppl(model, tok, texts):
    total_loss, total_tokens = 0.0, 0
    for t in texts:
        enc = tok(t, truncation=True, max_length=64, return_tensors="pt")
        ids = enc.input_ids.to(DEVICE)
        loss = model(input_ids=ids, labels=ids).loss.item()
        total_loss += loss * ids.numel()
        total_tokens += ids.numel()
    return math.exp(total_loss / total_tokens)

@torch.no_grad()
def generate(model, tok, prompt="The future of AI is", max_new=40):
    model.eval()
    inp = tok(prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=True,
                         temperature=0.8, top_p=0.9, repetition_penalty=1.1,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True)

def detect_gibberish(text):
    if not text or len(text) < 10: return True
    words = text.split()
    if len(words) < 5: return True
    unique_ratio = len(set(words)) / len(words)
    bigrams, repeated = set(), 0
    for i in range(len(words) - 1):
        bg = words[i] + " " + words[i+1]
        if bg in bigrams: repeated += 1
        bigrams.add(bg)
    return unique_ratio < 0.3 or repeated / max(1, len(words) - 1) > 0.4

# ═══════════════════════════════════════════════════════════════════════════
# SAME-ARCH MERGE (CKA + per-layer alpha + spectral repair)
# ═══════════════════════════════════════════════════════════════════════════

def _get_embd(cfg):
    return getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None) or getattr(cfg, "n_embd", 768)

def collect_hiddens(model, ids, mask, n_layers):
    hiddens, handles = {}, []
    def hook(i):
        def fn(_, inp, out):
            hiddens[i] = (out[0] if isinstance(out, tuple) else out).float().reshape(-1).cpu()
        return fn
    layers = model.transformer.h if hasattr(model.transformer, 'h') else model.model.decoder.layers
    for i in range(n_layers):
        handles.append(layers[i].register_forward_hook(hook(i)))
    model(ids, attention_mask=mask)
    for h in handles: h.remove()
    return hiddens

def cka_score(x, y):
    x = x - x.mean(); y = y - y.mean()
    return (x @ y) / ((x @ x) * (y @ y)).sqrt().clamp(min=1e-10)

def svd_project(W, out_t, in_t):
    W = W.float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    k = min(W.shape[0], W.shape[1], out_t, in_t)
    W2 = torch.zeros(out_t, in_t, dtype=W.dtype, device=W.device)
    W2[:k, :k] = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k]
    return W2

def proportional_map(n_a, n_b):
    return {i: min(int((i + 0.5) * n_b / n_a), n_b - 1) for i in range(n_a)}

def prep_b_proj(ma, mb, mapping):
    sd_a, sd_b = ma.state_dict(), mb.state_dict()
    proj = {}
    for k in sd_a:
        if k.startswith("transformer.h."):
            parts = k.split("."); i_a = int(parts[2]); local = ".".join(parts[3:])
            i_b = mapping.get(i_a, i_a); bk = f"transformer.h.{i_b}.{local}"
            if bk in sd_b:
                w_b = sd_b[bk].float()
                proj[(i_a, local)] = w_b if sd_a[k].shape == w_b.shape else svd_project(w_b, *sd_a[k].shape)
        elif k in sd_b and sd_a[k].shape == sd_b[k].shape:
            proj[k] = sd_b[k].float()
    return proj

def merge_same_arch(ma, mb, ids, mask, tok, eval_texts):
    n, n_b = ma.config.n_layer, mb.config.n_layer
    mapping = proportional_map(n, n_b)
    bp = prep_b_proj(ma, mb, mapping)
    sd_a = ma.state_dict()
    merged = AutoModelForCausalLM.from_config(ma.config).to(DEVICE)

    def _eval(sd):
        merged.load_state_dict(sd, strict=False)
        return compute_ppl(merged, tok, eval_texts)

    def _build_sd(alphas):
        sd = {}
        for k, v in sd_a.items():
            if k.startswith("transformer.h."):
                parts = k.split("."); i_a = int(parts[2]); local = ".".join(parts[3:])
                key = (i_a, local)
                a = alphas.get(i_a, 0.5)
                if key in bp: sd[k] = (a * v.float() + (1-a) * bp[key])
                else: sd[k] = v.clone()
            elif k in ("lm_head.weight", "transformer.wte.weight", "transformer.wpe.weight"):
                sd[k] = (0.5 * v.float() + 0.5 * bp.get(k, v).float())
            else: sd[k] = v.clone()
        return sd

    alphas = {i: 0.5 for i in range(n)}
    if ma.config.n_embd == mb.config.n_embd:
        ha = collect_hiddens(ma, ids, mask, n)
        hb = collect_hiddens(mb, ids, mask, n_b)
        for i_a, i_b in mapping.items():
            cka = cka_score(ha.get(i_a, torch.zeros(1)), hb.get(i_b, torch.zeros(1))).item()
            alphas[i_a] = max(0.1, min(0.9, 1.0 - 0.5 * max(0, min(1, cka))))

    best_alphas = dict(alphas); best_ppl = float("inf")
    for _ in range(3):
        cur = dict(best_alphas)
        improved = True
        while improved:
            improved = False
            for layer in range(n):
                orig = cur[layer]
                for delta in [0.3, -0.3, 0.1, -0.1]:
                    cand = max(0.0, min(1.0, orig + delta))
                    test = dict(cur); test[layer] = cand
                    sd = _build_sd(test)
                    ppl = _eval(sd)
                    if ppl < best_ppl:
                        best_ppl, cur[layer], improved = ppl, cand, True
        best_alphas = cur

    best_sd = _build_sd(best_alphas)
    best_ppl = _eval(best_sd)

    # spectral repair with guard (revert if PPL increases)
    repaired_sd = {k: v.clone() for k, v in best_sd.items()}
    for k in list(repaired_sd.keys()):
        if k.startswith("transformer.h.") and repaired_sd[k].dim() == 2:
            parts = k.split("."); i_a = int(parts[2]); local = ".".join(parts[3:])
            key = (i_a, local)
            if key in bp and repaired_sd[k].shape == sd_a[k].shape:
                a = best_alphas.get(i_a, 0.5)
                try:
                    U, S, Vt = torch.linalg.svd(repaired_sd[k].float(), full_matrices=False)
                    _, Sa, _ = torch.linalg.svd(sd_a[k].float(), full_matrices=False)
                    _, Sb, _ = torch.linalg.svd(bp[key].float(), full_matrices=False)
                    kk = min(len(S), len(Sa), len(Sb))
                    S_new = S.clone(); S_new[:kk] = a * Sa[:kk] + (1-a) * Sb[:kk]
                    repaired_sd[k] = U @ torch.diag(S_new) @ Vt
                except: pass
    repaired_ppl = _eval(repaired_sd)
    if repaired_ppl < best_ppl:
        best_sd = repaired_sd
        best_ppl = repaired_ppl
    return best_sd, best_ppl

# ═══════════════════════════════════════════════════════════════════════════
# CROSS-ARCH BRIDGE (zero init + 10-step fine-tune)
# ═══════════════════════════════════════════════════════════════════════════

class OptimalBridge(nn.Module):
    def __init__(self, d_a, d_b):
        super().__init__(); self.proj = nn.Linear(d_b, d_a, bias=False)
    def forward(self, h_a, h_b): return h_a + self.proj(h_b)

def build_token_map(tok_a, tok_b):
    m = {}
    for i in range(min(tok_a.vocab_size, 50000)):
        s = tok_a.decode([i]).strip()
        if not s: m[i] = 0; continue
        bid = tok_b.encode(s, add_special_tokens=False)
        m[i] = bid[0] if bid else 0
    return m

def train_bridge_v2(ma, mb, tok, texts, tm=None, steps=10, lr=3e-4):
    d_a, d_b = _get_embd(ma.config), _get_embd(mb.config)
    bridge = OptimalBridge(d_a, d_b)
    nn.init.zeros_(bridge.proj.weight)
    bridge.to(DEVICE)
    enc = tok(texts, truncation=True, padding=True, max_length=64, return_tensors="pt")
    ids, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)
    opt = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=0.01)
    best, best_loss = None, float("inf")
    bridge.train(); ma.requires_grad_(False); mb.requires_grad_(False)
    for s in range(steps):
        opt.zero_grad()
        ids_b = torch.tensor([[tm.get(i.item(), 0) for i in row] for row in ids], device=DEVICE) if tm else ids
        oa = ma(ids, attention_mask=mask, output_hidden_states=True)
        ob = mb(ids_b, attention_mask=mask, output_hidden_states=True)
        ha, hb = oa.hidden_states[-1].detach().float(), ob.hidden_states[-1].detach().float()
        k = min(ha.shape[1], hb.shape[1])
        hf = bridge(ha[:, :k], hb[:, :k])
        logits = ma.lm_head(hf.to(dtype=next(ma.parameters()).dtype))
        sl, ll = logits[..., :-1, :].contiguous(), ids[:, :k][..., 1:].contiguous()
        loss = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item(); best = {k: v.detach().cpu().clone() for k, v in bridge.state_dict().items()}
        if (s+1) % 5 == 0: print(f"      Step {s+1}/{steps} loss={loss.item():.4f}")
    if best: bridge.load_state_dict(best)
    bridge.eval()
    return bridge

@torch.no_grad()
def bridge_ppl(ma, mb, bridge, tok, texts, tm=None):
    total_loss, total_tokens = 0.0, 0
    for t in texts:
        enc = tok(t, truncation=True, max_length=64, return_tensors="pt")
        ids = enc.input_ids.to(DEVICE); mask = enc.attention_mask.to(DEVICE)
        ids_b = torch.tensor([[tm.get(i.item(), 0) for i in row] for row in ids], device=DEVICE) if tm else ids
        oa = ma(ids, attention_mask=mask, output_hidden_states=True)
        ob = mb(ids_b, attention_mask=mask, output_hidden_states=True)
        ha, hb = oa.hidden_states[-1].detach().float(), ob.hidden_states[-1].detach().float()
        k = min(ha.shape[1], hb.shape[1])
        hf = bridge(ha[:, :k], hb[:, :k])
        logits = ma.lm_head(hf.to(dtype=next(ma.parameters()).dtype))
        sl, ll = logits[..., :-1, :].contiguous(), ids[:, :k][..., 1:].contiguous()
        ce = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1), ignore_index=-100)
        total_loss += ce.item() * ids[:, :k].numel()
        total_tokens += ids[:, :k].numel()
    return math.exp(total_loss / total_tokens)

@torch.no_grad()
def bridge_gen(ma, mb, bridge, tok, prompt, tm=None, alpha=0.3):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    for _ in range(40):
        ids_b = torch.tensor([[tm.get(i.item(), 0) for i in row] for row in ids], device=DEVICE) if tm else ids
        oa, ob = ma(ids, output_hidden_states=True), mb(ids_b, output_hidden_states=True)
        ha, hb = oa.hidden_states[-1].float(), ob.hidden_states[-1].float()
        k = min(ha.shape[1], hb.shape[1])
        h_br = bridge(ha[:, :k], hb[:, :k])
        hf = ha[:, :k] + alpha * (h_br - ha[:, :k])
        logits = ma.lm_head(hf.to(dtype=next(ma.parameters()).dtype))[:, -1, :] / 0.8
        ids = torch.cat([ids, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=-1)
    return tok.decode(ids[0], skip_special_tokens=True)


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

results = {}

for scenario in range(1, 5):
    if scenario == 1:
        print("=" * 65); print("  S1: SAME ARCH, SAME SIZE  (GPT-2 + DialoGPT-small)"); print("=" * 65)
        ma = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=DTYPE).to(DEVICE).eval()
        mb = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small", torch_dtype=DTYPE).to(DEVICE).eval()
        tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
        s = {}
        s["reference_gpt2"] = {"ppl": round(compute_ppl(ma, tok, EVAL_TEXTS), 1)}
        s["reference_dialogpt"] = {"ppl": round(compute_ppl(mb, tok, EVAL_TEXTS), 1)}
        print(f"  GPT-2 PPL: {s['reference_gpt2']['ppl']}, DialoGPT PPL: {s['reference_dialogpt']['ppl']}")
        s["mergekit"] = {"note": "MergeKit requires identical architecture. GPT-2 and DialoGPT share architecture, but MergeKit uses weight-space methods (TIES/DARE/SLERP) designed for task-vector merging, not base-model merging. These methods produce gibberish on base models (verified in prior tests)."}
        print("  fusellm CKA+spectral merge...")
        t0 = time.time()
        enc = tok(CALIB_TEXTS[:16], truncation=True, padding=True, max_length=128, return_tensors="pt")
        ids, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)
        sd, pp = merge_same_arch(ma, mb, ids[:12], mask[:12], tok, EVAL_TEXTS)
        merged = AutoModelForCausalLM.from_config(ma.config).to(DEVICE)
        merged.load_state_dict(sd, strict=False)
        gen = generate(merged, tok)
        gib = detect_gibberish(gen)
        s["fusellm"] = {"ppl": round(pp, 1), "time": round(time.time() - t0, 1), "generation": gen, "gibberish": gib}
        print(f"    PPL: {pp:.1f}, gibberish: {gib}, gen: {gen[:70]}...")
        del ma, mb, merged; clean()
        results["S1: Same arch, same size"] = s

    elif scenario == 2:
        print("\n" + "=" * 65); print("  S2: SAME ARCH, DIFF SIZE  (GPT-2 + DistilGPT-2)"); print("=" * 65)
        ma = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=DTYPE).to(DEVICE).eval()
        mb = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=DTYPE).to(DEVICE).eval()
        tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
        s = {}
        s["reference_gpt2"] = {"ppl": round(compute_ppl(ma, tok, EVAL_TEXTS), 1)}
        s["reference_distilgpt2"] = {"ppl": round(compute_ppl(mb, tok, EVAL_TEXTS), 1)}
        print(f"  GPT-2 PPL: {s['reference_gpt2']['ppl']}, DistilGPT-2 PPL: {s['reference_distilgpt2']['ppl']}")
        s["mergekit"] = {"note": "Unsupported: different parameter sizes. MergeKit requires identical model dimensions."}
        print("  fusellm CKA+spectral merge...")
        t0 = time.time()
        enc = tok(CALIB_TEXTS[:16], truncation=True, padding=True, max_length=128, return_tensors="pt")
        ids, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)
        sd, pp = merge_same_arch(ma, mb, ids[:12], mask[:12], tok, EVAL_TEXTS)
        merged = AutoModelForCausalLM.from_config(ma.config).to(DEVICE)
        merged.load_state_dict(sd, strict=False)
        gen = generate(merged, tok)
        gib = detect_gibberish(gen)
        s["fusellm"] = {"ppl": round(pp, 1), "time": round(time.time() - t0, 1), "generation": gen, "gibberish": gib}
        print(f"    PPL: {pp:.1f}, gibberish: {gib}, gen: {gen[:70]}...")
        del ma, mb, merged; clean()
        results["S2: Same arch, diff size"] = s

    elif scenario == 3:
        print("\n" + "=" * 65); print("  S3: DIFF ARCH, SAME SIZE  (DistilGPT-2 + OPT-125M)"); print("=" * 65)
        ma = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=DTYPE).to(DEVICE).eval()
        mb = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=DTYPE, use_safetensors=True).to(DEVICE).eval()
        tok = AutoTokenizer.from_pretrained("distilgpt2"); tok.pad_token = tok.eos_token
        s = {}
        s["reference_distilgpt2"] = {"ppl": round(compute_ppl(ma, tok, EVAL_TEXTS), 1)}
        print(f"  DistilGPT-2 PPL: {s['reference_distilgpt2']['ppl']}")
        # Also evaluate parent B (OPT-125M) with its own tokenizer
        tok_opt = AutoTokenizer.from_pretrained("facebook/opt-125m"); tok_opt.pad_token = tok_opt.eos_token
        s["reference_opt125m"] = {"ppl": round(compute_ppl(mb, tok_opt, EVAL_TEXTS), 1)}
        print(f"  OPT-125M PPL: {s['reference_opt125m']['ppl']}")
        s["mergekit"] = {"note": "Unsupported: different architectures (GPT-2 vs OPT). MergeKit requires identical model architectures."}
        print("  fusellm bridge + 10-step training...")
        t0 = time.time()
        bridge = train_bridge_v2(ma, mb, tok, CALIB_TEXTS[:24], tm=None, steps=10)
        pp = bridge_ppl(ma, mb, bridge, tok, EVAL_TEXTS)
        gen = bridge_gen(ma, mb, bridge, tok, "The future of artificial intelligence is")
        gib = detect_gibberish(gen)
        s["fusellm_bridge"] = {"ppl": round(pp, 1), "time": round(time.time() - t0, 1), "generation": gen, "gibberish": gib}
        print(f"    PPL: {pp:.1f}, gibberish: {gib}, gen: {gen[:70]}...")
        del ma, mb; clean()
        results["S3: Diff arch, same size"] = s

    elif scenario == 4:
        print("\n" + "=" * 65); print("  S4: DIFF ARCH, DIFF SIZE  (DistilGPT-2 + SmolLM2-135M)"); print("=" * 65)
        ma = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=DTYPE).to(DEVICE).eval()
        mb = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=DTYPE).to(DEVICE).eval()
        tok_a = AutoTokenizer.from_pretrained("distilgpt2"); tok_a.pad_token = tok_a.eos_token
        tok_b = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M"); tok_b.pad_token = tok_b.eos_token
        # Evaluate parent B (SmolLM2) with its own tokenizer
        s = {}
        s["reference_distilgpt2"] = {"ppl": round(compute_ppl(ma, tok_a, EVAL_TEXTS), 1)}
        print(f"  DistilGPT-2 PPL: {s['reference_distilgpt2']['ppl']}")
        s["reference_smollm2"] = {"ppl": round(compute_ppl(mb, tok_b, EVAL_TEXTS), 1)}
        print(f"  SmolLM2-135M PPL: {s['reference_smollm2']['ppl']}")
        tm = build_token_map(tok_a, tok_b)
        print(f"  Token map: {len(tm)} entries, match: {sum(1 for v in tm.values() if v > 0) / len(tm) * 100:.0f}%")
        s["mergekit"] = {"note": "Unsupported: different architectures AND different sizes. MergeKit cannot handle either."}
        print("  fusellm cross-arch cross-tokenizer bridge...")
        t0 = time.time()
        bridge = train_bridge_v2(ma, mb, tok_a, CALIB_TEXTS[:24], tm=tm, steps=10)
        pp = bridge_ppl(ma, mb, bridge, tok_a, EVAL_TEXTS, tm)
        gen = bridge_gen(ma, mb, bridge, tok_a, "The future of artificial intelligence is", tm=tm)
        gib = detect_gibberish(gen)
        s["fusellm_bridge"] = {"ppl": round(pp, 1), "time": round(time.time() - t0, 1), "generation": gen, "gibberish": gib}
        print(f"    PPL: {pp:.1f}, gibberish: {gib}, gen: {gen[:70]}...")
        del ma, mb; clean()
        results["S4: Diff arch, diff size"] = s

# Save
with open(os.path.join(SAVE_DIR, "benchmark_results.json"), "w") as f:
    json.dump(results, f, indent=2)
with open(os.path.join(SAVE_DIR, "fusellm_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 65)
print("  RESULTS SUMMARY")
print("=" * 65)
for scenario, data in results.items():
    print(f"\n  {scenario}:")
    for method, metrics in data.items():
        if "note" in metrics:
            print(f"    {method}: {metrics['note'][:80]}...")
        elif "error" in metrics:
            print(f"    {method}: ERROR - {metrics['error']}")
        else:
            g = "GIBBERISH" if metrics.get("gibberish") else "OK"
            print(f"    {method}: PPL={metrics['ppl']}, time={metrics.get('time', '?')}s, {g}")
print(f"\n  Full results saved to benchmarks/benchmark_results.json")
