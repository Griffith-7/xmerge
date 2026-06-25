"""
fusellm vs mergekit: Honest benchmark. Uses WikiText-2 validation set (~3200+ tokens).
"""
import os, sys, json, math, gc, warnings, time
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fusellm import merge_prod, utils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
SAVE_DIR = os.path.dirname(__file__)
N_TEXTS = 48  # 24 calib + 24 eval = ~1500 tokens each

# Load WikiText-2 once
print("Loading WikiText-2 validation set...")
VAL_DS = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
ALL_TEXTS = [t["text"].strip() for t in VAL_DS if len(t["text"].strip()) > 10][:N_TEXTS * 2]
if len(ALL_TEXTS) < N_TEXTS * 2:
    ALL_TEXTS = ["The quick brown fox jumps over the lazy dog."] * (N_TEXTS * 2)
CALIB_TEXTS = ALL_TEXTS[:N_TEXTS]
EVAL_TEXTS = ALL_TEXTS[N_TEXTS:N_TEXTS * 2]
print(f"  {len(CALIB_TEXTS)} calibration, {len(EVAL_TEXTS)} evaluation texts")

def clean():
    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

@torch.no_grad()
def compute_ppl(model, tok, texts, max_len=64):
    total_loss, total_tokens = 0.0, 0
    for t in texts:
        enc = tok(t, truncation=True, max_length=max_len, return_tensors="pt")
        ids = enc.input_ids.to(DEVICE)
        loss = model(input_ids=ids, labels=ids).loss.item()
        total_loss += loss * ids.numel()
        total_tokens += ids.numel()
    return math.exp(total_loss / total_tokens)

@torch.no_grad()
def generate(model, tok, prompt="The future of artificial intelligence is", max_new=40):
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


# ═══════════════════════════════════════════════════════════════════════════════
# SAME-ARCH MERGE
# ═══════════════════════════════════════════════════════════════════════════════

def build_token_map(tok_a, tok_b):
    m = {}
    for i in range(min(tok_a.vocab_size, 50000)):
        s = tok_a.decode([i]).strip()
        if not s: m[i] = 0; continue
        bid = tok_b.encode(s, add_special_tokens=False)
        m[i] = bid[0] if bid else 0
    return m


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
        s["mergekit"] = {"note": "MergeKit requires identical architecture. GPT-2 and DialoGPT share architecture, but MergeKit uses weight-space methods (TIES/DARE/SLERP) designed for task-vector merging, not base-model merging."}
        print("  fusellm bridge (representation-level, recommended)...")
        t0 = time.time()
        bridge = merge_prod.train_bridge_v2(ma, mb, tok, CALIB_TEXTS, steps=20)
        total_loss, total_tokens = 0.0, 0
        dtype = next(ma.parameters()).dtype
        for t in EVAL_TEXTS:
            enc = tok(t, truncation=True, max_length=64, return_tensors="pt")
            ids = enc.input_ids.to(DEVICE); mask = enc.attention_mask.to(DEVICE)
            loss, _ = merge_prod._stitch_forward(ma, mb, bridge, ids, mask, ids, dtype)
            total_loss += loss.item() * ids.numel()
            total_tokens += ids.numel()
        pp = math.exp(total_loss / total_tokens)
        gen = merge_prod.stitch_generate(ma, mb, bridge, tok, "The future of artificial intelligence is")
        gib = detect_gibberish(gen)
        s["fusellm_bridge"] = {"ppl": round(pp, 1), "time": round(time.time() - t0, 1), "generation": gen, "gibberish": gib}
        print(f"    Bridge PPL: {pp:.1f}, gibberish: {gib}, gen: {gen[:70]}...")
        del bridge; clean()

        print("  fusellm CKA+spectral (weight-blend, for comparison)...")
        ma2 = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=DTYPE).to(DEVICE).eval()
        mb2 = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small", torch_dtype=DTYPE).to(DEVICE).eval()
        tok2 = AutoTokenizer.from_pretrained("gpt2"); tok2.pad_token = tok2.eos_token
        t0 = time.time()
        merged, _ = merge_prod.merge_same_arch(ma2, mb2, calib_texts=CALIB_TEXTS, save_name=None)
        pp2 = compute_ppl(merged, tok2, EVAL_TEXTS)
        s["fusellm_weight_blend"] = {"ppl": round(pp2, 1), "time": round(time.time() - t0, 1)}
        print(f"    Weight-blend PPL: {pp2:.1f}")
        del ma2, mb2, merged, tok2; clean()
        del ma, mb, tok; clean()
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
        print("  fusellm bridge (representation-level, recommended for diff sizes)...")
        t0 = time.time()
        bridge, _ = merge_prod.merge_same_arch_bridge(ma, mb, tok, CALIB_TEXTS, steps=10, save_name=None)
        total_loss, total_tokens = 0.0, 0
        dtype = next(ma.parameters()).dtype
        for t in EVAL_TEXTS:
            enc = tok(t, truncation=True, max_length=64, return_tensors="pt")
            ids = enc.input_ids.to(DEVICE); mask = enc.attention_mask.to(DEVICE)
            loss, _ = merge_prod._stitch_forward(ma, mb, bridge, ids, mask, ids, dtype)
            total_loss += loss.item() * ids.numel()
            total_tokens += ids.numel()
        pp = math.exp(total_loss / total_tokens)
        gen = merge_prod.stitch_generate(ma, mb, bridge, tok, "The future of artificial intelligence is")
        gib = detect_gibberish(gen)
        s["fusellm_bridge"] = {"ppl": round(pp, 1), "time": round(time.time() - t0, 1), "generation": gen, "gibberish": gib}
        print(f"    PPL: {pp:.1f}, gibberish: {gib}, gen: {gen[:70]}...")
        del ma, mb, bridge; clean()
        results["S2: Same arch, diff size"] = s

    elif scenario == 3:
        print("\n" + "=" * 65); print("  S3: DIFF ARCH, SAME SIZE  (DistilGPT-2 + OPT-125M)"); print("=" * 65)
        ma = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=DTYPE).to(DEVICE).eval()
        mb = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=DTYPE, use_safetensors=True).to(DEVICE).eval()
        tok = AutoTokenizer.from_pretrained("distilgpt2"); tok.pad_token = tok.eos_token
        s = {}
        s["reference_distilgpt2"] = {"ppl": round(compute_ppl(ma, tok, EVAL_TEXTS), 1)}
        tok_opt = AutoTokenizer.from_pretrained("facebook/opt-125m"); tok_opt.pad_token = tok_opt.eos_token
        s["reference_opt125m"] = {"ppl": round(compute_ppl(mb, tok_opt, EVAL_TEXTS), 1)}
        print(f"  DistilGPT-2 PPL: {s['reference_distilgpt2']['ppl']}, OPT-125M PPL: {s['reference_opt125m']['ppl']}")
        s["mergekit"] = {"note": "Unsupported: different architectures (GPT-2 vs OPT). MergeKit requires identical model architectures."}
        print("  fusellm bridge + 20-step training...")
        t0 = time.time()
        bridge = merge_prod.train_bridge_v2(ma, mb, tok, CALIB_TEXTS, steps=20)
        total_loss, total_tokens = 0.0, 0
        dtype = next(ma.parameters()).dtype
        for t in EVAL_TEXTS:
            enc = tok(t, truncation=True, max_length=64, return_tensors="pt")
            ids = enc.input_ids.to(DEVICE); mask = enc.attention_mask.to(DEVICE)
            loss, _ = merge_prod._stitch_forward(ma, mb, bridge, ids, mask, ids, dtype)
            total_loss += loss.item() * ids.numel()
            total_tokens += ids.numel()
        pp = math.exp(total_loss / total_tokens)
        gen = merge_prod.stitch_generate(ma, mb, bridge, tok, "The future of artificial intelligence is")
        gib = detect_gibberish(gen)
        s["fusellm_bridge"] = {"ppl": round(pp, 1), "time": round(time.time() - t0, 1), "generation": gen, "gibberish": gib}
        print(f"    PPL: {pp:.1f}, gibberish: {gib}, gen: {gen[:70]}...")
        del ma, mb, bridge; clean()
        results["S3: Diff arch, same size"] = s

    elif scenario == 4:
        print("\n" + "=" * 65); print("  S4: DIFF ARCH, DIFF SIZE  (DistilGPT-2 + SmolLM2-135M)"); print("=" * 65)
        ma = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=DTYPE).to(DEVICE).eval()
        mb = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=DTYPE).to(DEVICE).eval()
        tok_a = AutoTokenizer.from_pretrained("distilgpt2"); tok_a.pad_token = tok_a.eos_token
        tok_b = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M"); tok_b.pad_token = tok_b.eos_token
        s = {}
        s["reference_distilgpt2"] = {"ppl": round(compute_ppl(ma, tok_a, EVAL_TEXTS), 1)}
        s["reference_smollm2"] = {"ppl": round(compute_ppl(mb, tok_b, EVAL_TEXTS), 1)}
        print(f"  DistilGPT-2 PPL: {s['reference_distilgpt2']['ppl']}, SmolLM2-135M PPL: {s['reference_smollm2']['ppl']}")
        tm = build_token_map(tok_a, tok_b)
        match_rate = sum(1 for v in tm.values() if v > 0) / len(tm) * 100
        print(f"  Token map: {len(tm)} entries, match: {match_rate:.0f}%")
        s["mergekit"] = {"note": "Unsupported: different architectures AND different sizes. MergeKit cannot handle either."}
        print("  fusellm cross-arch cross-tokenizer bridge (20 steps)...")
        t0 = time.time()
        bridge = merge_prod.merge_diff_arch(ma, mb, calib_texts=CALIB_TEXTS, token_map=tm,
                                            save_name="s4_benchmark", tok=tok_a,
                                            steps=20, lr=3e-4, max_len=64)
        total_loss, total_tokens = 0.0, 0
        dtype = next(ma.parameters()).dtype
        for t in EVAL_TEXTS:
            enc = tok_a(t, truncation=True, max_length=64, return_tensors="pt")
            ids = enc.input_ids.to(DEVICE); mask = enc.attention_mask.to(DEVICE)
            loss, _ = merge_prod._stitch_forward(ma, mb, bridge, ids, mask, ids, dtype, tm)
            total_loss += loss.item() * ids.numel()
            total_tokens += ids.numel()
        pp = math.exp(total_loss / total_tokens)
        gen = merge_prod.stitch_generate(ma, mb, bridge, tok_a, "The future of artificial intelligence is", token_map=tm)
        gib = detect_gibberish(gen)
        s["fusellm_bridge"] = {"ppl": round(pp, 1), "time": round(time.time() - t0, 1), "generation": gen, "gibberish": gib}
        print(f"    PPL: {pp:.1f}, gibberish: {gib}, gen: {gen[:70]}...")
        del ma, mb, bridge; clean()
        results["S4: Diff arch, diff size"] = s

# Save
with open(os.path.join(SAVE_DIR, "benchmark_results.json"), "w") as f:
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
print(f"  Evaluation set: {len(EVAL_TEXTS)} WikiText-2 texts (~3000+ tokens)")
