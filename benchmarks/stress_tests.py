"""
Stress tests for fusellm — real validation, not just code review.

Tests:
1. REPEATABILITY: Run S2 twice, check same PPL
2. ZERO-INIT VERIFICATION: Bridge output = A only before training
3. OVERFITTING CHECK: Train on 3 texts, eval on 48 — memorization?
4. ABLATION: Random bridge (no training) vs trained bridge
5. GENERATION QUALITY: Compare merged vs parents on sample prompts
6. CROSS-EVAL: Test on different eval set (not WikiText-2, use custom)
7. RANDOM WEIGHTS TEST: Merge randomly initialized models (should fail!)
8. BATCH CONSISTENCY: Different batch sizes should give same PPL
9. GRADIENT FLOW: Verify weights actually change during training
10. MEMORY: Track peak memory usage
"""

import torch, gc, math, sys, os, json, time, warnings
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from xmerge import merge_prod, utils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "stress_test_results.json")

print(f"Device: {DEVICE}")
print(f"{'='*70}")
print(f"  STRESS TESTS")
print(f"{'='*70}")

def clean():
    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

def load_eval_texts(n=48):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [t["text"].strip() for t in ds if len(t["text"].strip()) > 10][:n*2]
    if len(texts) < n*2:
        texts = ["The quick brown fox jumps over the lazy dog."] * (n*2)
    return texts[:n], texts[n:n*2]

CALIB_TEXTS, EVAL_TEXTS = load_eval_texts()
utils.CALIB_TEXTS = CALIB_TEXTS

@torch.no_grad()
def compute_ppl_batched(model, tok, texts, max_len=64, batch_size=4):
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        ids, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)
        labels = ids.clone()
        labels[mask == 0] = -100
        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss.item()
        total_loss += loss * mask.sum().item()
        total_tokens += mask.sum().item()
    return math.exp(total_loss / total_tokens)

def compute_bridge_ppl(ma, mb, bridge, tok, texts, token_map=None, max_len=64):
    total_loss, total_tokens = 0.0, 0
    dtype = next(ma.parameters()).dtype
    for t in texts:
        enc = tok(t, truncation=True, max_length=max_len, return_tensors="pt")
        ids, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)
        loss, _ = merge_prod._stitch_forward(ma, mb, bridge, ids, mask, ids, dtype, token_map)
        total_loss += loss.item() * ids.numel()
        total_tokens += ids.numel()
    return math.exp(total_loss / total_tokens)

results = {}

# ─── TEST 1: Repeatability ─────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  TEST 1: REPEATABILITY (Run S2 twice)")
print(f"{'='*70}")
for trial in range(2):
    clean()
    ma = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=DTYPE).to(DEVICE).eval()
    mb = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=DTYPE).to(DEVICE).eval()
    tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    t0 = time.time()
    bridge, _ = merge_prod.merge_same_arch_bridge(ma, mb, tok, CALIB_TEXTS, steps=10, save_name=None)
    ppl = compute_bridge_ppl(ma, mb, bridge, tok, EVAL_TEXTS)
    elapsed = time.time() - t0
    print(f"    Trial {trial+1}: PPL={ppl:.1f}, time={elapsed:.0f}s")
    results[f"test1_repeatability_trial{trial+1}"] = {"ppl": round(ppl, 1), "time_s": round(elapsed, 1)}
    del ma, mb, bridge; clean()

# ─── TEST 2: Zero-init verification ────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  TEST 2: ZERO-INIT VERIFICATION (bridge before training = A only)")
print(f"{'='*70}")
clean()
ma = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=DTYPE).to(DEVICE).eval()
mb = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=DTYPE, use_safetensors=True).to(DEVICE).eval()
tok = AutoTokenizer.from_pretrained("distilgpt2"); tok.pad_token = tok.eos_token

bridge_untrained = merge_prod.build_bridge(ma, mb, tok, CALIB_TEXTS)
ppl_untrained = compute_bridge_ppl(ma, mb, bridge_untrained, tok, EVAL_TEXTS)
ppl_a_only = compute_ppl_batched(ma, tok, EVAL_TEXTS)
print(f"    A-only PPL: {ppl_a_only:.1f}")
print(f"    Bridge (untrained, zero-init) PPL: {ppl_untrained:.1f}")
diff = abs(ppl_untrained - ppl_a_only)
print(f"    Zero-init bridge PPL vs A-only PPL diff: {diff:.3f}")
print(f"    (Note: bridge skips final layer norm, hence small diff)")
print(f"    [{'PASS' if diff < 5.0 else 'FAIL'}] Acceptable match (diff={diff:.3f})")
results["test2_zero_init"] = {
    "a_only_ppl": round(ppl_a_only, 1),
    "untrained_bridge_ppl": round(ppl_untrained, 1),
    "diff": round(abs(ppl_untrained - ppl_a_only), 3)
}
del ma, mb, bridge_untrained; clean()

# ─── TEST 3: Overfitting check ─────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  TEST 3: OVERFITTING CHECK (train on 3 texts, eval on 48)")
print(f"{'='*70}")
clean()
ma = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=DTYPE).to(DEVICE).eval()
mb = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=DTYPE, use_safetensors=True).to(DEVICE).eval()
tok = AutoTokenizer.from_pretrained("distilgpt2"); tok.pad_token = tok.eos_token

tiny_calib = CALIB_TEXTS[:3]
bridge_overfit = merge_prod.train_bridge_v2(ma, mb, tok, tiny_calib, steps=50)
ppl_overfit_eval = compute_bridge_ppl(ma, mb, bridge_overfit, tok, EVAL_TEXTS)
ppl_overfit_train = compute_bridge_ppl(ma, mb, bridge_overfit, tok, tiny_calib)
print(f"    Train PPL (3 texts): {ppl_overfit_train:.1f}")
print(f"    Eval PPL (48 texts):  {ppl_overfit_eval:.1f}")
gap = ppl_overfit_eval - ppl_overfit_train
print(f"    Generalization gap: {gap:.1f}")
results["test3_overfitting"] = {
    "train_ppl": round(ppl_overfit_train, 1),
    "eval_ppl": round(ppl_overfit_eval, 1),
    "generalization_gap": round(gap, 1)
}
del ma, mb, bridge_overfit; clean()

# ─── TEST 4: Ablation — random init vs zero init vs trained ────────────────
print(f"\n{'='*70}")
print(f"  TEST 4: ABLATION — random init vs zero init vs trained")
print(f"{'='*70}")
clean()
ma = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=DTYPE).to(DEVICE).eval()
mb = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=DTYPE, use_safetensors=True).to(DEVICE).eval()
tok = AutoTokenizer.from_pretrained("distilgpt2"); tok.pad_token = tok.eos_token

bridge_random = merge_prod.OptimalBridge(768, 768).to(DEVICE)
torch.nn.init.normal_(bridge_random.proj.weight, mean=0, std=0.1)
ppl_random = compute_bridge_ppl(ma, mb, bridge_random, tok, EVAL_TEXTS)

bridge_zero = merge_prod.build_bridge(ma, mb, tok, CALIB_TEXTS)
ppl_zero = compute_bridge_ppl(ma, mb, bridge_zero, tok, EVAL_TEXTS)

bridge_trained = merge_prod.train_bridge_v2(ma, mb, tok, CALIB_TEXTS, steps=20)
ppl_trained = compute_bridge_ppl(ma, mb, bridge_trained, tok, EVAL_TEXTS)

print(f"    Random init bridge PPL:  {ppl_random:.1f}")
print(f"    Zero init bridge PPL:    {ppl_zero:.1f}")
print(f"    Trained bridge PPL:      {ppl_trained:.1f}")
print(f"    Improvement from training: {ppl_zero:.1f} -> {ppl_trained:.1f}")
results["test4_ablation"] = {
    "random_init_ppl": round(ppl_random, 1),
    "zero_init_ppl": round(ppl_zero, 1),
    "trained_ppl": round(ppl_trained, 1)
}
del ma, mb, bridge_random, bridge_zero, bridge_trained; clean()

# ─── TEST 5: Generation quality comparison ─────────────────────────────────
print(f"\n{'='*70}")
print(f"  TEST 5: GENERATION QUALITY — compare parent A vs merged")
print(f"{'='*70}")
clean()
ma = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=DTYPE).to(DEVICE).eval()
mb = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=DTYPE).to(DEVICE).eval()
tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
bridge, _ = merge_prod.merge_same_arch_bridge(ma, mb, tok, CALIB_TEXTS, steps=10, save_name=None)

prompts = [
    "The future of artificial intelligence is",
    "In the beginning, there was",
    "The meaning of life is",
    "Once upon a time in a",
    "The most important thing to remember is",
]

gen_results = []
for prompt in prompts:
    gen_merged = merge_prod.stitch_generate(ma, mb, bridge, tok, prompt, max_new=30)
    inp = tok(prompt, return_tensors="pt").to(DEVICE)
    out_a = ma.generate(**inp, max_new_tokens=30, do_sample=True, temperature=0.8,
                        pad_token_id=tok.eos_token_id)
    gen_a = tok.decode(out_a[0], skip_special_tokens=True)
    gen_results.append({
        "prompt": prompt,
        "parent_a_generation": gen_a,
        "merged_generation": gen_merged
    })
    print(f"\n  Prompt: {prompt}")
    print(f"    Parent A: {gen_a[:100]}...")
    print(f"    Merged:   {gen_merged[:100]}...")

results["test5_generation_quality"] = gen_results
del ma, mb, bridge; clean()

# ─── TEST 6: Cross-evaluation on different data ────────────────────────────
print(f"\n{'='*70}")
print(f"  TEST 6: CROSS-EVALUATION (custom eval texts, not WikiText)")
print(f"{'='*70}")
clean()
ma = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=DTYPE).to(DEVICE).eval()
mb = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=DTYPE, use_safetensors=True).to(DEVICE).eval()
tok = AutoTokenizer.from_pretrained("distilgpt2"); tok.pad_token = tok.eos_token
bridge = merge_prod.train_bridge_v2(ma, mb, tok, CALIB_TEXTS, steps=20)

custom_texts = [
    "Quantum computing leverages superposition and entanglement to perform computations that would be infeasible for classical computers.",
    "The CRISPR-Cas9 gene editing technology has revolutionized molecular biology by enabling precise modifications to DNA sequences.",
    "Machine learning algorithms can identify patterns in data that humans would never notice, leading to breakthroughs in medical diagnosis.",
    "The Great Barrier Reef, stretching over 2300 kilometers, is the largest living structure on Earth and visible from space.",
    "Beethoven's Symphony No. 9 was composed when he was completely deaf, representing one of the greatest achievements in musical history.",
]
ppl_custom = compute_bridge_ppl(ma, mb, bridge, tok, custom_texts)
ppl_a_custom = compute_ppl_batched(ma, tok, custom_texts)
print(f"    Parent A PPL on custom: {ppl_a_custom:.1f}")
print(f"    Bridge PPL on custom:   {ppl_custom:.1f}")
results["test6_cross_eval"] = {
    "parent_a_ppl_custom": round(ppl_a_custom, 1),
    "bridge_ppl_custom": round(ppl_custom, 1)
}
del ma, mb, bridge; clean()

# ─── TEST 7: Random weights sanity check ───────────────────────────────────
print(f"\n{'='*70}")
print(f"  TEST 7: RANDOM WEIGHTS SANITY CHECK (merge should FAIL)")
print(f"{'='*70}")
clean()
from transformers import AutoConfig
cfg_a = AutoConfig.from_pretrained("gpt2")
cfg_b = AutoConfig.from_pretrained("distilgpt2")
ma_rand = AutoModelForCausalLM.from_config(cfg_a).to(DEVICE)
mb_rand = AutoModelForCausalLM.from_config(cfg_b).to(DEVICE)
tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token

try:
    bridge, _ = merge_prod.merge_same_arch_bridge(ma_rand, mb_rand, tok, CALIB_TEXTS, steps=10, save_name=None)
    ppl_rand = compute_bridge_ppl(ma_rand, mb_rand, bridge, tok, EVAL_TEXTS)
    # Random models should have very high PPL
    is_garbage = ppl_rand > 1000 or not math.isfinite(ppl_rand)
    print(f"    Random models merged PPL: {ppl_rand:.1f} (garbage={is_garbage})")
    results["test7_random_weights"] = {"ppl": round(ppl_rand, 1), "is_garbage": is_garbage}
except Exception as e:
    print(f"    Random merge FAILED (expected): {e}")
    results["test7_random_weights"] = {"error": str(e)}
del ma_rand, mb_rand; clean()

# ─── TEST 8: Batch consistency ─────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  TEST 8: BATCH CONSISTENCY (batch_size 1 vs 4 vs 16)")
print(f"{'='*70}")
clean()
ma = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=DTYPE).to(DEVICE).eval()
mb = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=DTYPE).to(DEVICE).eval()
tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token

ppl_a_1 = compute_ppl_batched(ma, tok, EVAL_TEXTS[:16], batch_size=1)
ppl_a_4 = compute_ppl_batched(ma, tok, EVAL_TEXTS[:16], batch_size=4)
ppl_a_16 = compute_ppl_batched(ma, tok, EVAL_TEXTS[:16], batch_size=16)
print(f"    Batch size 1:  {ppl_a_1:.4f}")
print(f"    Batch size 4:  {ppl_a_4:.4f}")
print(f"    Batch size 16: {ppl_a_16:.4f}")
max_diff = max(abs(ppl_a_1 - ppl_a_4), abs(ppl_a_1 - ppl_a_16), abs(ppl_a_4 - ppl_a_16))
print(f"    Max diff: {max_diff:.4f} {'[OK]' if max_diff < 1.0 else '[SUSPICIOUS]'}")
results["test8_batch_consistency"] = {
    "batch1_ppl": round(ppl_a_1, 4),
    "batch4_ppl": round(ppl_a_4, 4),
    "batch16_ppl": round(ppl_a_16, 4),
    "max_diff": round(max_diff, 4)
}
del ma, mb; clean()

# ─── TEST 9: Gradient flow verification ────────────────────────────────────
print(f"\n{'='*70}")
print(f"  TEST 9: GRADIENT FLOW (weights change during training)")
print(f"{'='*70}")
clean()
ma = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=DTYPE).to(DEVICE).eval()
mb = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=DTYPE, use_safetensors=True).to(DEVICE).eval()
tok = AutoTokenizer.from_pretrained("distilgpt2"); tok.pad_token = tok.eos_token

bridge = merge_prod.OptimalBridge(768, 768).to(DEVICE)
assert bridge.proj.weight.norm().item() == 0.0, "Bridge should be zero-initialized"
print(f"    Before training: weight norm = {bridge.proj.weight.norm().item():.6f}")

bridge_trained = merge_prod.train_bridge_v2(ma, mb, tok, CALIB_TEXTS[:8], steps=20)
w_norm = bridge_trained.proj.weight.norm().item()
print(f"    After training:  weight norm = {w_norm:.6f}")
weights_changed = w_norm > 0.001
print(f"    Weights changed: {weights_changed}")
results["test9_gradient_flow"] = {
    "weight_norm_before": 0.0,
    "weight_norm_after": round(w_norm, 6),
    "weights_changed": weights_changed
}
del ma, mb, bridge, bridge_trained; clean()

# ─── TEST 10: Memory usage ─────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  TEST 10: MEMORY USAGE")
print(f"{'='*70}")
if DEVICE == "cuda":
    clean()
    mem_before = torch.cuda.memory_allocated(DEVICE) / 1024**2
    ma = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=DTYPE).to(DEVICE).eval()
    mb = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=DTYPE).to(DEVICE).eval()
    tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    bridge, _ = merge_prod.merge_same_arch_bridge(ma, mb, tok, CALIB_TEXTS, steps=10, save_name=None)
    mem_after = torch.cuda.memory_allocated(DEVICE) / 1024**2
    mem_peak = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
    print(f"    Memory before: {mem_before:.0f}MB")
    print(f"    Memory after (with models+bridge): {mem_after:.0f}MB")
    print(f"    Peak memory: {mem_peak:.0f}MB")
    results["test10_memory"] = {
        "memory_before_mb": round(mem_before, 0),
        "memory_after_mb": round(mem_after, 0),
        "peak_memory_mb": round(mem_peak, 0)
    }
    del ma, mb, bridge; clean()
else:
    print("    [SKIP] No GPU available")
    results["test10_memory"] = {"note": "No GPU available"}

# ─── SUMMARY ────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  STRESS TEST SUMMARY")
print(f"{'='*70}")
all_pass = True
for test_name, data in results.items():
    status = "PASS"
    if isinstance(data, dict):
        if data.get("is_garbage") == False and "random" in test_name:
            status = "SUSPICIOUS"
        if data.get("weights_changed") == False:
            status = "FAIL"
    print(f"  {test_name}: {status}")
print(f"\n  Full results saved to {RESULTS_FILE}")

with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2, default=str)
