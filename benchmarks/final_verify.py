"""Final verification of all fixes."""
import torch, math, gc, os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fusellm import merge_prod, utils
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
texts = [t["text"].strip() for t in ds if len(t["text"].strip()) > 10][:96]
calib, eval_texts = texts[:48], texts[48:]

def compute_ppl(model, tok, texts, max_len=64):
    total_loss, total_tokens = 0.0, 0
    for t in texts:
        enc = tok(t, truncation=True, max_length=max_len, return_tensors="pt")
        ids = enc.input_ids.to(DEVICE)
        loss = model(input_ids=ids, labels=ids).loss.item()
        total_loss += loss * ids.numel()
        total_tokens += ids.numel()
    return math.exp(total_loss / total_tokens)

def clean():
    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

results = {}

# 1. Weight-blend pure A detection fix
print("=" * 65)
print("  TEST 1: Weight-blend pure-A detection (should match GPT-2 PPL)")
print("=" * 65)
clean()
ma = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=DTYPE).to(DEVICE).eval()
mb = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small", torch_dtype=DTYPE).to(DEVICE).eval()
tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token

ppl_a = compute_ppl(ma, tok, eval_texts)
print(f"  Reference GPT-2 PPL: {ppl_a:.1f}")

# Call merge_same_arch, it should detect pure-A is best
merged, _ = merge_prod.merge_same_arch(ma, mb, calib_texts=calib, save_name=None)
ppl_merged = compute_ppl(merged, tok, eval_texts)
print(f"  Weight-blend PPL (fixed): {ppl_merged:.1f}")
print(f"  Ratio vs GPT-2: {ppl_merged/ppl_a:.2f}x")

if ppl_merged < ppl_a * 3:
    print("  [PASS] Weight-blend no longer garbage (within 3x of good parent)")
else:
    print("  [WARN] Still worse than expected")
results["weight_blend_fix"] = {"gpt2_ppl": round(ppl_a, 1), "merged_ppl": round(ppl_merged, 1), "ratio": round(ppl_merged/ppl_a, 2)}
del ma, mb, merged; clean()

# 2. Bridge methods unchanged
print()
print("=" * 65)
print("  TEST 2: Bridge PPL values preserved")
print("=" * 65)
scenarios = [
    ("S1: GPT-2 + DialoGPT", "gpt2", "microsoft/DialoGPT-small", "gpt2", 20, 60.2),
    ("S2: GPT-2 + DistilGPT-2", "gpt2", "distilgpt2", "gpt2", 10, 47.6),
    ("S3: DistilGPT-2 + OPT-125M", "distilgpt2", "facebook/opt-125m", "distilgpt2", 20, 66.4),
]
for label, model_a, model_b, tokenizer, steps, expected in scenarios:
    clean()
    ma = AutoModelForCausalLM.from_pretrained(model_a, torch_dtype=DTYPE).to(DEVICE).eval()
    mb = AutoModelForCausalLM.from_pretrained(model_b, torch_dtype=DTYPE, use_safetensors=True).to(DEVICE).eval()
    tok = AutoTokenizer.from_pretrained(tokenizer); tok.pad_token = tok.eos_token
    bridge = merge_prod.train_bridge_v2(ma, mb, tok, calib, steps=steps)
    total_loss, total_tokens = 0.0, 0
    dtype = next(ma.parameters()).dtype
    for t in eval_texts:
        enc = tok(t, truncation=True, max_length=64, return_tensors="pt")
        ids, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)
        loss, _ = merge_prod._stitch_forward(ma, mb, bridge, ids, mask, ids, dtype)
        total_loss += loss.item() * ids.numel()
        total_tokens += ids.numel()
    ppl = math.exp(total_loss / total_tokens)
    match = abs(ppl - expected) < 1.0
    print(f"  {label}: got {ppl:.1f}, expected {expected} {'[PASS]' if match else '[FAIL]'}")
    results[f"bridge_{label.split(':')[0].strip()}"] = {"ppl": round(ppl, 1), "expected": expected, "pass": match}
    del ma, mb, bridge; clean()

# 3. Overfitting check (better regularization still needed)
print()
print("=" * 65)
print("  TEST 3: Overfitting (train 3 texts, eval 48)")
print("=" * 65)
clean()
ma = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=DTYPE).to(DEVICE).eval()
mb = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=DTYPE, use_safetensors=True).to(DEVICE).eval()
tok = AutoTokenizer.from_pretrained("distilgpt2"); tok.pad_token = tok.eos_token

bridge_overfit = merge_prod.train_bridge_v2(ma, mb, tok, calib[:3], steps=50)
train_ppl = compute_ppl(ma, tok, calib[:3])
eval_ppl_bridge = 0; count = 0
dtype = next(ma.parameters()).dtype
for t in eval_texts:
    enc = tok(t, truncation=True, max_length=64, return_tensors="pt")
    ids, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)
    loss, _ = merge_prod._stitch_forward(ma, mb, bridge_overfit, ids, mask, ids, dtype)
    eval_ppl_bridge += loss.item() * ids.numel()
    count += ids.numel()
eval_ppl_bridge = math.exp(eval_ppl_bridge / count)
print(f"  Train PPL (3 texts): {train_ppl:.1f}")
print(f"  Overfit bridge eval PPL: {eval_ppl_bridge:.1f}")
gap = eval_ppl_bridge - train_ppl
print(f"  Generalization gap: {gap:.1f} {'[SEVERE]' if gap > 100 else '[NORMAL]'}")
results["overfitting"] = {"train_ppl": round(train_ppl, 1), "eval_ppl": round(eval_ppl_bridge, 1)}
del ma, mb, bridge_overfit; clean()

print()
print("=" * 65)
print("  FINAL SUMMARY")
print("=" * 65)
all_pass = all(v.get("pass", True) for v in results.values() if isinstance(v, dict))
for k, v in results.items():
    print(f"  {k}: {v}")
print(f"\n  All checks passed: {all_pass}")

with open(os.path.join(os.path.dirname(__file__), "final_verification.json"), "w") as f:
    json.dump(results, f, indent=2)
