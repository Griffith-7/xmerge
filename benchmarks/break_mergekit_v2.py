"""MergeKit comparison v2 — same-model verification + GPT-2 + GPT-2 medium"""
import torch, math, gc, os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fusellm import merge_prod, utils as futils
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

def clean():
    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
TEXTS = [t["text"].strip() for t in ds if len(t["text"].strip()) > 10][:96]
CALIB, EVAL = TEXTS[:48], TEXTS[48:]
tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token

def ppl_model(model, texts=EVAL[:32]):
    tl, tt = 0.0, 0
    for t in texts:
        enc = tok(t, truncation=True, max_length=128, return_tensors="pt")
        ids = enc.input_ids.to(DEVICE); n = ids.numel()
        tl += model(input_ids=ids, labels=ids).loss.item() * n; tt += n
    return math.exp(tl / tt)

def slerp(t0, t1, alpha):
    t0_f, t1_f = t0.float().flatten(), t1.float().flatten()
    dot = (t0_f * t1_f).sum().clamp(-1, 1)
    omega = torch.acos(dot)
    if omega.abs() < 1e-6:
        return (1 - alpha) * t0 + alpha * t1
    return (torch.sin((1 - alpha) * omega) / torch.sin(omega)) * t0 + \
           (torch.sin(alpha * omega) / torch.sin(omega)) * t1

# Test 1: Two copies of same GPT-2 (should produce identical PPL for linear/SLERP)
print("="*60)
print("  TEST 1: Two copies of GPT-2 (same weights)")
print("="*60)
ma = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=DTYPE).to(DEVICE).eval()
mb = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=DTYPE).to(DEVICE).eval()

sd_a, sd_b = ma.state_dict(), mb.state_dict()

# Linear merge (alpha=0.5) — same weights → same result
linear_sd = {k: (0.5 * sd_a[k].float() + 0.5 * sd_b[k].float()).to(DTYPE) for k in sd_a}
m_lin = AutoModelForCausalLM.from_config(ma.config).to(DEVICE).eval()
m_lin.load_state_dict(linear_sd, strict=False)

# SLERP merge — same vectors → should be identical
slerp_sd = {}
for k in sd_a:
    if k in sd_b and sd_a[k].shape == sd_b[k].shape:
        slerp_sd[k] = slerp(sd_a[k], sd_b[k], 0.5).to(DTYPE)
    else:
        slerp_sd[k] = sd_a[k].clone()
m_sl = AutoModelForCausalLM.from_config(ma.config).to(DEVICE).eval()
m_sl.load_state_dict(slerp_sd, strict=False)

pp_a = ppl_model(ma)
m_lin.eval(); m_sl.eval()
pp_lin = ppl_model(m_lin)
pp_sl = ppl_model(m_sl)
print(f"  Original:        PPL={pp_a:.1f}")
print(f"  Linear (0.5+0.5):PPL={pp_lin:.1f} {'[IDENTICAL]' if abs(pp_lin-pp_a) < 0.5 else '[DIFFERENT]'}")
print(f"  SLERP (0.5+0.5): PPL={pp_sl:.1f} {'[IDENTICAL]' if abs(pp_sl-pp_a) < 0.5 else '[DIFFERENT]'}")

del ma, mb, m_lin, m_sl; clean()

# Test 2: GPT-2 vs GPT-2 with noise injection on B
print("\n" + "="*60)
print("  TEST 2: GPT-2 vs GPT-2 + 5% noise (simulate 'different knowledge')")
print("="*60)
torch.manual_seed(42)
ma = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=DTYPE).to(DEVICE).eval()
mb = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=DTYPE).to(DEVICE).eval()
sd_a, sd_b = ma.state_dict(), mb.state_dict()

# Add small Gaussian noise to B
noise_std = 0.05
sd_b_noisy = {}
for k, v in sd_b.items():
    sd_b_noisy[k] = v + torch.randn_like(v) * v.std().item() * noise_std

# Load noisy B
mb_noisy = AutoModelForCausalLM.from_config(ma.config).to(DEVICE).eval()
mb_noisy.load_state_dict(sd_b_noisy, strict=False)
pp_b_noisy = ppl_model(mb_noisy)
pp_a = ppl_model(ma)
print(f"  A (GPT-2):        PPL={pp_a:.1f}")
print(f"  B (GPT-2 + noise):PPL={pp_b_noisy:.1f}")

# Linear merge
linear_sd = {}
for k in sd_a:
    if k in sd_b_noisy and sd_a[k].shape == sd_b_noisy[k].shape:
        linear_sd[k] = (0.5 * sd_a[k].float() + 0.5 * sd_b_noisy[k].float()).to(DTYPE)
    else:
        linear_sd[k] = sd_a[k].clone()
m_lin = AutoModelForCausalLM.from_config(ma.config).to(DEVICE).eval()
m_lin.load_state_dict(linear_sd, strict=False)

# SLERP
slerp_sd = {}
for k in sd_a:
    if k in sd_b_noisy and sd_a[k].shape == sd_b_noisy[k].shape:
        slerp_sd[k] = slerp(sd_a[k], sd_b_noisy[k], 0.5).to(DTYPE)
    else:
        slerp_sd[k] = sd_a[k].clone()
m_sl = AutoModelForCausalLM.from_config(ma.config).to(DEVICE).eval()
m_sl.load_state_dict(slerp_sd, strict=False)

m_lin.eval(); m_sl.eval()
pp_lin = ppl_model(m_lin)
pp_sl = ppl_model(m_sl)

# Bridge
bridge, _ = merge_prod.merge_same_arch_bridge(ma, mb_noisy, tok, CALIB[:32], steps=10, save_name=None)
dtype = next(ma.parameters()).dtype
tl, tt = 0.0, 0
for t in EVAL[:32]:
    enc = tok(t, truncation=True, max_length=128, return_tensors="pt")
    ids = enc.input_ids.to(DEVICE); mask = enc.attention_mask.to(DEVICE)
    loss, _ = merge_prod._stitch_forward(ma, mb_noisy, bridge, ids, mask, ids, dtype)
    tl += loss.item() * ids.numel(); tt += ids.numel()
pp_bridge = math.exp(tl / tt)

print(f"\n  RESULTS:")
print(f"  Linear (A+B_noisy): PPL={pp_lin:.1f}  (delta from A: {pp_lin-pp_a:+.1f})")
print(f"  SLERP (A+B_noisy):  PPL={pp_sl:.1f}   (delta from A: {pp_sl-pp_a:+.1f})")
print(f"  fusellm bridge:    PPL={pp_bridge:.1f} (delta from A: {pp_bridge-pp_a:+.1f})")

# Generation comparison
print(f"\n  Generations 'The future of AI is':")
inp = tok("The future of AI is", return_tensors="pt").to(DEVICE)
out_a = ma.generate(**inp, max_new_tokens=15, do_sample=False, pad_token_id=tok.eos_token_id)
out_lin = m_lin.generate(**inp, max_new_tokens=15, do_sample=False, pad_token_id=tok.eos_token_id)
out_sl = m_sl.generate(**inp, max_new_tokens=15, do_sample=False, pad_token_id=tok.eos_token_id)
gen_bridge = merge_prod.stitch_generate(ma, mb_noisy, bridge, tok, "The future of AI is", max_new=15)
print(f"  A:      {tok.decode(out_a[0], skip_special_tokens=True)[:40]}")
print(f"  Linear: {tok.decode(out_lin[0], skip_special_tokens=True)[:40]}")
print(f"  SLERP:  {tok.decode(out_sl[0], skip_special_tokens=True)[:40]}")
print(f"  Bridge: {gen_bridge[:40]}")

del ma, mb, mb_noisy, m_lin, m_sl, bridge; clean()

# Summary
print(f"\n{'='*60}")
print("  TAKEAWAY")
print(f"{'='*60}")
print("  - Linear/SLERP: zero-shot, work on same-arch models, good when models are similar.")
print("  - Bridge: needs calibration training (10-20 steps), works on DIFFERENT architectures.")
print("  - On same-arch noisy models, linear/SLERP preserve original PPL better than bridge")
print("    because they don't need training and don't overfit.")
print("  - Bridge's advantage is CROSS-ARCHITECTURE merging which SLERP/LINEAR cannot do.")
