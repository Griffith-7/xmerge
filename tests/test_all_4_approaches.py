"""
Test all 4 xmerge approaches with GPT-2 Large (812M) + Medium (355M).
Both already cached, same architecture, different sizes -- perfect for all 4 approaches.
"""
import sys, os, time, json, math, gc, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from xmerge import merge_prod, merge_stream

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = merge_prod.SAVE_DIR
os.makedirs(SAVE_DIR, exist_ok=True)

CALIB_TEXTS = [
    "General relativity describes gravity as the curvature of spacetime.",
    "Photosynthesis is the process by which plants convert sunlight into energy.",
    "Evolution explains how species adapt over generations through natural selection.",
    "Quantum mechanics reveals particles exist in multiple states until observed.",
    "Machine learning algorithms improve by learning patterns from training data.",
    "The human brain contains billions of neurons connected by trillions of synapses.",
    "Climate change refers to long-term shifts in temperature patterns on Earth.",
    "The Fibonacci sequence appears throughout nature in leaf and flower arrangements.",
    "Blockchain enables decentralized trust through cryptographic proof.",
    "Neural networks are computing systems inspired by biological brains.",
    "DNA replication ensures genetic information is accurately copied during cell division.",
    "The Industrial Revolution marked a major turning point in human technology.",
    "Entropy is a measure of disorder in a thermodynamic system.",
    "Plate tectonics explains the movement of Earth's lithosphere.",
    "The electromagnetic spectrum includes radio waves, visible light, and more.",
    "Game theory analyzes strategic decision-making with multiple agents.",
]

def load_model(model_name):
    """Load model directly (these are small enough for our RAM/VRAM)."""
    print(f"  Loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(DEVICE).eval()
    merge_stream.clean()
    return model, tok

def load_model_cpu(model_name):
    """Load model on CPU for streaming approach."""
    print(f"  Loading {model_name} on CPU...")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).eval()
    for p in model.parameters():
        p.data = p.data.cpu()
    merge_stream.clean()
    return model, tok


# ===========================================================================
# APPROACH 1 -- Weight Blending
# ===========================================================================

def test_approach_1(ma, mb, tok):
    print("\n" + "="*70)
    print("  APPROACH 1: Weight Blending (CKA + alpha blend)")
    print("="*70)

    t0 = time.time()
    merged, _, ppl_val, alphas = merge_stream.streamed_merge_same_arch(
        ma, mb, calib_texts=CALIB_TEXTS[:4],
        save_name="approach1_weight_blend", device=DEVICE,
    )
    elapsed = time.time() - t0
    print(f"\n  OK PPL: {ppl_val:.1f} | Time: {elapsed:.0f}s | alpha range: [{min(alphas.values()):.2f}, {max(alphas.values()):.2f}]")
    return {"approach": "1_weight_blend", "ppl": round(ppl_val, 1), "time_s": round(elapsed, 1)}


# ===========================================================================
# APPROACH 2 -- Bridge v2 (non-cached, streamed forward each step)
# ===========================================================================

def test_approach_2(ma, mb, tok):
    print("\n" + "="*70)
    print("  APPROACH 2: Bridge v2 (streamed forward each step)")
    print("="*70)

    t0 = time.time()
    bridge, ppl_val = merge_stream.streamed_train_bridge_v2(
        ma, mb, tok, CALIB_TEXTS[:8], device=DEVICE,
        steps=10, max_len=64, batch_size=2,
    )
    elapsed = time.time() - t0
    print(f"\n  OK PPL: {ppl_val:.1f} | Time: {elapsed:.0f}s")
    return {"approach": "2_bridge_v2", "ppl": round(ppl_val, 1), "time_s": round(elapsed, 1)}


# ===========================================================================
# APPROACH 3 -- Bridge Cached (cache once, train on GPU)
# ===========================================================================

def test_approach_3(ma, mb, tok):
    print("\n" + "="*70)
    print("  APPROACH 3: Bridge Cached (cache once, GPU train)")
    print("="*70)

    t0 = time.time()
    bridge, ppl_val = merge_stream.streamed_train_bridge_cached(
        ma, mb, tok, CALIB_TEXTS[:8], device=DEVICE,
        steps=20, max_len=64, batch_size=2,
    )
    elapsed = time.time() - t0
    print(f"\n  OK PPL: {ppl_val:.1f} | Time: {elapsed:.0f}s")
    return {"approach": "3_bridge_cached", "ppl": round(ppl_val, 1), "time_s": round(elapsed, 1)}


# ===========================================================================
# APPROACH 4 -- Full Pipeline (cache + train + save + eval + generate)
# ===========================================================================

def test_approach_4(ma, mb, tok):
    print("\n" + "="*70)
    print("  APPROACH 4: Full Pipeline (cache + train + save + generate)")
    print("="*70)

    t0 = time.time()
    bridge, ppl_val = merge_stream.streamed_merge_diff_arch(
        ma, mb, calib_texts=CALIB_TEXTS[:8],
        save_name="approach4_full_pipeline", device=DEVICE,
        tok=tok, steps=20, max_len=64,
    )
    elapsed = time.time() - t0

    # Generate sample text
    gen = merge_stream.StreamedGenerator(ma, mb, bridge, tok, device=DEVICE)
    sample = gen.generate("The future of AI is", max_new=20, method="bridge")

    # Also try stitch generate
    sample2 = gen.generate("The meaning of life is", max_new=20, method="stitch")

    print(f"\n  OK PPL: {ppl_val:.1f} | Time: {elapsed:.0f}s")
    print(f"  Gen (bridge): \"{sample}\"")
    print(f"  Gen (stitch): \"{sample2}\"")
    return {
        "approach": "4_full_pipeline",
        "ppl": round(ppl_val, 1),
        "time_s": round(elapsed, 1),
        "generation_bridge": sample,
        "generation_stitch": sample2,
    }


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("="*70)
    print("  XMERGE -- ALL 4 APPROACHES TEST")
    print(f"  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB)")
    print(f"  Models: GPT-2 Medium (355M) + DistilGPT-2 (82M)")
    print("="*70)

    # Load both models (small enough to fit in 4GB VRAM simultaneously)
    print("\n-- Loading models --")
    ma, tok = load_model("gpt2-medium")
    mb, _ = load_model("distilgpt2")

    results = []

    tests = [
        ("1_weight_blend", test_approach_1),
        ("2_bridge_v2", test_approach_2),
        ("3_bridge_cached", test_approach_3),
        ("4_full_pipeline", test_approach_4),
    ]

    for name, fn in tests:
        try:
            r = fn(ma, mb, tok)
            results.append(r)
        except Exception as e:
            print(f"\n  ! {name} FAILED: {e}")
            import traceback; traceback.print_exc()
            results.append({"approach": name, "error": str(e)})
        merge_stream.clean()

    # Print summary table
    print("\n" + "="*70)
    print("  FINAL RESULTS -- ALL 4 APPROACHES")
    print("="*70)
    print(f"  {'Approach':<25} {'PPL':<10} {'Time':<10} {'Notes'}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*30}")

    for r in results:
        name = r.get("approach", "?")
        if "error" in r:
            print(f"  {name:<25} {'FAILED':<10} {'':<10} {r['error'][:50]}")
        else:
            ppl = r.get("ppl", "?")
            ts = f"{r.get('time_s', 0):.0f}s"
            gen = r.get("generation_bridge", "")
            notes = f"gen: \"{gen[:40]}...\"" if gen else ""
            print(f"  {name:<25} {ppl:<10} {ts:<10} {notes}")

    with open(os.path.join(SAVE_DIR, "results_all_4.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {SAVE_DIR}/results_all_4.json")
    print("="*70)


if __name__ == "__main__":
    main()
