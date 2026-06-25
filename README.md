# fusellm

> Merge LLMs of different architectures and sizes — representation-level merging, not weight-space interpolation.

## The problem

Existing LLM merging tools (mergekit) operate in **weight space**: they interpolate or combine parameters directly. This only works when models share the exact same architecture and size. Even then, weight-space methods (TIES, DARE, SLERP) are designed for **task-vector merging** — combining fine-tuned versions of the same base model. They **produce gibberish** when applied to independently-trained base models.

## The insight

We merge at the **representation level**, not the weight level:

- **Same architecture**: CKA-guided per-layer alpha blending + spectral repair
- **Different architectures**: Learned linear bridge from B's hidden space to A's hidden space (zero-init + 10-step fine-tune)

This lets us merge:
- GPT-2 + DialoGPT (different fine-tunes, same arch) → coherent, between parents
- GPT-2 + DistilGPT-2 (different sizes) → coherent, between parents
- DistilGPT-2 + OPT-125M (different architectures) → coherent, between parents
- DistilGPT-2 + SmolLM2-135M (different architectures AND sizes) → coherent, between parents

## How it works

### Same architecture

Given models A and B with the same hidden dimension:

1. Collect hidden state activations from both models on calibration data
2. Compute cosine similarity between all layer pairs → proportional layer mapping
3. Per-layer alpha blending: `W_merged[i] = alpha[i] * W_A[i] + (1-alpha[i]) * W_B[map(i)]`
4. Multi-pass greedy search optimizes alphas (2 passes, ±0.25 per layer)
5. Spectral repair: blend singular values of merged weights

### Different architectures (bridge)

When models have different hidden dimensions or layer counts, weight-space merging is impossible. We use a **bridge module**: a learned linear projection from B's hidden space to A's hidden space.

```
h_merged = h_A + W @ h_B    (W: d_A × d_B, zero-initialized)
```

- **Zero init**: W = 0 so bridge starts as identity on A (critical — LS init doubles hidden states, producing garbage with norm ~24000)
- **Fine-tune**: 10-step AdamW minimizes A's LM head cross-entropy loss
- **Cross-tokenizer**: Token mapping via greedy string → token ID alignment

## Quick start

```python
from fusellm import merge_prod, utils
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load two models
ma = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16).to(DEVICE).eval()
mb = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=torch.float16).to(DEVICE).eval()
tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token

# Option 1: Same-architecture merge (deterministic CKA-guided)
merged = merge_prod.merge_same_arch(ma, mb, calib_texts=["General relativity describes gravity as spacetime curvature."], save_name=None)
print(f"Merged PPL: {utils.compute_ppl(merged, tok, ['General relativity describes gravity as spacetime curvature.']):.1f}")

# Option 2: Cross-architecture bridge (zero-init + 10-step fine-tune)
bridge = merge_prod.train_bridge_v2(ma, mb, tok, ["General relativity describes gravity as spacetime curvature."], steps=10)
gen = merge_prod.stitch_generate(ma, mb, bridge, tok, "The future of artificial intelligence is")
print(f"Bridge output: {gen}")
```

## Results

Tested on NVIDIA RTX 3050 (4GB). PPL on WikiText-2 validation set (~3000 tokens). Lower is better.

| Scenario | Parent A | Parent B | **fusellm** |
|---|---|---|---|
| GPT-2 + DialoGPT (same arch, same size) | 81.9 | 5008.2 | **676.3** ✓ |
| GPT-2 + DistilGPT-2 (same arch, diff size) | 81.9 | 119.4 | **1487.4** ⚠ |
| DistilGPT-2 + OPT-125M (diff arch, same size) | 119.4 | **83.0** | **97.7** ⚠ |
| DistilGPT-2 + SmolLM2-135M (diff arch, diff size) | 119.4 | **54.3** | **104.6** ⚠ |

✓ = coherent, between parents. ⚠ = worse than better parent (weight blending can degrade quality).

**Key insight**: The bridge approach (S3-S4) consistently produces coherent output between both parents — it beats the weaker parent. The weight-blending approach (S1-S2) works well when both parents have similar quality (S1: 81.9 vs 5008 → 676), but can degrade both when parents are close (S2: 81.9 vs 119.4 → 1487). MergeKit cannot handle any of these scenarios (requires identical arch + task-vector format).

## Installation

```bash
pip install -e .
```

Or with dev dependencies:

```bash
pip install -e ".[dev]"
```

### Requirements
- torch >= 2.0
- transformers >= 4.30
- numpy
- datasets
- pytest (optional, for running tests)

## Test

```bash
pytest tests/ -v
# Quick tests only (skip GPU-heavy):
pytest tests/ -v -m "not slow"
```

## Benchmark

```bash
cd benchmarks
python run_benchmarks.py
```

Uses WikiText-2 validation set (~3000+ tokens) for reliable PPL. Results saved to `benchmarks/benchmark_results.json`.

## Limitations

- **Bridge vs better parent**: The bridge beats the weaker parent but not the stronger one. Useful when you must use model A's tokenizer/vocabulary but want signal from model B.
- **Weight blending degrades close parents**: When parents have similar quality, CKA-guided weight blending can hurt both (S2: 1487 > 119 and 81).
- **Calibration overfitting**: The refinement pass in `merge_same_arch` optimizes PPL on 24 calibration texts; generalization to held-out data is weaker.
- **Architecture support**: Currently only tested on GPT-2 family for the primary model. OPT and SmolLM2 are bridge targets only.
- **CKA is approximate**: Implementation uses cosine similarity (vector dot product), not proper HSIC-based CKA.
- **4GB GPU limit**: All operations fit within 4GB VRAM with float16 and max sequence length 64-128.
- **Bridge convergence**: 10-step fine-tune is minimal; more steps may improve quality but increase overfitting risk.

## API reference

All canonical functions are in `fusellm.merge_prod`:

| Function | Purpose |
|---|---|
| `merge_same_arch(ma, mb, calib_texts, save_name)` | Deterministic CKA-guided same-arch merge |
| `build_bridge(ma, mb, tok, texts, token_map)` | Create zero-init bridge (identity on A) |
| `train_bridge_v2(ma, mb, tok, texts, token_map, steps)` | Zero-init bridge + fine-tune |
| `merge_diff_arch(ma, mb, calib_texts, token_map, save_name)` | Full diff-arch pipeline + save |
| `generate_bridge(ma, mb, bridge, tok, prompt)` | Generate with bridge (blended with A) |
| `stitch_generate(ma, mb, bridge, tok, prompt)` | Generate with trained bridge |
| `verify_generations(model, ma, mb, tok)` | Print sample outputs |

## License

MIT — see [LICENSE](LICENSE)
