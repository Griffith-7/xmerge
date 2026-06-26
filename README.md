# fusellm

> Merge LLMs of different architectures and sizes — representation-level merging, not weight-space interpolation.

## The problem

Existing LLM merging tools (mergekit) operate in **weight space**: they interpolate or combine parameters directly. This only works when models share the exact same architecture and size. Even then, weight-space methods (TIES, DARE, SLERP) are designed for **task-vector merging** — combining fine-tuned versions of the same base model. They **produce gibberish** when applied to independently-trained base models.

## The insight

We merge at the **representation level**, not the weight level:

- **All scenarios**: Learned linear bridge from B's hidden space to A's hidden space (zero-init + cosine-LR fine-tune)
- **Same architecture**: Also supports activation-similarity-guided weight blending as alternative

This lets us merge:
- GPT-2 + DialoGPT (different fine-tunes, same arch) → coherent, near GPT-2 quality
- GPT-2 + DistilGPT-2 (different sizes) → coherent, **beats both parents**
- DistilGPT-2 + OPT-125M (different architectures) → coherent, between parents
- DistilGPT-2 + SmolLM2-135M (different architectures AND sizes) → coherent, between parents

## How it works

### Bridge (recommended for all scenarios)

A **bridge module** is a learned linear projection from B's hidden space to A's hidden space:

```
h_merged = h_A + W @ h_B    (W: d_A × d_B, zero-initialized)
```

Applied at the final hidden layer before the LM head. The bridge starts as identity (A only) and learns to incorporate signal from B.

- **Zero init**: W = 0 so bridge starts as identity on A (critical — random init produces garbage)
- **Fine-tune**: 20-step AdamW with cosine LR schedule minimizes A's LM head cross-entropy loss on calibration data
- **Cross-tokenizer**: Token mapping via greedy string → token ID alignment (99.9% match rate for GPT-2 ↔ SmolLM2)

### Weight blending (alternative, same architecture only)

Given models A and B with the same hidden dimension:

1. Collect hidden state activations from both models on calibration data
2. Compute cosine similarity between all layer pairs → proportional layer mapping
3. Per-layer alpha blending: `W_merged[i] = alpha[i] * W_A[i] + (1-alpha[i]) * W_B[map(i)]`
4. Pure-A baseline check (if blending degrades PPL, fall back to pure A)
5. Multi-pass greedy search optimizes alphas (coarse ±0.25, fine ±0.1, ultra ±0.5/±0.05)
6. Spectral repair: blend singular values of merged weights

⚠ Weight blending degrades badly when one parent has poor LM quality. The bridge approach is recommended for all scenarios.

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

# Option 1 (recommended): Bridge (works for any architecture/size)
bridge = merge_prod.train_bridge_v2(ma, mb, tok, ["General relativity describes gravity as spacetime curvature."], steps=20)
gen = merge_prod.stitch_generate(ma, mb, bridge, tok, "The future of artificial intelligence is")
print(f"Bridge output: {gen}")

# Option 2: Same-architecture weight-blend (alternative)
merged, _ = merge_prod.merge_same_arch(ma, mb, calib_texts=["General relativity describes gravity as spacetime curvature."], save_name=None)
print(f"Merged PPL: {utils.compute_ppl(merged, tok, ['General relativity describes gravity as spacetime curvature.']):.1f}")
```

## Results

Tested on NVIDIA RTX 3050 (4GB). PPL on WikiText-2 validation set (~1500 tokens). Lower is better.

| Scenario | Parent A | Parent B | **fusellm** |
|---|---|---|---|
| GPT-2 + DialoGPT (same arch, same size) | 51.9 | 5721.4 | **60.2** ✓ |
| GPT-2 + DistilGPT-2 (same arch, diff size) | 51.9 | 80.5 | **47.6** ✓✓ |
| DistilGPT-2 + OPT-125M (diff arch, same size) | 80.5 | **59.5** | **66.4** ✓ |
| DistilGPT-2 + SmolLM2-135M (diff arch, diff size) | 80.5 | **34.1** | **70.9** ✓ |

✓ = coherent, close to better parent. ✓✓ = beats both parents.

**Key insight**: The **bridge approach** (representation-level) consistently beats weight-blending (after weight-blend fixes: PPL ~173-220 vs bridge **60.2** for S1). For same-arch diff-size (S2), the bridge beats **both** parents. MergeKit cannot handle cross-arch or cross-size merging (requires identical arch + task-vector format).

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
- **Bridge overfits small calibration sets**: With only 24-48 calibration texts, the bridge shows some overfitting. More calibration data would improve generalization.
- **Weight blending is not recommended**: The bridge approach consistently beats weight-blending in all scenarios tested. Weight-blend is kept for comparison only.
- **Architecture support**: Now architecture-agnostic — supports GPT-2, Llama (SmolLM2), OPT, and any HuggingFace-compatible CausalLM. Architecture detection uses `model.config.model_type` with dynamic layer/head resolution. Tested on GPT-2 (124M), DistilGPT-2 (82M), OPT-125M, SmolLM2-135M.
- **Memory (measured peak)**: ~1.08 GB for 164M params, ~1.19 GB for 206M params, ~1.36 GB for 437M params (RTX 3050 4GB, float16, max_len=64). Projected ~45-80 GB for 7B models — requires multi-GPU.
- **Bridge bypasses ln_f**: The final layer norm is not included in the bridge; fixing this would require retraining all bridge weights from scratch. This limits output quality ceiling.
- **PPL ≠ generation quality**: The bridge beats parents on PPL but generations are often worse than parent A in practice. PPL improvement is necessary but not sufficient for better text.
- **7B+ scaling**: Bridge requires backprop through the full model — each training step is as expensive as a forward+backward pass. Gradient checkpointing + multi-GPU required for models >1B.

## API reference

All canonical functions are in `fusellm.merge_prod`:

| Function | Purpose |
|---|---|
| `merge_same_arch(ma, mb, calib_texts, save_name)` | Activation-similarity-guided same-arch merge |
| `merge_same_arch_bridge(ma, mb, tok, calib_texts, steps, lr)` | Bridge-based same-arch merge (beats weight-blending for diff-size) |
| `train_bridge_v2(ma, mb, tok, texts, token_map, steps, lr, weight_decay, max_len)` | Zero-init bridge + cosine-LR fine-tune |
| `merge_diff_arch(ma, mb, calib_texts, token_map, save_name, tok, steps, lr)` | Full diff-arch pipeline (trains bridge + saves) |
| `generate_bridge(ma, mb, bridge, tok, prompt, token_map)` | Generate with bridge (blended with A) |
| `stitch_generate(ma, mb, bridge, tok, prompt, token_map)` | Generate with trained bridge |
| `verify_generations(model, ma, mb, tok)` | Print sample outputs |

## License

MIT — see [LICENSE](LICENSE)
