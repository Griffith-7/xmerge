# fusellm

> Merge LLMs of different architectures and sizes — representation-level merging, not weight-space interpolation.

## The problem

Existing LLM merging tools (mergekit) operate in **weight space**: they interpolate or combine parameters directly. This only works when models share the exact same architecture and size. Even then, weight-space methods (TIES, DARE, SLERP) are designed for **task-vector merging** — combining fine-tuned versions of the same base model. They **produce gibberish** when applied to independently-trained base models.

## The insight

We merge at the **representation level**, not the weight level:

- **Same architecture**: Activation-similarity-guided per-layer alpha blending + spectral repair
- **Different architectures**: Learned linear bridge from B's hidden space to A's hidden space (zero-init + cosine-LR fine-tune)

This lets us merge:
- GPT-2 + DialoGPT (different fine-tunes, same arch) → coherent, between parents
- GPT-2 + DistilGPT-2 (different sizes) → coherent, **beats both parents**
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
- **Fine-tune**: 20-step AdamW with cosine LR schedule minimizes A's LM head cross-entropy loss
- **Cross-tokenizer**: Token mapping via greedy string → token ID alignment (99.9% match rate)

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

# Option 1: Same-architecture merge (returns model + tokenizer)
merged, _ = merge_prod.merge_same_arch(ma, mb, calib_texts=["General relativity describes gravity as spacetime curvature."], save_name=None)
print(f"Merged PPL: {utils.compute_ppl(merged, tok, ['General relativity describes gravity as spacetime curvature.']):.1f}")

# Option 2: Cross-architecture bridge (zero-init + cosine-LR fine-tune)
bridge = merge_prod.train_bridge_v2(ma, mb, tok, ["General relativity describes gravity as spacetime curvature."], steps=20)
gen = merge_prod.stitch_generate(ma, mb, bridge, tok, "The future of artificial intelligence is")
print(f"Bridge output: {gen}")
```

## Results

Tested on NVIDIA RTX 3050 (4GB). PPL on WikiText-2 validation set (~1500 tokens). Lower is better.

| Scenario | Parent A | Parent B | **fusellm** |
|---|---|---|---|
| GPT-2 + DialoGPT (same arch, same size) | 51.9 | 5721.4 | **5386.5** ⚠ |
| GPT-2 + DistilGPT-2 (same arch, diff size) | 51.9 | 80.5 | **47.6** ✓✓ |
| DistilGPT-2 + OPT-125M (diff arch, same size) | 80.5 | **59.5** | **66.4** ✓ |
| DistilGPT-2 + SmolLM2-135M (diff arch, diff size) | 80.5 | **34.1** | **70.9** ✓ |

✓ = coherent, better than weaker parent. ✓✓ = beats both parents. ⚠ = worse than weaker parent.

**Key insight**: The bridge approach (S2-S4) consistently beats the weaker parent. For same-arch diff-size (S2), the bridge beats **both** parents. The weight-blending approach (S1) struggles when one parent has very poor LM quality (DialoGPT PPL 5721.4). MergeKit cannot handle any of these scenarios (requires identical arch + task-vector format).

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
- **Weight blending degrades with poor parents**: S1 weight-blending can produce very high PPL if one parent has poor LM quality (e.g., DialoGPT as dialogue model on WikiText).
- **Calibration overfitting**: The refinement pass in `merge_same_arch` optimizes PPL on 24 calibration texts; generalization to held-out data is weaker.
- **Architecture support**: Currently only tested on GPT-2 family for the primary model. OPT and SmolLM2 are bridge targets only.
- **4GB GPU limit**: All operations fit within 4GB VRAM with float16 and max sequence length 64, 24 calibration texts.

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
