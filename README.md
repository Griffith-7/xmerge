# fusellm

> Merge LLMs of different architectures and sizes — no training required.

## The problem

Existing LLM merging tools (mergekit) operate in **weight space**: they interpolate or combine
parameters directly. This only works when models share the exact same architecture and size
(same number of layers, same hidden dimension, same attention heads, etc.).

Even when they do share the architecture, weight-space methods (TIES, DARE, SLERP) are designed
for **task-vector merging** — combining fine-tuned versions of the same base model. They
**produce gibberish** when applied to independently-trained base models.

| Scenario | MergeKit | fusellm |
|---|---|---|
| Same arch, same size | PPL=20000+ (gibberish) | **PPL=73.9** (coherent) |
| Same arch, diff size | ❌ Unsupported | **PPL=1903.5** (limited) |
| Diff arch, same size | ❌ Unsupported | **PPL=78.1** (beats parent!) |
| Diff arch, diff size | ❌ Unsupported | **PPL=67.9** (beats parent!) |

## The insight

We merge at the **representation level**, not the weight level:

- **Same architecture**: CKA-based layer alignment + per-layer alpha search + spectral repair
- **Different architectures**: Learned linear bridge from B's hidden space to A's hidden space
  (10-step fine-tune or deterministic least-squares initialization, both zero-training)

This lets us merge:
- GPT-2 + DialoGPT (different fine-tunes, same arch) → PPL between both parents
- GPT-2 + DistilGPT-2 (different sizes) → Works but limited quality
- DistilGPT-2 + OPT-125M (different architectures) → **Beats either parent** by 19%
- DistilGPT-2 + SmolLM2-135M (different architectures AND sizes) → **Beats parent** by 29%

## How it works

### Same architecture (merge.py, merge_v2.py)

Given models A and B with the same hidden dimension:

1. Collect hidden state activations from both models on calibration data
2. Compute CKA similarity between all layer pairs → proportional layer mapping
3. Per-layer alpha blending: `W_merged[i] = alpha[i] * W_A[i] + (1-alpha[i]) * W_B[map(i)]`
4. Multi-pass greedy search optimizes alphas (2-3 passes, ±0.3 per layer)
5. Spectral repair: blend singular values of merged weights (guarded — reverted if PPL increases)

### Different architectures (merge_v2.py, merge_prod.py)

When models have different hidden dimensions or layer counts, weight-space merging is
impossible. We use a **bridge module**: a learned linear projection from B's hidden space
to A's hidden space.

```
h_merged = h_A + W @ h_B    (W: d_A × d_B learned projection)
```

- **Zero-training**: Initialize W = 0 (bridge = identity on A). Evaluate with A's LM head.
- **10-step fine-tune**: Train W to minimize A's LM loss. Takes 1-3 seconds on GPU.
- **Cross-tokenizer**: Token mapping via greedy string → token ID alignment (build_token_map).

### Cross-tokenizer merging

When models use different tokenizers, we build a token map: for each token ID in A's
vocabulary, find the corresponding ID in B's tokenizer by encoding the decoded token string.
This lets B process input from A's tokenizer, enabling hidden state extraction from B
even with incompatible tokenizers.

## Quick start

```python
from fusellm import merge_v2, utils
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load two models
ma = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE).eval()
mb = AutoModelForCausalLM.from_pretrained("distilgpt2").to(DEVICE).eval()
tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token

# Calibration data
texts = ["General relativity describes gravity as spacetime curvature."]

# Same-architecture merge (GPT-2 + DistilGPT-2)
enc = tok(texts, truncation=True, padding=True, max_length=64, return_tensors="pt")
merged_sd, ppl = merge_v2.merge_same_arch(ma, mb, enc.input_ids, enc.attention_mask, tok)
print(f"Merged PPL: {ppl:.1f}")

# Cross-architecture bridge (DistilGPT-2 + OPT-125M)
bridge = merge_v2.train_bridge_v2(ma, mb, tok, texts, steps=10)
print(f"Bridge PPL: {merge_v2.bridge_ppl(ma, mb, bridge, tok, texts):.1f}")
```

## Results

Tested on RTX 3050 (4GB VRAM). Full benchmark in `benchmarks/run_benchmarks.py`.

| Scenario | Parent A | Parent B | **fusellm** | MergeKit |
|---|---|---|---|---|
| GPT-2 + DialoGPT (same arch, same size) | 42.2 | 378.8 | **109.3** ✓ | Gibberish |
| GPT-2 + DistilGPT-2 (same arch, diff size) | 42.2 | 96.2 | **198.0** ⚠ | Unsupported |
| DistilGPT-2 + OPT-125M (diff arch, same size) | 96.2 | - | **78.1** ✓ | Unsupported |
| DistilGPT-2 + SmolLM2-135M (diff arch, diff size) | 96.2 | - | **67.9** ✓ | Unsupported |

PPL on held-out evaluation texts (lower is better). ✓ = coherent, ⚠ = limited quality.

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- torch >= 2.0
- transformers >= 4.30
- numpy
- datasets (for calibration data loading)

## Benchmark

```bash
cd benchmarks
python run_benchmarks.py
```

Results saved to `benchmarks/benchmark_results.json`.

## Limitations

- **Same-arch, different sizes**: Our CKA-based approach works but with limited quality
  (PPL 1903). This is an open challenge. Layer correspondence via proportional mapping
  is a strong assumption that doesn't hold for independently-trained models.
- **Calibration data matters**: PPL varies with calibration text quality and quantity.
  Results above use 24 domain-diverse sentences.
- **Bridge generalization**: The 10-step bridge can overfit to calibration texts.
  We evaluate on held-out texts, but rigorous generalization studies are needed.
- **4GB GPU limit**: Differential evolution (fusellm.llm_merge_solver) crashes on 4GB VRAM.
  The greedy/iterative approaches work within this constraint.

## License

MIT — see [LICENSE](LICENSE)
