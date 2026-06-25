# fusellm

> Merge LLMs of different architectures and sizes — representation-level merging, not weight-space interpolation.

## The problem

Existing LLM merging tools (mergekit) operate in **weight space**: they interpolate or combine parameters directly. This only works when models share the exact same architecture and size. Even then, weight-space methods (TIES, DARE, SLERP) are designed for **task-vector merging** — combining fine-tuned versions of the same base model. They **produce gibberish** when applied to independently-trained base models.

## The insight

We merge at the **representation level**, not the weight level:

- **Same architecture**: Cosine-similarity-based layer alignment + per-layer alpha search + spectral repair
- **Different architectures**: Learned linear bridge from B's hidden space to A's hidden space (zero-init + 10-step fine-tune)

This lets us merge:
- GPT-2 + DialoGPT (different fine-tunes, same arch) → coherent output between both parents
- GPT-2 + DistilGPT-2 (different sizes) → coherent output between both parents
- DistilGPT-2 + OPT-125M (different architectures) → coherent, but does NOT beat OPT-125M alone
- DistilGPT-2 + SmolLM2-135M (different architectures AND sizes) → coherent, but does NOT beat SmolLM2-135M alone

## How it works

### Same architecture

Given models A and B with the same hidden dimension:

1. Collect hidden state activations from both models on calibration data
2. Compute cosine similarity between all layer pairs → proportional layer mapping
3. Per-layer alpha blending: `W_merged[i] = alpha[i] * W_A[i] + (1-alpha[i]) * W_B[map(i)]`
4. Multi-pass greedy search optimizes alphas (2-3 passes, ±0.3 per layer)
5. Spectral repair: blend singular values of merged weights

### Different architectures

When models have different hidden dimensions or layer counts, weight-space merging is impossible. We use a **bridge module**: a learned linear projection from B's hidden space to A's hidden space.

```
h_merged = h_A + W @ h_B    (W: d_A × d_B learned projection)
```

- **Zero-init + fine-tune**: Initialize W = 0 (bridge = identity on A). 10-step fine-tune minimizes A's LM loss.
- **Cross-tokenizer**: Token mapping via greedy string → token ID alignment (build_token_map).

### Cross-tokenizer merging

When models use different tokenizers, we build a token map: for each token ID in A's vocabulary, find the corresponding ID in B's tokenizer by encoding the decoded token string.

## Quick start

```python
from fusellm import utils
from fusellm.merge_prod import build_bridge, generate_bridge
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load two models
ma = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE).eval()
mb = AutoModelForCausalLM.from_pretrained("distilgpt2").to(DEVICE).eval()
tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token

# Calibration data
texts = ["General relativity describes gravity as spacetime curvature."]

# Cross-architecture bridge (zero-init + 10-step fine-tune)
from fusellm.merge_v2 import train_bridge_v2
bridge = train_bridge_v2(ma, mb, tok, texts, steps=10)
print(f"Bridge PPL: {utils.compute_ppl(ma, tok, texts):.1f}")
```

## Results

Tested on NVIDIA GPU. Full benchmark in `benchmarks/run_benchmarks.py`.

| Scenario | Parent A | Parent B | **fusellm** | MergeKit |
|---|---|---|---|---|
| GPT-2 + DialoGPT (same arch, same size) | 42.2 | 378.8 | **109.3** ✓ | Gibberish |
| GPT-2 + DistilGPT-2 (same arch, diff size) | 42.2 | 96.2 | **80.3** ✓ | Unsupported |
| DistilGPT-2 + OPT-125M (diff arch, same size) | 96.2 | **56.1** | **78.1** ⚠ | Unsupported |
| DistilGPT-2 + SmolLM2-135M (diff arch, diff size) | 96.2 | **18.0** | **67.9** ⚠ | Unsupported |

PPL on held-out evaluation texts (lower is better). ✓ = coherent, ⚠ = beats worse parent only, not the better one.

**Key insight**: The bridge consistently beats the weaker parent (DistilGPT-2), but does NOT beat the stronger parent (OPT-125M or SmolLM2-135M) alone. The same-arch merge (CKA-guided) produces coherent blends between two models where naive weight averaging fails completely.

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- torch >= 2.0
- transformers >= 4.30
- numpy
- datasets

## Benchmark

```bash
cd benchmarks
python run_benchmarks.py
```

Results saved to `benchmarks/benchmark_results.json`.

## Limitations

- **Bridge vs better parent**: The cross-arch bridge beats the weaker parent but not the stronger one. It's useful when you must use model A's tokenizer/vocabulary but want to incorporate signal from model B.
- **Calibration data matters**: PPL varies with calibration text quality and quantity. Results above use 10 domain-diverse sentences.
- **Bridge generalization**: The 10-step fine-tune can overfit to calibration texts. More rigorous generalization studies are needed.
- **Architecture support**: Currently only tested on GPT-2 family models. OPT and SmolLM2 are only used as bridge targets, not as the primary model.
- **CKA is approximate**: The current implementation uses cosine similarity, not proper CKA (Centered Kernel Alignment).
- **4GB GPU limit**: Differential evolution (llm_merge_solver.py) crashes on 4GB VRAM. The greedy/iterative approaches work within this constraint.

## Approaches overview

The repo contains 4 separate implementations:

| File | Method | Quality | Speed | Notes |
|---|---|---|---|---|
| `llm_merge_solver.py` | Differential evolution + activation repair | Potentially best but unstable | Slow | Requires scipy, may OOM on 4GB |
| `merge.py` | Random search + spectral repair | Moderate | Slow (40 trials) | Simplest baseline |
| `merge_v2.py` | CKA-guided init + iterative refinement + LS bridge | Good | Fast | LS bridge init is broken (produces garbage); use zero-init instead |
| `merge_prod.py` | Deterministic CKA + LS bridge (zero training) | Moderate | Fastest | Cleanest code, no fine-tuning needed for same-arch |

**Recommendation**: Use `merge_prod.py` for deterministic results. Use `merge_v2.py`'s training loop (with zero init, not LS init) for the bridge. The benchmark script (`run_benchmarks.py`) has the most reliable implementations.

## License

MIT — see [LICENSE](LICENSE)
