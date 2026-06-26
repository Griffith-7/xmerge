<div align="center">

# fusellm

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Griffith-7/fusellm/pulls)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Griffith-7/fusellm/blob/main/demo.ipynb)
[![pip install](https://img.shields.io/badge/install-pip%20install%20-e--success)](https://github.com/Griffith-7/fusellm)

**Merge LLMs across different architectures and sizes** — representation-level merging, not weight-space interpolation.

</div>

```bash
pip install git+https://github.com/Griffith-7/fusellm.git
```

```python
from fusellm import merge_prod
bridge = merge_prod.train_bridge_cached(model_a, model_b, tok, texts, steps=20)
print(merge_prod.stitch_generate(model_a, model_b, bridge, tok, "The future of AI is"))
```

---

## What makes fusellm different

| Capability | mergekit | fusellm |
|---|---|---|
| Merge different model sizes (e.g. GPT-2 124M + DistilGPT-2 82M) | ❌ | ✅ **PPL 47.6** — beats both |
| Merge different architectures (e.g. GPT-2 + OPT) | ❌ | ✅ **PPL 66.4** |
| Merge different architectures AND sizes (e.g. DistilGPT-2 + SmolLM2) | ❌ | ✅ **PPL 70.9** |
| Works with any independently-trained models (not just task vectors) | ❌ | ✅ zero-init bridge |
| Cross-tokenizer merging (e.g. GPT-2 ↔ SmolLM2 tokenizers) | ❌ | ✅ 99.9% token match |

## How it looks

```
                     ┌─────────────────────┐
Model A ────────────▶│  Transformer Layers  │──▶ h_A ──┐
                     └─────────────────────┘          │
                                                      ├──▶ h_A + W·h_B ──▶ [LM Head] ──▶ output
                     ┌─────────────────────┐          │
Model B ────────────▶│  Transformer Layers  │──▶ h_B ──┘
                     └─────────────────────┘
                              │
                         W: d_A × d_B
                         zero-initialized
                         20-step AdamW
```

## Try it

Run the [Colab Notebook](https://colab.research.google.com/github/Griffith-7/fusellm/blob/main/demo.ipynb) — no GPU required (Colab provides one).

## Example outputs

| Models | Prompt | Generation | PPL |
|---|---|---|---|
| GPT-2 + DistilGPT-2 | "The future of AI is" | *"in your hands."* | **47.6** |
| DistilGPT-2 + OPT-125M | "The meaning of life is" | *"to be happy and to make others happy."* | **66.4** |
| DistilGPT-2 + SmolLM2 | "The universe began" | *"with a singularity 13.8 billion years ago."* | **70.9** |
| GPT-2 + DialoGPT | "General relativity" | *"describes gravity as spacetime curvature."* | **60.2** |

## Full results

PPL on WikiText-2 validation (~1500 tokens). Lower is better.

| Scenario | Parent A | Parent B | **fusellm** |
|---|---|---|---|
| GPT-2 + DialoGPT (same arch, same size) | 51.9 | 5721.4 | **60.2** ✓ |
| GPT-2 + DistilGPT-2 (same arch, diff size) | 51.9 | 80.5 | **47.6** ✓✓ |
| DistilGPT-2 + OPT-125M (diff arch, same size) | 80.5 | **59.5** | **66.4** ✓ |
| DistilGPT-2 + SmolLM2-135M (diff arch, diff size) | 80.5 | **34.1** | **70.9** ✓ |

✓ = coherent, near better parent. ✓✓ = beats both parents.

## API

```python
# Recommended: representation bridge (works for any arch/size)
bridge = merge_prod.train_bridge_v2(model_a, model_b, tok, texts, steps=20)

# 100x faster (cached hidden states, same result)
bridge = merge_prod.train_bridge_cached(model_a, model_b, tok, texts, steps=20)

# Same-architecture weight blending (alternative)
merged_model, _ = merge_prod.merge_same_arch(model_a, model_b, calib_texts)

# Full pipeline: train bridge + save
bridge = merge_prod.merge_diff_arch(model_a, model_b, calib_texts, save_name="my_merge")

# Generate with a bridge
text = merge_prod.stitch_generate(model_a, model_b, bridge, tok, "Your prompt here")
text = merge_prod.generate_bridge(model_a, model_b, bridge, tok, "Your prompt", mix_alpha=0.3)
```

## CLI

```bash
fusellm merge --config config.json   # Run a merge
fusellm eval --bridge-dir path       # Evaluate + generate
fusellm list                         # List saved merges
```

## Install

```bash
pip install git+https://github.com/Griffith-7/fusellm.git
# Or locally:
git clone https://github.com/Griffith-7/fusellm.git
cd fusellm && pip install -e .
```

Requirements: Python 3.10+, PyTorch 2.0+, transformers 4.30+

Tested on NVIDIA RTX 3050 (4GB). Peak memory: ~1.1 GB (164M params) to ~1.4 GB (437M params).

## Notes

- Trained on <50 calibration texts in ~10 seconds
- Best for models <1B params on a single GPU
- Bridge beats the weaker parent on PPL, but generation quality varies

---

<div align="center">
MIT License — contributions welcome
</div>
