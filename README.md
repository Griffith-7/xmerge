# fusellm

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Griffith-7/fusellm/pulls)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

> Merge LLMs across **different architectures and sizes** — no weight-space interpolation needed.

Merge GPT-2 with DistilGPT-2. Merge GPT-2 with OPT. Merge DistilGPT-2 with SmolLM2. **Representation-level merging** works where mergekit can't.

```bash
pip install -e .
fusellm merge --config config.json
```

## What it does

| Scenario | Result |
|---|---|
| GPT-2 + DialoGPT (same arch) | Bridge PPL **60.2** ← near GPT-2 quality |
| GPT-2 + DistilGPT-2 (diff size) | Bridge PPL **47.6** ← **beats both parents** |
| DistilGPT-2 + OPT-125M (diff arch) | Bridge PPL **66.4** ← between parents |
| DistilGPT-2 + SmolLM2-135M (diff arch + size) | Bridge PPL **70.9** ← between parents |

MergeKit can't handle any of these (requires identical architecture + task-vector format).

## How it works

```
Model A ──→ [layers] ──→ h_A ──┐
                                 ├──→ h_A + W·h_B ──→ [ln_f] ──→ [lm_head] ──→ output
Model B ──→ [layers] ──→ h_B ──┘
                                 W: zero-initialized, trained 20 steps
```

A **linear bridge** learns to project B's hidden representations into A's space. Zero-initialized (starts as pure A), then fine-tuned on ~48 calibration texts via AdamW + cosine LR. Takes **~10 seconds** on a laptop GPU.

## Quick start

```python
from fusellm import merge_prod
from transformers import AutoModelForCausalLM, AutoTokenizer

ma = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16).to("cuda").eval()
mb = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=torch.float16).to("cuda").eval()
tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token

bridge = merge_prod.train_bridge_v2(ma, mb, tok, ["General relativity describes gravity as spacetime curvature."], steps=20)
print(merge_prod.stitch_generate(ma, mb, bridge, tok, "The future of AI is"))
```

Or via the CLI:

```bash
fusellm merge --config config.json
fusellm list
fusellm eval --bridge-dir merged_models/my_merge
```

## Why not just use mergekit?

| Capability | mergekit | fusellm |
|---|---|---|
| Same arch, same size | ✅ (task vectors only) | ✅ (any checkpoints) |
| Same arch, different sizes | ❌ | ✅ **PPL 47.6** |
| Different architectures | ❌ | ✅ **PPL 66.4** |
| Different architectures + sizes | ❌ | ✅ **PPL 70.9** |
| Cross-tokenizer (e.g. GPT-2 ↔ SmolLM2) | ❌ | ✅ **99.9% match rate** |
| Works with any independently-trained models | ❌ (requires task vectors) | ✅ (zero-init bridge) |

## API

| Function | What it does |
|---|---|
| `train_bridge_v2(...)` | Train a representation bridge (works for any arch/size) |
| `train_bridge_cached(...)` | Same result, **100x faster** (cached hidden states) |
| `merge_same_arch(...)` | Weight-blend merge for same-architecture models |
| `stitch_generate(...)` | Generate text through a trained bridge |
| `generate_bridge(...)` | Generate with mix-alpha blending |

See full reference in [`merge_prod.py`](src/fusellm/merge_prod.py).

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers 4.30+

## Tested on

- **GPU**: NVIDIA RTX 3050 (4GB VRAM) — runs all 4 scenarios
- **Models**: GPT-2 (124M), DistilGPT-2 (82M), DialoGPT-small, OPT-125M, SmolLM2-135M
- **Memory (peak)**: ~1.1 GB (164M params) to ~1.4 GB (437M params)

## Limitations

- Best for models <1B params on single GPU. 7B+ needs multi-GPU.
- Bridge beats weaker parent but not always the stronger one on generation quality (PPL isn't everything).
- Overfits with <24 calibration texts. More data helps.

## License

MIT
