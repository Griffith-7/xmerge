"""
fusellm: Merge LLMs of different architectures and sizes WITHOUT full training.

Canonical API (merge_prod):
  merge_same_arch    — Deterministic CKA-guided merge for same-architecture models
  build_bridge       — Zero-init bridge for cross-architecture fusion
  train_bridge_v2    — Zero-init + fine-tune bridge (10 steps typical)
  merge_diff_arch    — Full pipeline: bridge + eval + save for diff architectures
  generate_bridge    — Generate text through a bridge
  stitch_generate    — Generate text through a trained bridge
  verify_generations — Print sample outputs from merged model or bridge

Utilities (utils):
  svd_project        — Project weight matrix to target dimensions via SVD
  proportional_map   — Map n_a layers to n_b layers proportionally
  compute_ppl        — Compute perplexity on given data
  generate_text      — Generate text from a model
  build_token_map    — Map token IDs from tokenizer A to tokenizer B
"""
from . import merge_prod, utils
