"""
xmerge: Merge LLMs of different architectures and sizes WITHOUT full training.

Canonical API (merge_prod):
  merge_same_arch         — Activation-similarity-guided weight-blend merge for same-architecture models (returns model, tokenizer)
  merge_same_arch_bridge  — Representation-level bridge merge for same-architecture models, better for different sizes (returns bridge, tokenizer)
  train_bridge_v2    — Zero-init bridge + cosine-LR fine-tune (20 steps, configurable)
  train_bridge_cached—— 100x faster bridge training using cached hidden states (no backprop through transformers)
  merge_diff_arch    — Full pipeline: trained bridge + eval + save for diff architectures (configurable steps/lr)
  generate_bridge    — Generate text through a bridge (with mix_alpha blending)
  stitch_generate    — Generate text through a trained bridge
  verify_generations — Print sample outputs from merged model or bridge

Utilities (utils):
  svd_project        — Project weight matrix to target dimensions via SVD
  proportional_map   — Map n_a layers to n_b layers proportionally
  compute_ppl        — Compute perplexity on given data
  generate_text      — Generate text from a model
  build_token_map    — Map token IDs from tokenizer A to tokenizer B

CLI:
  xmerge merge --config config.json   — Run a merge from config file
  xmerge eval --bridge-dir path       — Evaluate a saved bridge
  xmerge list                         — List saved merges
  xmerge clean                        — Clear GPU memory cache
"""
from . import merge_prod, utils, merge_stream
