"""xmerge: Merge LLMs of different architectures and sizes WITHOUT full training.

Canonical API:
  merge_same_arch       — CKA-guided weight blending for same-architecture models
  train_bridge_v2       — Zero-init bridge + cosine-LR fine-tune (with backprop)
  train_bridge_cached   — 100x faster bridge training using cached hidden states
  merge_diff_arch       — Full cross-architecture bridge pipeline
  merge_diff_arch_streamed — Low-VRAM streaming cross-architecture merge

Generation:
  stitch_generate       — Generate through a trained bridge
  generate_bridge       — Generate with controllable blending (mix_alpha)

Evaluation:
  load_merged           — Load a saved bridge
  verify_generations    — Print sample outputs

Utilities:
  OptimalBridge, MLPBridge — Bridge network modules
  activation_similarity    — CKA similarity between hidden states
  load_texts               — Load WikiText-2 calibration data

Streaming (low VRAM):
  merge_stream.streamed_train_bridge_cached — 7B models on 4GB VRAM
  merge_stream.StreamedGenerator            — Streamed generation

CLI:
  xmerge merge --config config.json   — Run a merge from config
  xmerge eval --bridge-dir path       — Evaluate a saved bridge
  xmerge list                         — List saved merges
  xmerge clean                        — Clear GPU memory
"""

import logging

from . import merge_prod, merge_stream, utils
from .merge_prod import (
    DEVICE,
    SAVE_DIR,
    CkaComputer,
    MLPBridge,
    OptimalBridge,
    activation_similarity,
    build_bridge,
    clean,
    generate_bridge,
    load_merged,
    load_texts,
    merge_diff_arch,
    merge_diff_arch_streamed,
    merge_same_arch,
    merge_same_arch_bridge,
    ppl,
    stitch_generate,
    train_bridge_cached,
    train_bridge_v2,
    verify_generations,
)
from .utils import (
    build_token_map,
    compute_ppl,
    generate_text,
    hidden_dim,
    proportional_map,
    resolve_device,
    svd_project,
    validate_model_pair,
)

__all__ = [
    # Core merge functions
    "merge_same_arch",
    "merge_same_arch_bridge",
    "train_bridge_v2",
    "train_bridge_cached",
    "merge_diff_arch",
    "merge_diff_arch_streamed",
    # Generation
    "stitch_generate",
    "generate_bridge",
    "verify_generations",
    "load_merged",
    # Bridge modules
    "OptimalBridge",
    "MLPBridge",
    "CkaComputer",
    "activation_similarity",
    "build_bridge",
    # Utilities
    "svd_project",
    "proportional_map",
    "compute_ppl",
    "generate_text",
    "build_token_map",
    "hidden_dim",
    "validate_model_pair",
    "resolve_device",
    "ppl",
    "clean",
    "load_texts",
    "merge_prod",
    "merge_stream",
    "utils",
    # Constants
    "DEVICE",
    "SAVE_DIR",
]

__version__ = "0.2.0"

# Configure default logger
logging.getLogger(__name__).addHandler(logging.NullHandler())
