"""
fusellm: Merge LLMs of different architectures and sizes WITHOUT training.

Exports:
  merge_v2      — CKA-guided same-arch merge + cross-arch bridge (10-step fine-tune)
  merge_prod    — Deterministic same-arch merge + LS bridge (zero training)
  merge         — Random search baseline (original)
  utils         — Shared: CKA, SVD, PPL, token maps, calibration data
"""
from . import merge_v2, merge_prod, merge, utils
