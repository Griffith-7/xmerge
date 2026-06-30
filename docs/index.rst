Welcome to xmerge's documentation!
===================================

**xmerge** merges LLMs of different architectures and sizes **without full training**.

.. code-block:: python

    from xmerge import train_bridge_cached, stitch_generate

    bridge = train_bridge_cached(model_a, model_b, tok, texts)
    text = stitch_generate(model_a, model_b, bridge, tok, "The future of AI is")
    print(text)


Features
--------

* **Cross-architecture merging** — Merge GPT-2 with Llama, Mistral with SmolLM2, etc.
* **Zero-init bridge** — Starts as identity, fine-tuned via next-token prediction.
* **Cached training** — 10-100x faster by caching hidden states once.
* **Streaming (low VRAM)** — Merge 7B models on 4GB VRAM via layer-by-layer streaming.
* **MLP bridge** — More capacity than linear bridge (~15% better PPL).
* **Weight blending** — CKA-guided per-layer alpha blending for same-arch models.
* **CLI** — ``xmerge merge --config config.json``, ``xmerge eval``, ``xmerge list``.

Contents
--------

.. toctree::
   :maxdepth: 2

   installation
   usage
   api
