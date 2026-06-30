Usage
=====

Quick Start
-----------

Train a bridge between two models and generate text:

.. code-block:: python

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from xmerge import merge_prod

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_a = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device).eval()
    model_b = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device).eval()
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    tok.pad_token = tok.eos_token

    texts = ["The quick brown fox jumps over the lazy dog."] * 8
    bridge = merge_prod.train_bridge_cached(model_a, model_b, tok, texts, steps=10)
    text = merge_prod.stitch_generate(model_a, model_b, bridge, tok, "The future of AI is")
    print(text)

CLI
---

Merge two models using a config file:

.. code-block:: bash

    xmerge merge --config config.json

Example ``config.json``:

.. code-block:: json

    {
        "model_a": "distilgpt2",
        "model_b": "distilgpt2",
        "method": "bridge",
        "training": {
            "steps": 20,
            "lr": 3e-4,
            "bridge_type": "linear",
            "use_cached": true
        },
        "calibration": {
            "n_texts": 48
        }
    }

Evaluate a saved bridge:

.. code-block:: bash

    xmerge eval --bridge-dir merged_models/my_merge --ppl

List saved merges:

.. code-block:: bash

    xmerge list

Memory-Efficient Streaming
--------------------------

For large models (7B+) on low VRAM (4GB):

.. code-block:: python

    from xmerge import merge_stream

    model_a, tok = merge_stream.load_model_streamed("mistralai/Mistral-7B-v0.1")
    model_b, _ = merge_stream.load_model_streamed("HuggingFaceTB/SmolLM2-360M")
    bridge, ppl = merge_stream.streamed_train_bridge_cached(
        model_a, model_b, tok, texts, device="cuda", steps=20
    )

    gen = merge_stream.StreamedGenerator(model_a, model_b, bridge, tok, device="cuda")
    print(gen.generate("The future of AI is", max_new=20))
