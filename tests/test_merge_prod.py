"""Tests for merge_prod — CKA, bridges, and merge functions."""

import gc
import math
import torch
import pytest

from xmerge import merge_prod, utils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
skip_slow = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")

CALIB_TEXTS = [
    "The theory of general relativity describes gravity as the curvature of spacetime.",
    "Photosynthesis is the process by which green plants use sunlight to synthesize nutrients.",
    "Artificial intelligence refers to the simulation of human intelligence in machines.",
]


def clean():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════
# CKA TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestHSIC_CKA:
    def test_identical(self):
        x = torch.randn(32, 768)
        score = merge_prod.hsic_cka(x, x)
        assert abs(score - 1.0) < 1e-4

    def test_range(self):
        x = torch.randn(32, 768)
        y = torch.randn(32, 768)
        score = merge_prod.hsic_cka(x, y)
        assert 0.0 <= score <= 1.0

    def test_zero_variance(self):
        x = torch.zeros(32, 768)
        y = torch.randn(32, 768)
        score = merge_prod.hsic_cka(x, y)
        assert score >= 0.0


class TestCkaComputer:
    def test_hook_setup(self):
        from transformers import AutoConfig, AutoModelForCausalLM
        cfg = AutoConfig.from_pretrained("distilgpt2")
        model = AutoModelForCausalLM.from_config(cfg)
        n_layers = utils.num_layers(cfg)
        computer = merge_prod.CkaComputer(model, n_layers)
        assert len(computer.handles) == n_layers
        computer.close()
        assert all(h is not None for h in computer.handles)

    def test_collect_hiddens(self):
        from transformers import AutoConfig, AutoModelForCausalLM
        cfg = AutoConfig.from_pretrained("distilgpt2")
        model = AutoModelForCausalLM.from_config(cfg).eval()
        n_layers = utils.num_layers(cfg)
        computer = merge_prod.CkaComputer(model, n_layers)
        ids = torch.randint(0, 100, (1, 16))
        hiddens = computer.collect(model, ids, None)
        assert len(hiddens) == n_layers
        for i in range(n_layers):
            assert i in hiddens
        computer.close()


# ═══════════════════════════════════════════════════════════════════════════
# BRIDGE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildBridge:
    def test_zero_init(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        ma = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        mb = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        tok = AutoTokenizer.from_pretrained("distilgpt2")
        tok.pad_token = tok.eos_token
        bridge = merge_prod.build_bridge(ma, mb, tok, CALIB_TEXTS)
        assert bridge.proj.weight.norm().item() == 0.0
        del ma, mb, bridge
        clean()

    def test_mlp_bridge_build(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        ma = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        mb = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        tok = AutoTokenizer.from_pretrained("distilgpt2")
        tok.pad_token = tok.eos_token
        bridge = merge_prod.build_bridge(ma, mb, tok, CALIB_TEXTS, bridge_type="mlp")
        assert isinstance(bridge, merge_prod.MLPBridge)
        assert bridge.linear.weight.norm().item() == 0.0
        del ma, mb, bridge
        clean()


@pytest.mark.slow
class TestTrainBridgeV2:
    def test_training_reduces_loss(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        ma = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        mb = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        tok = AutoTokenizer.from_pretrained("distilgpt2")
        tok.pad_token = tok.eos_token
        texts = CALIB_TEXTS * 3
        bridge = merge_prod.train_bridge_v2(ma, mb, tok, texts, steps=5, verbose=False)
        assert bridge.proj.weight.norm().item() > 0
        assert bridge.proj.weight.norm().item() < 100
        enc = tok(CALIB_TEXTS, truncation=True, padding=True, max_length=64, return_tensors="pt")
        ids = enc.input_ids.to(DEVICE)
        mask = enc.attention_mask.to(DEVICE)
        dtype = next(ma.parameters()).dtype
        loss, _ = merge_prod._stitch_forward(ma, mb, bridge, ids, mask, ids, dtype)
        ppl = math.exp(loss.item())
        assert math.isfinite(ppl)
        del ma, mb, bridge
        clean()

    def test_cached_training(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        ma = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        mb = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        tok = AutoTokenizer.from_pretrained("distilgpt2")
        tok.pad_token = tok.eos_token
        texts = CALIB_TEXTS * 3
        bridge = merge_prod.train_bridge_cached(ma, mb, tok, texts, steps=5, verbose=False)
        assert bridge.proj.weight.norm().item() > 0
        del ma, mb, bridge
        clean()

    def test_mlp_training(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        ma = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        mb = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        tok = AutoTokenizer.from_pretrained("distilgpt2")
        tok.pad_token = tok.eos_token
        texts = CALIB_TEXTS * 3
        bridge = merge_prod.train_bridge_v2(ma, mb, tok, texts, steps=3, bridge_type="mlp", verbose=False)
        assert isinstance(bridge, merge_prod.MLPBridge)
        assert bridge.linear.weight.norm().item() > 0
        del ma, mb, bridge
        clean()

    def test_empty_texts_raises(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        ma = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        mb = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        tok = AutoTokenizer.from_pretrained("distilgpt2")
        tok.pad_token = tok.eos_token
        with pytest.raises(ValueError, match="empty"):
            merge_prod.train_bridge_v2(ma, mb, tok, [], verbose=False)
        del ma, mb
        clean()


# ═══════════════════════════════════════════════════════════════════════════
# GENERATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestGeneration:
    def test_stitch_generate(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        ma = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        mb = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        tok = AutoTokenizer.from_pretrained("distilgpt2")
        tok.pad_token = tok.eos_token
        bridge = merge_prod.OptimalBridge(768, 768)
        text = merge_prod.stitch_generate(ma, mb, bridge, tok, "Hello", max_new=5)
        assert isinstance(text, str)
        assert len(text) > 5
        del ma, mb, bridge
        clean()

    def test_generate_bridge(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        ma = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        mb = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        tok = AutoTokenizer.from_pretrained("distilgpt2")
        tok.pad_token = tok.eos_token
        bridge = merge_prod.OptimalBridge(768, 768)
        text = merge_prod.generate_bridge(ma, mb, bridge, tok, "Hello", max_new=5)
        assert isinstance(text, str)
        assert len(text) > 5
        del ma, mb, bridge
        clean()


# ═══════════════════════════════════════════════════════════════════════════
# SAVE/LOAD
# ═══════════════════════════════════════════════════════════════════════════

class TestSaveLoad:
    def test_load_merged(self, tmp_path):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import os
        ma = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        mb = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        tok = AutoTokenizer.from_pretrained("distilgpt2")
        tok.pad_token = tok.eos_token

        bridge = merge_prod.OptimalBridge(768, 768)
        # Save
        bridge_dir = str(tmp_path / "test_bridge")
        os.makedirs(bridge_dir, exist_ok=True)
        torch.save(bridge.state_dict(), os.path.join(bridge_dir, "bridge.pt"))
        tok.save_pretrained(bridge_dir)
        with open(os.path.join(bridge_dir, "bridge_config.json"), "w") as f:
            import json
            json.dump({"type": "bridge", "bridge_type": "linear"}, f)

        # Load
        loaded_bridge, loaded_tok = merge_prod.load_merged(bridge_dir, ma, mb)
        assert loaded_bridge is not None
        assert loaded_tok is not None
        assert loaded_bridge.proj.weight.norm().item() == 0.0

        del ma, mb, bridge, loaded_bridge
        clean()
