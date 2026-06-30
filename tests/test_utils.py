"""Comprehensive tests for xmerge utilities and core functions."""

import torch
import pytest

from xmerge import utils, merge_prod

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════════════════
# SVD PROJECT
# ═══════════════════════════════════════════════════════════════════════════

class TestSVDProject:
    def test_identity_shape(self):
        W = torch.randn(64, 128)
        result = utils.svd_project(W, 64, 128)
        assert result.shape == (64, 128)
        assert torch.allclose(W, result, atol=1e-5)

    def test_expand(self):
        W = torch.randn(64, 128)
        result = utils.svd_project(W, 256, 512)
        assert result.shape == (256, 512)
        assert result[:64, :128].shape == (64, 128)

    def test_shrink(self):
        W = torch.randn(256, 512)
        result = utils.svd_project(W, 64, 128)
        assert result.shape == (64, 128)

    def test_same_dim(self):
        W = torch.randn(64, 128)
        result = utils.svd_project(W, 128, 64)
        assert result.shape == (128, 64)

    def test_invalid_dimensions(self):
        W = torch.randn(64, 128)
        with pytest.raises(ValueError, match="2D"):
            utils.svd_project(W.view(1, 64, 128), 64, 128)

    def test_preserves_singular_values(self):
        W = torch.randn(32, 32)
        result = utils.svd_project(W, 32, 32)
        U1, S1, _ = torch.linalg.svd(W.float(), full_matrices=False)
        U2, S2, _ = torch.linalg.svd(result.float(), full_matrices=False)
        assert torch.allclose(S1, S2, atol=1e-4)


# ═══════════════════════════════════════════════════════════════════════════
# PROPORTIONAL MAP
# ═══════════════════════════════════════════════════════════════════════════

class TestProportionalMap:
    def test_same_n(self):
        m = utils.proportional_map(6, 6)
        assert m == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

    def test_double_n(self):
        m = utils.proportional_map(6, 12)
        assert len(m) == 6
        assert all(0 <= v < 12 for v in m.values())

    def test_half_n(self):
        m = utils.proportional_map(12, 6)
        assert len(m) == 12
        assert all(0 <= v < 6 for v in m.values())

    def test_no_nans(self):
        for a, b in [(1, 100), (100, 1), (7, 13), (13, 7)]:
            m = utils.proportional_map(a, b)
            assert len(m) == a
            assert all(0 <= v < b for v in m.values())

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            utils.proportional_map(0, 10)
        with pytest.raises(ValueError):
            utils.proportional_map(10, 0)


# ═══════════════════════════════════════════════════════════════════════════
# HIDDEN DIM / LAYERS
# ═══════════════════════════════════════════════════════════════════════════

class TestHiddenDim:
    def test_gpt2(self):
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained("distilgpt2")
        assert utils.hidden_dim(cfg) == 768

    def test_opt(self):
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained("facebook/opt-125m")
        assert utils.hidden_dim(cfg) == 768

    def test_llama(self):
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        assert utils.hidden_dim(cfg) == 576


class TestNumLayers:
    def test_gpt2(self):
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained("distilgpt2")
        assert utils.num_layers(cfg) == 6

    def test_smollm2(self):
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        assert utils.num_layers(cfg) == 30


# ═══════════════════════════════════════════════════════════════════════════
# TOKEN MAP
# ═══════════════════════════════════════════════════════════════════════════

class TestTokenMap:
    def test_gpt2_to_gpt2_single(self):
        from transformers import AutoTokenizer
        tok_a = AutoTokenizer.from_pretrained("distilgpt2")
        tok_b = AutoTokenizer.from_pretrained("distilgpt2")
        tm = utils.build_token_map(tok_a, tok_b, strategy="single")
        assert tm[0] == 0
        for tid in [10, 100, 1000, 10000]:
            if tid in tok_a.vocab:
                assert tm.get(tid) == tid

    def test_gpt2_to_gpt2_multi(self):
        from transformers import AutoTokenizer
        tok_a = AutoTokenizer.from_pretrained("distilgpt2")
        tok_b = AutoTokenizer.from_pretrained("distilgpt2")
        tm = utils.build_token_map(tok_a, tok_b, strategy="multi")
        assert len(tm) == tok_a.vocab_size

    def test_diff_tokenizers(self):
        from transformers import AutoTokenizer
        tok_a = AutoTokenizer.from_pretrained("distilgpt2")
        tok_b = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        if tok_b.pad_token is None:
            tok_b.pad_token = tok_b.eos_token
        tm = utils.build_token_map(tok_a, tok_b)
        match_rate = sum(1 for v in tm.values() if v > 0) / len(tm) * 100
        assert match_rate > 50, f"Coverage too low: {match_rate:.1f}%"
        assert len(tm) == tok_a.vocab_size

    def test_invalid_strategy(self):
        from transformers import AutoTokenizer
        tok_a = AutoTokenizer.from_pretrained("distilgpt2")
        tok_b = AutoTokenizer.from_pretrained("distilgpt2")
        with pytest.raises(ValueError, match="strategy"):
            utils.build_token_map(tok_a, tok_b, strategy="invalid")


# ═══════════════════════════════════════════════════════════════════════════
# CKA SIMILARITY
# ═══════════════════════════════════════════════════════════════════════════

class TestActivationSimilarity:
    def test_identical(self):
        x = torch.randn(2, 10, 768)
        score = merge_prod.activation_similarity(x, x)
        assert abs(score - 1.0) < 1e-4

    def test_different(self):
        x = torch.randn(2, 10, 768)
        y = torch.randn(2, 10, 768)
        score = merge_prod.activation_similarity(x, y)
        assert 0.0 <= score <= 1.0

    def test_zero_input(self):
        x = torch.zeros(2, 10, 768)
        y = torch.randn(2, 10, 768)
        score = merge_prod.activation_similarity(x, y)
        assert torch.isfinite(score) and score >= 0.0

    def test_orthogonal(self):
        x = torch.randn(4, 32, 768)
        y = torch.randn(4, 32, 768)
        score_self = merge_prod.activation_similarity(x, x)
        score_cross = merge_prod.activation_similarity(x, y)
        assert score_self >= score_cross  # Self should be higher


# ═══════════════════════════════════════════════════════════════════════════
# BRIDGE MODULES
# ═══════════════════════════════════════════════════════════════════════════

class TestOptimalBridge:
    def test_zero_init(self):
        bridge = merge_prod.OptimalBridge(768, 768)
        assert bridge.proj.weight.norm().item() == 0.0

    def test_identity_forward_same_dims(self):
        bridge = merge_prod.OptimalBridge(768, 768)
        h_a = torch.randn(2, 10, 768)
        h_b = torch.randn(2, 10, 768)
        out = bridge(h_a, h_b)
        assert out.shape == h_a.shape
        assert torch.allclose(out, h_a)

    def test_identity_forward_asymmetric(self):
        bridge = merge_prod.OptimalBridge(768, 512)
        h_a = torch.randn(2, 10, 768)
        h_b = torch.randn(2, 10, 512)
        out = bridge(h_a, h_b)
        assert out.shape == h_a.shape
        assert torch.allclose(out, h_a)

    def test_invalid_dims(self):
        with pytest.raises(ValueError):
            merge_prod.OptimalBridge(0, 768)
        with pytest.raises(ValueError):
            merge_prod.OptimalBridge(768, -1)

    def test_after_training_projection_changes(self):
        bridge = merge_prod.OptimalBridge(768, 512)
        h_a = torch.randn(2, 10, 768)
        h_b = torch.randn(2, 10, 512)
        out_before = bridge(h_a, h_b)
        # Simulate training by setting non-zero weights
        with torch.no_grad():
            bridge.proj.weight.normal_(0, 0.1)
        out_after = bridge(h_a, h_b)
        assert not torch.allclose(out_before, out_after)


class TestMLPBridge:
    def test_zero_init(self):
        bridge = merge_prod.MLPBridge(768, 768)
        assert bridge.linear.weight.norm().item() == 0.0
        assert bridge.mlp[-1].weight.norm().item() == 0.0

    def test_identity_forward(self):
        bridge = merge_prod.MLPBridge(768, 512)
        h_a = torch.randn(2, 10, 768)
        h_b = torch.randn(2, 10, 512)
        out = bridge(h_a, h_b)
        assert out.shape == h_a.shape
        assert torch.allclose(out, h_a)

    def test_after_training(self):
        bridge = merge_prod.MLPBridge(768, 512)
        h_a = torch.randn(2, 10, 768)
        h_b = torch.randn(2, 10, 512)
        out_before = bridge(h_a, h_b)
        with torch.no_grad():
            bridge.linear.weight.normal_(0, 0.1)
        out_after = bridge(h_a, h_b)
        assert not torch.allclose(out_before, out_after)


# ═══════════════════════════════════════════════════════════════════════════
# DEVICE / CLEAN
# ═══════════════════════════════════════════════════════════════════════════

class TestDevice:
    def test_resolve_device(self):
        device = utils.resolve_device("cpu")
        assert device.type == "cpu"
        assert str(device) == "cpu"

    def test_clean_no_crash(self):
        utils.clean()  # Should not raise


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

class TestValidateModelPair:
    def test_same_model(self):
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained("distilgpt2")
        info = utils.validate_model_pair(
            type("m", (), {"config": cfg})(),
            type("m", (), {"config": cfg})(),
        )
        assert info["same_arch"] is True
        assert info["d_a"] == 768

    def test_invalid_config(self):
        from transformers import AutoConfig
        cfg_a = AutoConfig.from_pretrained("distilgpt2")
        # Missing config should still work
        cfg_b = AutoConfig.from_pretrained("distilgpt2")
        info = utils.validate_model_pair(
            type("m", (), {"config": cfg_a})(),
            type("m", (), {"config": cfg_b})(),
        )
        assert "d_a" in info
        assert "n_layers_a" in info
