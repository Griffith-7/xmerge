"""Edge case and boundary condition tests for xmerge."""

import gc
import math

import pytest
import torch

from xmerge import utils, merge_prod

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def clean():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════
# SVD PROJECT — edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestSVDProjectEdge:
    def test_1d_tensor_raises(self):
        W = torch.randn(64)
        with pytest.raises(ValueError, match="2D"):
            utils.svd_project(W, 32, 32)

    def test_3d_tensor_raises(self):
        W = torch.randn(4, 64, 128)
        with pytest.raises(ValueError, match="2D"):
            utils.svd_project(W, 64, 128)

    def test_zero_dim_target(self):
        W = torch.randn(64, 128)
        result = utils.svd_project(W, 0, 0)
        assert result.shape == (0, 0)

    def test_one_dim_target(self):
        W = torch.randn(64, 128)
        result = utils.svd_project(W, 1, 1)
        assert result.shape == (1, 1)

    def test_target_larger_than_source(self):
        W = torch.randn(4, 8)
        result = utils.svd_project(W, 64, 128)
        assert result.shape == (64, 128)
        assert result[:4, :8].shape == (4, 8)

    def test_identical_after_svd(self):
        W = torch.randn(32, 32)
        result = utils.svd_project(W, 32, 32)
        assert torch.allclose(W, result, atol=1e-5)

    def test_singular_matrix(self):
        W = torch.zeros(64, 128)
        W[0, 0] = 1.0
        result = utils.svd_project(W, 32, 64)
        assert result.shape == (32, 64)
        assert torch.isfinite(result).all()

    def test_all_zeros(self):
        W = torch.zeros(64, 128)
        result = utils.svd_project(W, 32, 64)
        assert result.shape == (32, 64)
        assert result.norm().item() == 0.0

    def test_float16_matrix(self):
        W = torch.randn(64, 128, dtype=torch.float16)
        result = utils.svd_project(W, 32, 64)
        assert result.shape == (32, 64)


# ═══════════════════════════════════════════════════════════════════════════
# PROPORTIONAL MAP — edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestProportionalMapEdge:
    def test_single_layer(self):
        m = utils.proportional_map(1, 10)
        assert len(m) == 1
        assert 0 <= m[0] < 10

    def test_many_to_one(self):
        m = utils.proportional_map(10, 1)
        assert len(m) == 10
        assert all(v == 0 for v in m.values())

    def test_equal_layers(self):
        for n in [1, 2, 3, 50, 100]:
            m = utils.proportional_map(n, n)
            assert m == {i: i for i in range(n)}

    def test_large_ratio(self):
        m = utils.proportional_map(1000, 3)
        assert len(m) == 1000
        assert len(set(m.values())) <= 3

    def test_zero_n_a_raises(self):
        with pytest.raises(ValueError):
            utils.proportional_map(0, 10)

    def test_zero_n_b_raises(self):
        with pytest.raises(ValueError):
            utils.proportional_map(10, 0)

    def test_negative_n_a_raises(self):
        with pytest.raises(ValueError):
            utils.proportional_map(-1, 10)

    def test_negative_n_b_raises(self):
        with pytest.raises(ValueError):
            utils.proportional_map(10, -1)


# ═══════════════════════════════════════════════════════════════════════════
# HIDDEN DIM / NUM LAYERS — edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestHiddenDimEdge:
    def test_minimal_config(self):
        class FakeConfig:
            pass
        assert utils.hidden_dim(FakeConfig()) == 768

    def test_config_with_n_embd(self):
        class FakeConfig:
            n_embd = 512
        assert utils.hidden_dim(FakeConfig()) == 512

    def test_config_with_d_model(self):
        class FakeConfig:
            d_model = 1024
        assert utils.hidden_dim(FakeConfig()) == 1024

    def test_config_none_attrs(self):
        class FakeConfig:
            hidden_size = None
            n_embd = None
            d_model = None
        assert utils.hidden_dim(FakeConfig()) == 768


class TestNumLayersEdge:
    def test_minimal_config(self):
        class FakeConfig:
            pass
        assert utils.num_layers(FakeConfig()) is None

    def test_config_with_n_layer(self):
        class FakeConfig:
            n_layer = 12
        assert utils.num_layers(FakeConfig()) == 12

    def test_config_with_num_layers(self):
        class FakeConfig:
            num_layers = 24
        assert utils.num_layers(FakeConfig()) == 24

    def test_all_none(self):
        class FakeConfig:
            num_hidden_layers = None
            n_layer = None
            num_layers = None
        assert utils.num_layers(FakeConfig()) is None


# ═══════════════════════════════════════════════════════════════════════════
# HSIC_CKA — edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestHSIC_CKAEdge:
    def test_single_element(self):
        x = torch.randn(1, 768)
        score = merge_prod.hsic_cka(x, x)
        assert 0.0 <= score <= 1.0

    def test_same_value_all(self):
        x = torch.ones(32, 768) * 0.5
        y = torch.ones(32, 768) * 0.5
        score = merge_prod.hsic_cka(x, y)
        assert 0.0 <= score <= 1.0

    def test_one_row(self):
        x = torch.randn(2, 768)
        y = torch.randn(2, 768)
        score = merge_prod.hsic_cka(x, y)
        assert 0.0 <= score <= 1.0

    def test_different_n(self):
        x = torch.randn(10, 768)
        y = torch.randn(10, 768)
        score = merge_prod.hsic_cka(x, y)
        assert 0.0 <= score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# ACTIVATION SIMILARITY — edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestActivationSimilarityEdge:
    def test_batch_sequence_format(self):
        x = torch.randn(2, 10, 768)
        y = torch.randn(2, 10, 768)
        score = merge_prod.activation_similarity(x, y)
        assert 0.0 <= score <= 1.0

    def test_different_seq_lens(self):
        x = torch.randn(2, 10, 768)
        y = torch.randn(2, 5, 768)
        score = merge_prod.activation_similarity(x, y)
        assert 0.0 <= score <= 1.0

    def test_different_batch_sizes(self):
        x = torch.randn(4, 10, 768)
        y = torch.randn(2, 10, 768)
        score = merge_prod.activation_similarity(x, y)
        assert 0.0 <= score <= 1.0

    def test_all_zeros(self):
        x = torch.zeros(2, 10, 768)
        y = torch.randn(2, 10, 768)
        score = merge_prod.activation_similarity(x, y)
        assert torch.isfinite(score)


# ═══════════════════════════════════════════════════════════════════════════
# BRIDGE MODULES — edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestOptimalBridgeEdge:
    def test_zero_d_a_raises(self):
        with pytest.raises(ValueError):
            merge_prod.OptimalBridge(0, 768)

    def test_zero_d_b_raises(self):
        with pytest.raises(ValueError):
            merge_prod.OptimalBridge(768, 0)

    def test_negative_d_a_raises(self):
        with pytest.raises(ValueError):
            merge_prod.OptimalBridge(-1, 768)

    def test_negative_d_b_raises(self):
        with pytest.raises(ValueError):
            merge_prod.OptimalBridge(768, -1)

    def test_one_dim(self):
        bridge = merge_prod.OptimalBridge(1, 1)
        h_a = torch.randn(2, 10, 1)
        h_b = torch.randn(2, 10, 1)
        out = bridge(h_a, h_b)
        assert out.shape == (2, 10, 1)
        assert torch.allclose(out, h_a)

    def test_very_large_dims(self):
        bridge = merge_prod.OptimalBridge(4096, 4096)
        h_a = torch.randn(1, 4, 4096)
        h_b = torch.randn(1, 4, 4096)
        out = bridge(h_a, h_b)
        assert out.shape == (1, 4, 4096)

    def test_asymmetric_large_to_small(self):
        bridge = merge_prod.OptimalBridge(4096, 1024)
        h_a = torch.randn(1, 4, 4096)
        h_b = torch.randn(1, 4, 1024)
        out = bridge(h_a, h_b)
        assert out.shape == (1, 4, 4096)

    def test_different_seq_lengths_trim(self):
        bridge = merge_prod.OptimalBridge(768, 512)
        h_a = torch.randn(2, 20, 768)
        h_b = torch.randn(2, 10, 512)
        k = min(h_a.shape[1], h_b.shape[1])
        out = bridge(h_a[:, :k], h_b[:, :k])
        assert out.shape == (2, 10, 768)

    def test_no_batch_dim(self):
        bridge = merge_prod.OptimalBridge(768, 512)
        h_a = torch.randn(10, 768)
        h_b = torch.randn(10, 512)
        out = bridge(h_a, h_b)
        assert out.shape == (10, 768)

    def test_float16_forward(self):
        bridge = merge_prod.OptimalBridge(768, 512).half()
        h_a = torch.randn(2, 10, 768, dtype=torch.float16)
        h_b = torch.randn(2, 10, 512, dtype=torch.float16)
        out = bridge(h_a, h_b)
        assert out.shape == (2, 10, 768)


class TestMLPBridgeEdge:
    def test_zero_d_a_raises(self):
        with pytest.raises(ValueError):
            merge_prod.MLPBridge(0, 768)

    def test_zero_d_b_raises(self):
        with pytest.raises(ValueError):
            merge_prod.MLPBridge(768, 0)

    def test_negative_d_a_raises(self):
        with pytest.raises(ValueError):
            merge_prod.MLPBridge(-1, 768)

    def test_one_dim(self):
        bridge = merge_prod.MLPBridge(1, 1)
        h_a = torch.randn(2, 10, 1)
        h_b = torch.randn(2, 10, 1)
        out = bridge(h_a, h_b)
        assert out.shape == (2, 10, 1)

    def test_custom_hidden_dim(self):
        bridge = merge_prod.MLPBridge(768, 512, hidden_dim=128)
        h_a = torch.randn(2, 10, 768)
        h_b = torch.randn(2, 10, 512)
        out = bridge(h_a, h_b)
        assert out.shape == (2, 10, 768)

    def test_asymmetric_dims(self):
        bridge = merge_prod.MLPBridge(4096, 1024)
        h_a = torch.randn(1, 4, 4096)
        h_b = torch.randn(1, 4, 1024)
        out = bridge(h_a, h_b)
        assert out.shape == (1, 4, 4096)

    def test_different_seq_lengths_trim(self):
        bridge = merge_prod.MLPBridge(768, 512)
        h_a = torch.randn(2, 20, 768)
        h_b = torch.randn(2, 10, 512)
        k = min(h_a.shape[1], h_b.shape[1])
        out = bridge(h_a[:, :k], h_b[:, :k])
        assert out.shape == (2, 10, 768)


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATE MODEL PAIR — edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestValidateModelPairEdge:
    def setUp(self):
        pass

    def test_same_arch_different_size(self):
        cfg1 = type("cfg", (), {"model_type": "gpt2", "n_embd": 768, "n_layer": 12})()
        cfg2 = type("cfg", (), {"model_type": "gpt2", "n_embd": 512, "n_layer": 6})()
        m1 = type("m", (), {"config": cfg1})()
        m2 = type("m", (), {"config": cfg2})()
        info = utils.validate_model_pair(m1, m2)
        assert info["same_arch"] is True
        assert info["d_a"] == 768
        assert info["d_b"] == 512

    def test_diff_arch(self):
        cfg1 = type("cfg", (), {"model_type": "gpt2", "n_embd": 768, "n_layer": 12})()
        cfg2 = type("cfg", (), {"model_type": "llama", "hidden_size": 768, "num_hidden_layers": 12})()
        m1 = type("m", (), {"config": cfg1})()
        m2 = type("m", (), {"config": cfg2})()
        info = utils.validate_model_pair(m1, m2)
        assert info["same_arch"] is False

    def test_require_same_arch_raises(self):
        cfg1 = type("cfg", (), {"model_type": "gpt2", "n_embd": 768, "n_layer": 12})()
        cfg2 = type("cfg", (), {"model_type": "llama", "hidden_size": 768, "num_hidden_layers": 12})()
        m1 = type("m", (), {"config": cfg1})()
        m2 = type("m", (), {"config": cfg2})()
        with pytest.raises(utils.ArchitectureMismatchError):
            utils.validate_model_pair(m1, m2, require_same_arch=True)


# ═══════════════════════════════════════════════════════════════════════════
# COMPUTE PPL — edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestComputePPLEdge:
    def test_empty_ids_raises(self):
        from transformers import AutoConfig, AutoModelForCausalLM
        cfg = AutoConfig.from_pretrained("distilgpt2")
        model = AutoModelForCausalLM.from_config(cfg).eval()
        ids = torch.zeros(0, 10, dtype=torch.long)
        with pytest.raises(ValueError, match="empty"):
            utils.compute_ppl(model, ids)
        del model

    def test_two_tokens(self):
        from transformers import AutoConfig, AutoModelForCausalLM
        cfg = AutoConfig.from_pretrained("distilgpt2")
        model = AutoModelForCausalLM.from_config(cfg).eval()
        ids = torch.randint(0, 100, (1, 2))
        ppl_val = utils.compute_ppl(model, ids)
        assert math.isfinite(ppl_val)
        del model


# ═══════════════════════════════════════════════════════════════════════════
# LOAD / SAVE — edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadMergedEdge:
    def test_missing_directory_raises(self):
        mock_cfg = type("cfg", (), {"model_type": "gpt2", "n_embd": 768, "n_layer": 12})()
        mock_model = type("m", (), {"config": mock_cfg})()
        with pytest.raises((FileNotFoundError, RuntimeError, OSError)):
            merge_prod.load_merged("nonexistent_path", mock_model, mock_model)

    def test_load_without_config(self, tmp_path):
        bridge = merge_prod.OptimalBridge(768, 768)
        torch.save(bridge.state_dict(), tmp_path / "bridge.pt")
        with pytest.raises(Exception):
            from transformers import AutoConfig, AutoModelForCausalLM
            cfg = AutoConfig.from_pretrained("distilgpt2")
            ma = AutoModelForCausalLM.from_config(cfg).eval()
            mb = AutoModelForCausalLM.from_config(cfg).eval()
            merge_prod.load_merged(str(tmp_path), ma, mb)
            del ma, mb

    def test_corrupt_state_dict(self, tmp_path):
        (tmp_path / "bridge.pt").write_text("not a tensor file")
        cfg = type("cfg", (), {"model_type": "gpt2", "n_embd": 768, "n_layer": 12})()
        ma = type("m", (), {"config": cfg})()
        mb = type("m", (), {"config": cfg})()
        with pytest.raises(Exception):
            merge_prod.load_merged(str(tmp_path), ma, mb)


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING — edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestTrainingEdge:
    def test_single_text_train(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        ma = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        mb = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        tok = AutoTokenizer.from_pretrained("distilgpt2")
        tok.pad_token = tok.eos_token
        bridge = merge_prod.train_bridge_v2(ma, mb, tok, ["Hello world"], steps=2, verbose=False)
        assert bridge.proj.weight is not None
        del ma, mb, bridge
        clean()

    def test_very_long_text(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        ma = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        mb = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        tok = AutoTokenizer.from_pretrained("distilgpt2")
        tok.pad_token = tok.eos_token
        long_text = "word " * 1000
        bridge = merge_prod.train_bridge_v2(ma, mb, tok, [long_text], steps=2, max_len=64, verbose=False)
        assert bridge.proj.weight is not None
        del ma, mb, bridge
        clean()

    def test_empty_eval_texts(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        ma = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        mb = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        tok = AutoTokenizer.from_pretrained("distilgpt2")
        tok.pad_token = tok.eos_token
        bridge = merge_prod.train_bridge_v2(ma, mb, tok, ["Hello"], eval_texts=[], steps=2, verbose=False)
        assert hasattr(bridge, "eval_ppl")
        assert not math.isfinite(bridge.eval_ppl)
        del ma, mb, bridge
        clean()

    def test_zero_steps(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        ma = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        mb = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        tok = AutoTokenizer.from_pretrained("distilgpt2")
        tok.pad_token = tok.eos_token
        bridge = merge_prod.train_bridge_v2(ma, mb, tok, ["Hello"], steps=0, verbose=False)
        assert bridge.proj.weight.norm().item() == 0.0
        del ma, mb, bridge
        clean()

    def test_cached_single_text(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        ma = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        mb = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        tok = AutoTokenizer.from_pretrained("distilgpt2")
        tok.pad_token = tok.eos_token
        bridge = merge_prod.train_bridge_cached(ma, mb, tok, ["Hello"], steps=2, verbose=False)
        assert bridge.proj.weight is not None
        del ma, mb, bridge
        clean()

    def test_cached_zero_steps(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        ma = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        mb = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        tok = AutoTokenizer.from_pretrained("distilgpt2")
        tok.pad_token = tok.eos_token
        bridge = merge_prod.train_bridge_cached(ma, mb, tok, ["Hello"], steps=0, verbose=False)
        assert bridge.proj.weight.norm().item() == 0.0
        del ma, mb, bridge
        clean()

    def test_empty_texts_raises(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        ma = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        mb = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
        tok = AutoTokenizer.from_pretrained("distilgpt2")
        tok.pad_token = tok.eos_token
        with pytest.raises(ValueError, match="empty"):
            merge_prod.train_bridge_cached(ma, mb, tok, [], verbose=False)
        del ma, mb
        clean()


# ═══════════════════════════════════════════════════════════════════════════
# GENERATION — edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestGenerationEdge:
    def test_short_prompt(self):
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
