import pytest, torch, math, gc, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from fusellm import utils

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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

    def test_same_shape(self):
        W = torch.randn(64, 128)
        result = utils.svd_project(W, 128, 64)
        assert result.shape == (128, 64)


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


@pytest.mark.slow
class TestPPL:
    def test_small_model_ppl(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained('distilgpt2', torch_dtype=torch.float16).to(DEVICE).eval()
        tok = AutoTokenizer.from_pretrained('distilgpt2'); tok.pad_token = tok.eos_token
        enc = tok("Hello world", truncation=True, max_length=64, return_tensors="pt")
        p = utils.compute_ppl(model, enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE))
        assert isinstance(p, float)
        assert 10 < p < 500000
        del model; gc.collect(); torch.cuda.empty_cache()

    def test_finite_for_garbage_model(self):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        cfg = AutoConfig.from_pretrained('distilgpt2')
        model = AutoModelForCausalLM.from_config(cfg).to(DEVICE)
        tok = AutoTokenizer.from_pretrained('distilgpt2'); tok.pad_token = tok.eos_token
        enc = tok("Hello world", truncation=True, max_length=64, return_tensors="pt")
        ids = enc.input_ids.to(DEVICE)
        p = utils.compute_ppl(model, ids)
        assert math.isfinite(p)
        del model; gc.collect(); torch.cuda.empty_cache()


class TestTokenMap:
    def test_gpt2_to_gpt2_identity(self):
        from transformers import AutoTokenizer
        tok_a = AutoTokenizer.from_pretrained('distilgpt2')
        tok_b = AutoTokenizer.from_pretrained('distilgpt2')
        tm = utils.build_token_map(tok_a, tok_b)
        assert tm[0] == 0
        # Most common tokens should map to themselves
        for tid in [10, 100, 1000, 10000]:
            if tid in tok_a.vocab and tid in tok_b.vocab:
                assert tm.get(tid) == tid

    def test_diff_tokenizers(self):
        from transformers import AutoTokenizer
        tok_a = AutoTokenizer.from_pretrained('distilgpt2')
        tok_b = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M')
        if tok_b.pad_token is None: tok_b.pad_token = tok_b.eos_token
        tm = utils.build_token_map(tok_a, tok_b)
        match_rate = sum(1 for v in tm.values() if v > 0) / len(tm) * 100
        assert match_rate > 50
        assert len(tm) == tok_a.vocab_size


class TestHiddenDim:
    def test_gpt2(self):
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained('distilgpt2')
        assert utils.hidden_dim(cfg) == 768

    def test_opt(self):
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained('facebook/opt-125m')
        assert utils.hidden_dim(cfg) == 768
