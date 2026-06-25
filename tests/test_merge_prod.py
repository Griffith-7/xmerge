import pytest, torch, gc, math, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from fusellm import merge_prod, utils

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
skip_slow = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")

CALIB_TEXTS = [
    'The theory of general relativity describes gravity as the curvature of spacetime.',
    'Photosynthesis is the process by which green plants use sunlight to synthesize nutrients.',
    'Artificial intelligence refers to the simulation of human intelligence in machines.',
]


def clean():
    gc.collect()
    if DEVICE == 'cuda': torch.cuda.empty_cache()


# ─── CKA Tests ──────────────────────────────────────────────────────────────

class TestCkaScore:
    def test_identical_vectors(self):
        x = torch.randn(6400)
        score = merge_prod.cka_score(x, x)
        assert abs(score - 1.0) < 1e-4

    def test_random_vectors(self):
        x = torch.randn(6400)
        y = torch.randn(6400)
        score = merge_prod.cka_score(x, y)
        assert -1 <= score <= 1

    def test_zero_input(self):
        x = torch.zeros(6400)
        y = torch.randn(6400)
        score = merge_prod.cka_score(x, y)
        assert math.isfinite(score)


# ─── Bridge Tests ───────────────────────────────────────────────────────────

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

    def test_identity_forward_asymmetric_dims(self):
        bridge = merge_prod.OptimalBridge(768, 512)
        h_a = torch.randn(2, 10, 768)
        h_b = torch.randn(2, 10, 512)
        out = bridge(h_a, h_b)
        assert out.shape == h_a.shape
        assert torch.allclose(out, h_a)


@pytest.mark.slow
class TestBuildBridge:
    def test_zero_init_from_build_bridge(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        ma = AutoModelForCausalLM.from_pretrained('distilgpt2', torch_dtype=torch.float16).to(DEVICE).eval()
        mb = AutoModelForCausalLM.from_pretrained('facebook/opt-125m', torch_dtype=torch.float16,
                                                    use_safetensors=True).to(DEVICE).eval()
        tok = AutoTokenizer.from_pretrained('distilgpt2'); tok.pad_token = tok.eos_token
        bridge = merge_prod.build_bridge(ma, mb, tok, CALIB_TEXTS)
        assert bridge.proj.weight.norm().item() == 0.0
        del ma, mb, bridge; clean()

    def test_bridge_produces_finite_ppl(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch.nn.functional as F
        ma = AutoModelForCausalLM.from_pretrained('distilgpt2', torch_dtype=torch.float16).to(DEVICE).eval()
        mb = AutoModelForCausalLM.from_pretrained('facebook/opt-125m', torch_dtype=torch.float16,
                                                    use_safetensors=True).to(DEVICE).eval()
        tok = AutoTokenizer.from_pretrained('distilgpt2'); tok.pad_token = tok.eos_token
        bridge = merge_prod.build_bridge(ma, mb, tok, CALIB_TEXTS)
        enc = tok(CALIB_TEXTS, truncation=True, padding=True, max_length=64, return_tensors='pt')
        ids, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)
        dtype = next(ma.parameters()).dtype
        loss, logits = merge_prod._stitch_forward(ma, mb, bridge, ids, mask, ids, dtype)
        ppl = math.exp(loss.item())
        assert math.isfinite(ppl)
        assert isinstance(ppl, float) and ppl > 0
        del ma, mb, bridge; clean()


@pytest.mark.slow
class TestTrainBridgeV2:
    def test_training_reduces_loss(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        ma = AutoModelForCausalLM.from_pretrained('distilgpt2', torch_dtype=torch.float16).to(DEVICE).eval()
        mb = AutoModelForCausalLM.from_pretrained('facebook/opt-125m', torch_dtype=torch.float16,
                                                    use_safetensors=True).to(DEVICE).eval()
        tok = AutoTokenizer.from_pretrained('distilgpt2'); tok.pad_token = tok.eos_token

        texts = CALIB_TEXTS * 3
        bridge = merge_prod.train_bridge_v2(ma, mb, tok, texts, steps=5)
        w_norm = bridge.proj.weight.norm().item()
        assert w_norm > 0
        assert w_norm < 100

        enc = tok(CALIB_TEXTS, truncation=True, padding=True, max_length=64, return_tensors='pt')
        ids, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)
        dtype = next(ma.parameters()).dtype
        loss, _ = merge_prod._stitch_forward(ma, mb, bridge, ids, mask, ids, dtype)
        ppl = math.exp(loss.item())
        assert math.isfinite(ppl)
        del ma, mb, bridge; clean()


# ─── Same-Arch Merge Tests ──────────────────────────────────────────────────

@pytest.mark.slow
class TestMergeSameArch:
    def test_merge_does_not_crash(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        ma = AutoModelForCausalLM.from_pretrained('distilgpt2', torch_dtype=torch.float16).to(DEVICE).eval()
        mb = AutoModelForCausalLM.from_pretrained('distilgpt2', torch_dtype=torch.float16).to(DEVICE).eval()
        merged = merge_prod.merge_same_arch(ma, mb, calib_texts=CALIB_TEXTS[:10], save_name=None)
        assert merged is not None
        del ma, mb, merged; clean()

    def test_merged_model_generates(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        ma = AutoModelForCausalLM.from_pretrained('distilgpt2', torch_dtype=torch.float16).to(DEVICE).eval()
        mb = AutoModelForCausalLM.from_pretrained('distilgpt2', torch_dtype=torch.float16).to(DEVICE).eval()
        tok = AutoTokenizer.from_pretrained('distilgpt2'); tok.pad_token = tok.eos_token
        merged = merge_prod.merge_same_arch(ma, mb, calib_texts=CALIB_TEXTS[:10], save_name=None)
        inp = tok("Hello world", return_tensors='pt').to(DEVICE)
        out = merged.generate(**inp, max_new_tokens=5, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0], skip_special_tokens=True)
        assert isinstance(text, str) and len(text) > 0
        del ma, mb, merged; clean()


# ─── Generation Tests ───────────────────────────────────────────────────────

@pytest.mark.slow
class TestStitchGenerate:
    def test_generate_does_not_crash(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        ma = AutoModelForCausalLM.from_pretrained('distilgpt2', torch_dtype=torch.float16).to(DEVICE).eval()
        mb = AutoModelForCausalLM.from_pretrained('facebook/opt-125m', torch_dtype=torch.float16,
                                                    use_safetensors=True).to(DEVICE).eval()
        tok = AutoTokenizer.from_pretrained('distilgpt2'); tok.pad_token = tok.eos_token
        bridge = merge_prod.build_bridge(ma, mb, tok, CALIB_TEXTS)
        text = merge_prod.stitch_generate(ma, mb, bridge, tok, "Hello", max_new=5)
        assert isinstance(text, str) and len(text) > 5
        del ma, mb, bridge; clean()

    def test_generate_bridge_does_not_crash(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        ma = AutoModelForCausalLM.from_pretrained('distilgpt2', torch_dtype=torch.float16).to(DEVICE).eval()
        mb = AutoModelForCausalLM.from_pretrained('facebook/opt-125m', torch_dtype=torch.float16,
                                                    use_safetensors=True).to(DEVICE).eval()
        tok = AutoTokenizer.from_pretrained('distilgpt2'); tok.pad_token = tok.eos_token
        bridge = merge_prod.build_bridge(ma, mb, tok, CALIB_TEXTS)
        text = merge_prod.generate_bridge(ma, mb, bridge, tok, "Hello", max_new=5)
        assert isinstance(text, str) and len(text) > 5
        del ma, mb, bridge; clean()
