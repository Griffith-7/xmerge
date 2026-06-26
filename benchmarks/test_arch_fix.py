"""Quick verification: architecture-agnostic fixes work on GPT-2, SmolLM2, OPT"""
import torch, gc, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fusellm import merge_prod
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

def clean():
    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

tests = [
    ("GPT-2", "gpt2", "distilgpt2"),
    ("SmolLM2 (Llama arch)", "HuggingFaceTB/SmolLM2-135M", "distilgpt2"),
    ("OPT architecture", "facebook/opt-125m", "distilgpt2"),
]

for label, ma_name, mb_name in tests:
    print(f"\n{'='*60}")
    print(f"  {label}: A={ma_name}, B={mb_name}")
    print(f"{'='*60}")
    clean()
    try:
        ma = AutoModelForCausalLM.from_pretrained(ma_name, torch_dtype=DTYPE, use_safetensors=True).to(DEVICE).eval()
        mb = AutoModelForCausalLM.from_pretrained(mb_name, torch_dtype=DTYPE, use_safetensors=True).to(DEVICE).eval()
        
        # Test architecture detection
        n_a = merge_prod._get_n_layers(ma.config)
        n_b = merge_prod._get_n_layers(mb.config)
        prefix_a = merge_prod._get_layer_prefix(ma)
        prefix_b = merge_prod._get_layer_prefix(mb)
        layer_list_a = merge_prod._get_layer_list(ma)
        lm_head = merge_prod._get_lm_head(ma)
        
        print(f"  Layers: A={n_a}, B={n_b}")
        print(f"  Prefixes: A='{prefix_a}', B='{prefix_b}'")
        print(f"  Layer list A: {type(layer_list_a).__name__}[{len(layer_list_a)}]")
        print(f"  lm_head: {'found' if lm_head is not None else 'NOT FOUND'}")
        
        # Test CkaComputer
        try:
            cka = merge_prod.CkaComputer(ma, n_a)
            cka.close()
            print(f"  CkaComputer: OK")
        except Exception as e:
            print(f"  CkaComputer FAILED: {e}")
            del ma, mb; clean()
            continue
        
        # Test tokenizer
        tok = AutoTokenizer.from_pretrained(ma_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        
        # Test bridge training
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        texts = [t["text"].strip() for t in ds if len(t["text"].strip()) > 10][:8]
        
        try:
            bridge = merge_prod.train_bridge_v2(ma, mb, tok, texts[:4], steps=3)
            gen = merge_prod.stitch_generate(ma, mb, bridge, tok, "The future of AI is", max_new=10)
            print(f"  Bridge training: OK")
            print(f"  Generation: {gen[:60]}")
        except Exception as e:
            print(f"  Bridge FAILED: {type(e).__name__}: {str(e)[:120]}")
        
        # Test merge_same_arch (same-arch weight blend)
        if prefix_a == prefix_b and n_a and n_b and n_a > 0:
            try:
                merged, mtok = merge_prod.merge_same_arch(ma, mb, calib_texts=texts[:8], save_name=None)
                print(f"  merge_same_arch: OK")
            except Exception as e:
                print(f"  merge_same_arch FAILED: {type(e).__name__}: {str(e)[:120]}")
        else:
            print(f"  merge_same_arch: SKIP (different archs or no layers)")
        
        del ma, mb
        if 'bridge' in dir(): del bridge
        if 'merged' in dir(): del merged
        clean()
        
    except Exception as e:
        print(f"  LOAD FAILED: {type(e).__name__}: {str(e)[:150]}")
        clean()

print(f"\n{'='*60}")
print("  DONE")
print(f"{'='*60}")
