"""Shared utilities for all merge approaches."""
import torch, gc, math, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

def clean():
    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

def svd_project(W, out_t, in_t):
    if W.shape == (out_t, in_t): return W
    W = W.float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    m, n = U.shape[0], Vh.shape[1]
    k = min(m, n, out_t, in_t)
    W2 = torch.zeros(out_t, in_t, dtype=W.dtype, device=W.device)
    reconstructed = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k]
    W2[:min(m, out_t), :min(n, in_t)] = reconstructed[:min(m, out_t), :min(n, in_t)]
    return W2

def proportional_map(n_a, n_b):
    return {i: min(int((i + 0.5) * n_b / n_a), n_b - 1) for i in range(n_a)}

def hidden_dim(cfg):
    return getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None) or getattr(cfg, "d_model", None) or 768

def num_layers(cfg):
    return getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None) or getattr(cfg, "num_layers", None)

@torch.no_grad()
def compute_ppl(model, ids, mask=None):
    model.eval()
    return math.exp(model(input_ids=ids, attention_mask=mask, labels=ids).loss.item())

@torch.no_grad()
def generate_text(model, tokenizer, prompt="The future of AI is", max_new=40):
    model.eval()
    inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=True,
                         temperature=0.8, top_p=0.9, repetition_penalty=1.1,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def load_calibration(name="gpt2", n=32, seq=128):
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t["text"] for t in ds if len(t["text"].strip()) > 50][:n*2]
    if not texts: texts = ["The quick brown fox jumps over the lazy dog."] * n
    enc = tokenizer(texts[:n], truncation=True, padding=True, max_length=seq, return_tensors="pt")
    return enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE), tokenizer

def build_token_map(tok_a, tok_b):
    token_map = {}
    for i in range(tok_a.vocab_size):
        s = tok_a.decode([i]).strip()
        if not s: token_map[i] = 0; continue
        bid = tok_b.encode(s, add_special_tokens=False)
        token_map[i] = bid[0] if bid else 0
    return token_map
