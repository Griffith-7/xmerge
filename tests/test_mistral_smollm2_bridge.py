"""
Full bridge merge: Mistral-7B (mmap streamed) + SmolLM2-360M.
Cross-architecture bridge cached training on 4GB VRAM.
"""
import torch, gc, time, os, math, json, copy, sys
import torch.nn as nn, torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer, MistralRotaryEmbedding, MistralRMSNorm
)
from datasets import load_dataset
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DEVICE = "cuda"
from huggingface_hub import snapshot_download
SNAP_M = snapshot_download("mistralai/Mistral-7B-v0.1")
MODEL_S = "HuggingFaceTB/SmolLM2-360M"

def load_shard(file):
    return torch.load(file, map_location="cpu", mmap=True, weights_only=True)

def clean():
    gc.collect(); torch.cuda.empty_cache()

# ===== TOKEN MAP =====
print("--- Building token map ---")
tok_m = AutoTokenizer.from_pretrained(SNAP_M)
tok_s = AutoTokenizer.from_pretrained(MODEL_S)
tok_m.pad_token = tok_m.eos_token
tok_s.pad_token = tok_s.eos_token

token_map = {}
for i in range(len(tok_m)):
    t = tok_m.decode([i])
    if t and t.strip():
        s = tok_s.encode(t, add_special_tokens=False)
        token_map[i] = s[0] if len(s) == 1 else 0
    else:
        token_map[i] = 0
print(f"Token map built: {len(token_map)} entries, {sum(1 for v in token_map.values() if v > 0)} non-zero")

# ===== CALIBRATION TEXTS =====
print("--- Loading texts ---")
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts = [t["text"] for t in ds if len(t["text"].strip()) > 50][:110]
train_txt = texts[:100]; eval_txt = texts[100:]

print(f"Train: {len(train_txt)}, Eval: {len(eval_txt)}")

# ===== CACHE MISTRAL-7B HIDDEN STATES (STREAMED) =====
print("\n--- Caching Mistral-7B hidden states (mmap streamed) ---")
cfg_m = AutoConfig.from_pretrained(SNAP_M)
cfg_m._attn_implementation = "eager"
cfg_s = AutoConfig.from_pretrained(MODEL_S)
n_layers_m = cfg_m.num_hidden_layers
hidden_m = cfg_m.hidden_size

shard_files = sorted([
    os.path.join(SNAP_M, f) for f in os.listdir(SNAP_M)
    if f.startswith("pytorch_model") and f.endswith(".bin")
])
shards_m = [load_shard(f) for f in shard_files]
layer_shard_m = {}
for si, sd in enumerate(shards_m):
    for k in sd:
        if k.startswith("model.layers."):
            idx = int(k.split(".")[2])
            layer_shard_m[idx] = si

embed_m = norm_m = lm_head_m = None
for sd in shards_m:
    if "model.embed_tokens.weight" in sd: embed_m = sd["model.embed_tokens.weight"]
    if "model.norm.weight" in sd: norm_m = sd["model.norm.weight"]
    if "lm_head.weight" in sd: lm_head_m = sd["lm_head.weight"]

# Encode training texts with Mistral tokenizer
enc = tok_m(train_txt, truncation=True, padding=True, max_length=64, return_tensors="pt")
ids = enc.input_ids
mask = enc.attention_mask
print(f"Encoded: {ids.shape}")

# RoPE
rot = MistralRotaryEmbedding(cfg_m).to(DEVICE)
xd = torch.randn(1, 1, ids.shape[1], cfg_m.head_dim, device=DEVICE)
pos = torch.arange(ids.shape[1], device=DEVICE).unsqueeze(0)
cos, sin = rot(xd, pos)
del rot, xd
# Causal mask
sl = ids.shape[1]
attn_mask = torch.full((sl, sl), float("-inf"), dtype=torch.float32, device=DEVICE)
attn_mask = torch.triu(attn_mask, diagonal=1).unsqueeze(0).unsqueeze(0)
pos_ids = torch.arange(0, sl, device=DEVICE).unsqueeze(0)

# Streamed forward to get pre-ln_f hidden states (all batches per layer to minimize weight transfers)
t0 = time.time()
BATCH_SIZE = 32
embed_w = embed_m.to(DEVICE, dtype=torch.float32)
del embed_m

# Embed all batches upfront
batch_embeds = []
for b_start in range(0, ids.shape[0], BATCH_SIZE):
    b_end = min(b_start + BATCH_SIZE, ids.shape[0])
    batch_embeds.append(F.embedding(ids[b_start:b_end].to(DEVICE), embed_w))
del embed_w

batch_hiddens = [be.clone() for be in batch_embeds]

layer = MistralDecoderLayer(cfg_m, 0).to(DEVICE)
for i in range(n_layers_m):
    si = layer_shard_m.get(i, 0)
    sd = shards_m[si]
    prefix = f"model.layers.{i}."
    layer_sd_ = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    
    with torch.no_grad():
        for name, param in layer.named_parameters():
            if name in layer_sd_:
                param.data = layer_sd_[name].to(DEVICE, dtype=torch.float32, non_blocking=True)
        torch.cuda.synchronize()
        
        for bi in range(len(batch_hiddens)):
            out = layer(batch_hiddens[bi], attention_mask=attn_mask, position_ids=pos_ids,
                        past_key_value=None, output_attentions=False, use_cache=False,
                        position_embeddings=(cos, sin))
            batch_hiddens[bi] = out
            del out
    
    if (i+1) % 8 == 0 or i == 0:
        print(f"  Mistral layer {i+1}/{n_layers_m} done ({time.time()-t0:.0f}s)")

del layer
ha = torch.cat([h.float().cpu() for h in batch_hiddens], dim=0)
del batch_hiddens, batch_embeds; clean()
# Free remaining GPU tensors
del attn_mask, cos, sin, pos_ids; clean()
print(f"  Mistral cached: {ha.shape}, time: {time.time()-t0:.1f}s")

# ===== CACHE SmolLM2 HIDDEN STATES =====
print("\n--- Caching SmolLM2-360M hidden states ---")
t0 = time.time()
print("  Loading SmolLM2 model...")
model_s = AutoModelForCausalLM.from_pretrained(MODEL_S, torch_dtype=torch.bfloat16).eval().to(DEVICE)
for p in model_s.parameters(): p.requires_grad_(False)
print(f"  Model loaded, forward pass...")

# Token-map the Mistral IDs for SmolLM2
ids_s = torch.tensor(
    [[token_map.get(i.item(), 0) for i in row] for row in ids],
    device=DEVICE
)
with torch.no_grad():
    out_s = model_s(input_ids=ids_s, attention_mask=mask.to(DEVICE), output_hidden_states=True)
hb = out_s.hidden_states[-1].float().cpu()
print(f"  Forward done. Saving hb...")
del model_s, out_s; clean()
print(f"  SmolLM2 cached: {hb.shape}, time: {time.time()-t0:.1f}s")

# ===== TRAIN BRIDGE ON GPU =====
print("\n--- Training bridge ---")
from xmerge import utils as xutils
d_a = xutils.hidden_dim(cfg_m)  # 4096
d_b = xutils.hidden_dim(cfg_s)  # 960

# Make a standalone lm_head and final_norm
lm_head = nn.Linear(hidden_m, cfg_m.vocab_size, bias=False).to(DEVICE)
lm_head.weight.data = lm_head_m.to(DEVICE, dtype=torch.float32, copy=True)
del lm_head_m
final_norm = MistralRMSNorm(hidden_m).to(DEVICE)
final_norm.weight.data = norm_m.to(DEVICE, dtype=torch.float32, copy=True)
del norm_m

# OptimalBridge
from xmerge.merge_prod import OptimalBridge
bridge = OptimalBridge(d_a, d_b).to(DEVICE)
opt = torch.optim.AdamW(bridge.parameters(), lr=1e-4, weight_decay=0.02)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)

# Trim to min seq len
k = min(ha.shape[1], hb.shape[1])
ha, hb, ids_t = ha[:, :k].to(DEVICE), hb[:, :k].to(DEVICE), ids[:, :k].to(DEVICE)

best_loss = float("inf")
best_sd = None
for step in range(200):
    opt.zero_grad()
    hf = bridge(ha, hb)
    hf = final_norm(hf)
    logits = lm_head(hf)
    sl = logits[..., :-1, :].contiguous()
    ll = ids_t[..., 1:].contiguous()
    loss = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1), ignore_index=-100)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
    opt.step()
    sched.step()
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_sd = copy.deepcopy(bridge.state_dict())
    print(f"  Step {step+1}: loss={loss.item():.4f}")

bridge.load_state_dict(best_sd if best_sd else bridge.state_dict())
bridge.eval()
print(f"  Best loss: {best_loss:.4f}")

# ===== EVALUATE ON HELD-OUT =====
print("\n--- Evaluation ---")
# Encode eval texts
enc_eval = tok_m(eval_txt, truncation=True, padding=True, max_length=128, return_tensors="pt")
ids_eval = enc_eval.input_ids
mask_eval = enc_eval.attention_mask

# Baseline PPL: pure Mistral-7B (one text only for speed)
enc_b = tok_m(eval_txt[:1], truncation=True, padding=True, max_length=128, return_tensors="pt")
ids_b = enc_b.input_ids

# Re-do streamed forward for eval (one text)
print("  Computing Mistral eval hidden...")
sl = ids_b.shape[1]
attn_mask_eval = torch.full((sl, sl), float("-inf"), dtype=torch.float32, device=DEVICE)
attn_mask_eval = torch.triu(attn_mask_eval, diagonal=1).unsqueeze(0).unsqueeze(0)
pos_ids_eval = torch.arange(0, sl, device=DEVICE).unsqueeze(0)

rot = MistralRotaryEmbedding(cfg_m).to(DEVICE)
xd = torch.randn(1, 1, sl, cfg_m.head_dim, device=DEVICE)
pos = torch.arange(sl, device=DEVICE).unsqueeze(0)
cos_eval, sin_eval = rot(xd, pos)
del rot, xd

# Mistral forward
shard0 = load_shard(shard_files[0])
shard1 = load_shard(shard_files[1])
shards_m_eval = [shard0, shard1]
embed_me = shards_m_eval[0]["model.embed_tokens.weight"] if "model.embed_tokens.weight" in shards_m_eval[0] else shards_m_eval[1]["model.embed_tokens.weight"]

hidden_e = F.embedding(ids_b.to(DEVICE), embed_me.to(DEVICE, dtype=torch.float32))
layer = MistralDecoderLayer(cfg_m, 0).to(DEVICE)
for i in range(n_layers_m):
    si = layer_shard_m.get(i, 0)
    sd = shards_m_eval[si]
    ls = {k[len(f"model.layers.{i}."):]: v for k, v in sd.items() if k.startswith(f"model.layers.{i}.")}
    with torch.no_grad():
        for n, p in layer.named_parameters():
            if n in ls: p.data = ls[n].to(DEVICE, dtype=torch.float32, non_blocking=True)
        torch.cuda.synchronize()
        hidden_e = layer(hidden_e, attention_mask=attn_mask_eval, position_ids=pos_ids_eval,
                         past_key_value=None, output_attentions=False, use_cache=False,
                         position_embeddings=(cos_eval, sin_eval))
    if (i+1) % 8 == 0: print(f"    Eval layer {i+1}/{n_layers_m}")
del layer

# Mistral PPL (apply final norm then lm_head)
with torch.no_grad():
    hidden_en = final_norm(hidden_e)
    loge = lm_head(hidden_en)
    sl = loge[..., :-1, :].contiguous()
    ll = ids_b[..., 1:].to(DEVICE).contiguous()
    mistral_ppl = math.exp(F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1), ignore_index=-100).item())
print(f"  Mistral baseline PPL: {mistral_ppl:.1f}")

# Free GPU memory before loading SmolLM2 for eval
hidden_e = hidden_e.cpu()
del hidden_en, loge; clean()

# Bridge PPL - use same text as Mistral eval
print("  Computing bridge eval...")
ids_es = torch.tensor([[token_map.get(i.item(), 0) for i in row] for row in ids_b], device=DEVICE)
with torch.no_grad():
    model_s = AutoModelForCausalLM.from_pretrained(MODEL_S, torch_dtype=torch.bfloat16).eval().to(DEVICE)
    for p in model_s.parameters(): p.requires_grad_(False)
    out_se = model_s(input_ids=ids_es, attention_mask=mask_eval[:1].to(DEVICE), output_hidden_states=True)
    hb_eval = out_se.hidden_states[-1].float()
    del model_s, out_se; clean()
    
    hb_eval = hb_eval.cpu()
    
    # Bridge
    k_eval = min(hidden_e.shape[1], hb_eval.shape[1])
    hf_eval = bridge(hidden_e[:, :k_eval].to(DEVICE), hb_eval[:, :k_eval].to(DEVICE))
    hf_eval = final_norm(hf_eval)
    loge_b = lm_head(hf_eval)
    sl = loge_b[..., :-1, :].contiguous()
    ll = ids_b[..., 1:].to(DEVICE).contiguous()
    bridge_ppl = math.exp(F.cross_entropy(sl[:, :k_eval-1].reshape(-1, sl.size(-1)), ll[:, :k_eval-1].reshape(-1), ignore_index=-100).item())
print(f"  Bridge PPL: {bridge_ppl:.1f}")

print(f"\n{'='*50}")
print(f"  Mistral-7B baseline: {mistral_ppl:.1f}")
print(f"  Mistral+SmolLM2 bridge: {bridge_ppl:.1f}")
print(f"{'='*50}")
print("DONE!")
