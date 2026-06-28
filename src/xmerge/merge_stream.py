"""
Layer-by-layer streaming for xmerge -- enables merging 7B models on 4GB VRAM.

Strategy for low RAM:
  1. Models loaded via accelerate with CPU offload (weights stay on disk until needed)
  2. Forward passes process 1 layer at a time on GPU, rest on disk
  3. For bridge training: cache hidden states from ONE model at a time, then train bridge purely on GPU

Approach 1 -- Weight blending:    Streamed CKA + CPU weight merge
Approach 2 -- Bridge v2:          Streamed forward each step
Approach 3 -- Bridge cached:      Stream cache once per model, train bridge on GPU
Approach 4 -- Full pipeline:      Cache + train + save + eval
"""

import torch, gc, copy, math, os, json, numpy as np
import torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from . import merge_prod, utils

__all__ = [
    "clean", "StreamedForward", "load_model_streamed", "load_one_model_at_a_time",
    "activation_similarity", "compute_cka_streamed",
    "streamed_merge_same_arch", "streamed_train_bridge_v2",
    "streamed_train_bridge_cached", "streamed_merge_diff_arch",
    "StreamedGenerator",
]

def clean():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- ARCHITECTURE HELPERS -------------------------------------------------

def _get_embedding_modules(model):
    mt = getattr(model.config, 'model_type', '').lower()
    if mt == 'gpt2':
        return {'wte': model.transformer.wte, 'wpe': model.transformer.wpe}
    if mt in ('llama', 'mistral', 'qwen2', 'phi', 'phi3', 'smollm2', 'gemma', 'zephyr'):
        return {'embed_tokens': model.model.embed_tokens}
    if mt == 'opt':
        d = {'embed_tokens': model.model.decoder.embed_tokens}
        if hasattr(model.model.decoder, 'embed_positions'):
            d['embed_positions'] = model.model.decoder.embed_positions
        return d
    for name, mod in model.named_modules():
        if 'wte' in name or 'embed_tokens' in name:
            if isinstance(mod, nn.Embedding):
                return {name: mod}
    return {}

def _get_final_norm(model):
    return merge_prod._get_final_norm(model)


# --- STREAMED FORWARD PASS -----------------------------------------------

class StreamedForward:
    """
    Forward pass with layer-by-layer GPU streaming.
    Only 1 layer on GPU at a time. Supports hooks for CKA.
    Models should be loaded via accelerate CPU offload or regular CPU.

    Usage:
        stream = StreamedForward(model, device="cuda")
        hidden = stream(input_ids)                    # last hidden state
        hidden, all_h = stream(input_ids, output_hidden_states=True)
    """

    def __init__(self, model, device="cuda", hooks=None):
        self.model = model
        self.device = device
        self.model.config.use_cache = False
        self.layers = merge_prod._get_layer_list(model)
        self.n_layers = len(self.layers)
        self.final_norm = _get_final_norm(model)
        self.embeddings = _get_embedding_modules(model)
        self.model_type = getattr(model.config, 'model_type', '').lower()
        self.hooks = hooks or {}

    def _apply_hooks(self, i, hidden):
        if i in self.hooks:
            self.hooks[i](hidden)

    def _compute_embeddings(self, input_ids):
        device = self.device
        if 'wte' in self.embeddings and 'wpe' in self.embeddings:
            wte = self.embeddings['wte'].to(device)
            wpe = self.embeddings['wpe'].to(device)
            pos = torch.arange(0, input_ids.shape[1], device=device).unsqueeze(0)
            hidden = wte(input_ids.to(device)) + wpe(pos)
            self.embeddings['wte'].to("cpu"); self.embeddings['wpe'].to("cpu")
        elif 'embed_tokens' in self.embeddings:
            et = self.embeddings['embed_tokens'].to(device)
            hidden = et(input_ids.to(device))
            self.embeddings['embed_tokens'].to("cpu")
            if 'embed_positions' in self.embeddings:
                ep = self.embeddings['embed_positions'].to(device)
                pos = torch.arange(0, input_ids.shape[1], device=device).unsqueeze(0)
                hidden = hidden + ep(pos)
                self.embeddings['embed_positions'].to("cpu")
        else:
            et = list(self.embeddings.values())[0].to(device)
            hidden = et(input_ids.to(device))
            list(self.embeddings.values())[0].to("cpu")
        clean()
        return hidden

    @torch.no_grad()
    def __call__(self, input_ids, output_hidden_states=False):
        hidden = self._compute_embeddings(input_ids)
        self._apply_hooks(-1, hidden.cpu())

        if output_hidden_states:
            all_hidden = [hidden.cpu()]

        position_embeddings = None
        if self.model_type in ('llama', 'mistral', 'qwen2', 'gemma'):
            model_obj = self.model.model
            if hasattr(model_obj, 'rotary_emb'):
                seq_len = hidden.shape[1]
                position_ids = torch.arange(0, seq_len, device=self.device).unsqueeze(0)
                cos, sin = model_obj.rotary_emb(hidden.to(self.device), position_ids)
                position_embeddings = (cos.to(hidden.dtype), sin.to(hidden.dtype))

        for i in range(self.n_layers):
            layer = self.layers[i].to(self.device)
            hidden = hidden.to(self.device)
            if position_embeddings is not None:
                hidden_out = layer(hidden, position_embeddings=position_embeddings)
            else:
                hidden_out = layer(hidden)
            hidden = hidden_out[0] if isinstance(hidden_out, tuple) else hidden_out
            self.layers[i] = self.layers[i].to("cpu")
            hidden_cpu = hidden.cpu()
            self._apply_hooks(i, hidden_cpu)

            if output_hidden_states:
                all_hidden.append(hidden_cpu.clone())

            del layer, hidden_out
            clean()

        hidden = hidden_cpu
        if self.final_norm is not None:
            self.final_norm = self.final_norm.to("cpu")
            hidden = self.final_norm(hidden)

        if output_hidden_states:
            return hidden, all_hidden
        return hidden


# --- MODEL LOADING (memory-efficient) ------------------------------------

def load_model_streamed(model_name, dtype=torch.float16, cache_dir=None,
                         use_offload=True, offload_folder="offload"):
    """
    Load a model with CPU offloading for memory efficiency.
    When use_offload=True, uses accelerate's device_map="cpu" so
    weights are accessed from disk on demand.
    """
    from transformers import AutoConfig

    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"  Loading {model_name} (dtype={dtype})...")

    if use_offload:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="cpu",       # force all params to CPU, no GPU used
            offload_folder=offload_folder,
            offload_state_dict=True, # offload to disk
            cache_dir=cache_dir,
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            cache_dir=cache_dir,
        ).eval()
        for p in model.parameters():
            p.data = p.data.cpu()

    clean()
    return model, tok


def load_one_model_at_a_time(model_name, dtype=torch.float16, cache_dir=None):
    """
    Load a model, yield it, then unload. For processing models sequentially.
    Usage:
        with load_one_model_at_a_time("model_a") as (ma, tok_a):
            ...
        # model A is now deleted
        with load_one_model_at_a_time("model_b") as (mb, tok_b):
            ...
    """
    import contextlib

    @contextlib.contextmanager
    def loader():
        print(f"\n  Loading {model_name} into memory...")
        model, tok = load_model_streamed(model_name, dtype, cache_dir, use_offload=False)
        try:
            yield model, tok
        finally:
            print(f"  Unloading {model_name}...")
            del model
            clean()

    return loader()


# ===========================================================================
# APPROACH 1 -- Weight blending (streamed CKA)
# ===========================================================================

def activation_similarity(h_a, h_b):
    """CKA between two hidden states from different models.
    h_a, h_b: [batch, seq, d_a], [batch, seq, d_b] or [*, d].
    Reshapes to 2D and computes HSIC CKA.
    """
    ha_2d = h_a.reshape(-1, h_a.shape[-1])
    hb_2d = h_b.reshape(-1, h_b.shape[-1])
    min_n = min(ha_2d.shape[0], hb_2d.shape[0])
    return merge_prod.hsic_cka(ha_2d[:min_n], hb_2d[:min_n])

def compute_cka_streamed(model_a, model_b, input_ids, device="cuda"):
    """Compute CKA similarity matrix between two models via streaming."""
    n_a = len(merge_prod._get_layer_list(model_a))
    n_b = len(merge_prod._get_layer_list(model_b))
    hiddens_a = {}
    hiddens_b = {}

    def make_hook(store):
        def hook(idx):
            def fn(h):
                store[idx] = h.float().cpu()
            return fn
        return hook

    hooks_a = {i: make_hook(hiddens_a)(i) for i in range(n_a)}
    hooks_a[-1] = make_hook(hiddens_a)(-1)
    hooks_b = {i: make_hook(hiddens_b)(i) for i in range(n_b)}
    hooks_b[-1] = make_hook(hiddens_b)(-1)

    sa = StreamedForward(model_a, device, hooks=hooks_a)
    sb = StreamedForward(model_b, device, hooks=hooks_b)

    print("    Streaming forward through model A...")
    sa(input_ids)
    print("    Streaming forward through model B...")
    sb(input_ids)

    sim = {}
    for i_a in range(n_a):
        for i_b in range(n_b):
            ha = hiddens_a.get(i_a)
            hb = hiddens_b.get(i_b)
            if ha is None or hb is None:
                sim[(i_a, i_b)] = 0.0
            else:
                sim[(i_a, i_b)] = activation_similarity(ha, hb).item()

    return sim, hiddens_a, hiddens_b


def streamed_merge_improved(model_a, model_b, calib_texts=None,
                            save_name=None, device="cuda", use_procrustes=False):
    """Improved weight blending with interpolated layer mapping + alpha search."""
    print("="*60)
    print("  IMPROVED Weight Blending (interpolated layers + alpha search)")
    print("="*60)

    tok = AutoTokenizer.from_pretrained(
        model_a.config._name_or_path or "distilgpt2"
    )
    tok.pad_token = tok.eos_token

    if calib_texts is None:
        calib_texts = merge_prod.load_texts(8)

    enc = tok(calib_texts[:4], truncation=True, padding=True, max_length=64, return_tensors="pt")
    ids = enc.input_ids

    n_a = merge_prod._get_n_layers(model_a.config)
    n_b = merge_prod._get_n_layers(model_b.config)
    print(f"  Layers: A={n_a}, B={n_b}")

    # Step 1: Compute CKA similarity matrix for layer mapping
    print("  Computing CKA similarity...")
    sim, ha, hb = compute_cka_streamed(model_a, model_b, ids, device)
    avg_sim = np.mean(list(sim.values())) if sim else 0.0

    # Step 2: Interpolated layer mapping
    # If n_a == n_b: direct 1-to-1
    # If n_a != n_b: interpolate B's layers to match A's count
    print("  Building interpolated layer mapping...")
    prefix_a = merge_prod._get_layer_prefix(model_a)
    prefix_b = merge_prod._get_layer_prefix(model_b)
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()

    layers_a = {}
    for k, v in sd_a.items():
        if k.startswith(prefix_a):
            parts = k[len(prefix_a):].split(".")
            idx, local = int(parts[0]), ".".join(parts[1:])
            layers_a.setdefault(idx, {})[local] = v

    layers_b = {}
    for k, v in sd_b.items():
        if k.startswith(prefix_b):
            parts = k[len(prefix_b):].split(".")
            idx, local = int(parts[0]), ".".join(parts[1:])
            layers_b.setdefault(idx, {})[local] = v

    # For layer mapping, find closest B layer for each A layer using CKA
    mapping = {}
    for i_a in range(n_a):
        # Find the B layer with highest CKA to this A layer
        best_b = max(range(n_b), key=lambda ib: sim.get((i_a, ib), 0))
        mapping[i_a] = best_b

    print(f"  CKA-based mapping done (avg sim: {avg_sim:.3f})")

    # Step 3: Project B weights to A's dimensions via SVD for each layer
    def interpolate_weights(w_b, target_shape):
        """Interpolate B weights to match target shape via SVD or resize."""
        if w_b.shape == target_shape:
            return w_b
        if w_b.dim() == 1:
            # 1D: interpolate (e.g., for norms)
            return F.interpolate(
                w_b.view(1, 1, -1), size=target_shape[0], mode='linear'
            ).view(-1)
        if w_b.dim() == 2:
            return utils.svd_project(w_b, target_shape[0], target_shape[1])
        return w_b

    b_proj = {}
    skipped = []
    for i_a, i_b in mapping.items():
        for local, w_a in layers_a[i_a].items():
            if i_b not in layers_b:
                continue
            w_b = layers_b[i_b].get(local)
            if w_b is None:
                continue
            try:
                w_b_proj = interpolate_weights(w_b.float(), w_a.shape)
                b_proj[(i_a, local)] = w_b_proj
            except Exception as e:
                skipped.append((i_a, local, str(e)))

    # Also handle non-layer shared keys
    for k in sd_a:
        if not k.startswith(prefix_a) and k in sd_b:
            if sd_a[k].shape == sd_b[k].shape:
                b_proj[k] = sd_b[k].float()

    if skipped:
        print(f"  Skipped {len(skipped)} incompatible weights")

    # Optionally apply Procrustes alignment
    if use_procrustes and 'ha' in dir() and 'hb' in dir():
        print("  Applying orthogonal Procrustes alignment...")
        try:
            b_proj = merge_prod._apply_procrustes_alignment(
                model_a, model_b, mapping, b_proj, ha, hb
            )
        except Exception as e:
            print(f"  Procrustes skipped: {e}")

    # Step 4: Optimize per-layer alpha
    # Try a few alpha values, pick the one that gives lowest loss on calibration
    print("  Optimizing per-layer alphas...")
    alpha_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # Use avg CKA-based alpha as starting point, then refine
    alpha_default = {i: max(0.2, min(0.9, 0.7 - 0.5 * np.mean([sim.get((i, j), 0) for j in range(n_b)]))) for i in range(n_a)}

    # Quick grid search: try uniform alphas, pick best
    best_alpha_val = 0.5
    best_loss = float("inf")
    for a_test in [0.3, 0.5, 0.7]:
        merged_sd = _blend_weights(sd_a, b_proj, prefix_a, layers_a, {i: a_test for i in range(n_a)}, n_a)
        try:
            merged_test = AutoModelForCausalLM.from_config(model_a.config)
            merged_test.load_state_dict(merged_sd, strict=False)
            sf = StreamedForward(merged_test, device)
            h = sf(ids)
            lm = merge_prod._get_lm_head(merged_test)
            if lm is not None:
                logits = lm(h.to(dtype=next(merged_test.parameters()).dtype))
                sl, ll = logits[..., :-1, :].contiguous(), ids[..., 1:].contiguous()
                loss = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1), ignore_index=-100)
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_alpha_val = a_test
        except Exception:
            pass
        del merged_test
        clean()

    alphas = {i: best_alpha_val for i in range(n_a)}
    print(f"  Best uniform alpha: {best_alpha_val:.2f}")

    # Step 5: Final blend with best alpha
    print("  Final blending...")
    merged_sd = _blend_weights(sd_a, b_proj, prefix_a, layers_a, alphas, n_a)

    # Create merged model
    merged = AutoModelForCausalLM.from_config(model_a.config)
    merged.load_state_dict(merged_sd, strict=False)

    # Compute PPL
    print("  Computing merged model PPL (streamed)...")
    stream = StreamedForward(merged, device)
    hidden = stream(ids)
    lm_head = merge_prod._get_lm_head(merged)
    if lm_head is not None:
        logits = lm_head(hidden.to(dtype=next(merged.parameters()).dtype))
        sl, ll = logits[..., :-1, :].contiguous(), ids[..., 1:].contiguous()
        loss = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1), ignore_index=-100)
        ppl_val = math.exp(loss.item())
    else:
        ppl_val = float("inf")
    print(f"  Merged model PPL: {ppl_val:.1f}")

    if save_name:
        save_dir = os.path.join(merge_prod.SAVE_DIR, save_name)
        os.makedirs(save_dir, exist_ok=True)
        merged.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)

    clean()
    return merged, tok, ppl_val, alphas

streamed_merge_same_arch = streamed_merge_improved


def _blend_weights(sd_a, b_proj, prefix_a, layers_a, alphas, n_a):
    """Helper: blend A weights with projected B weights."""
    merged_sd = {}
    for k, v in sd_a.items():
        if merge_prod._is_embed_or_output_key(k):
            merged_sd[k] = v.clone()
        elif k in b_proj and not k.startswith(prefix_a):
            avg_a = sum(alphas.values()) / max(len(alphas), 1)
            merged_sd[k] = (avg_a * v.float() + (1 - avg_a) * b_proj[k]).to(v.dtype)
        else:
            merged_sd[k] = v.clone()

    for k in sd_a:
        if k.startswith(prefix_a):
            rest = k[len(prefix_a):]
            parts = rest.split(".")
            i_a = int(parts[0])
            local = ".".join(parts[1:])
            a = alphas.get(i_a, 0.5)
            key = (i_a, local)
            if key in b_proj:
                try:
                    merged_sd[k] = (a * sd_a[k].float() + (1 - a) * b_proj[key]).to(sd_a[k].dtype)
                except Exception:
                    merged_sd[k] = sd_a[k].clone()
    return merged_sd


# ===========================================================================
# APPROACH 2 -- Bridge v2 (streamed forward each step)
# ===========================================================================

def _copy_lm_head(model, device):
    """Extract lm_head as an independent copy (avoids tied-weight issues)."""
    lh = merge_prod._get_lm_head(model)
    if lh is None:
        return None
    has_bias = lh.bias is not None
    dtype = lh.weight.dtype
    copy_lh = nn.Linear(lh.in_features, lh.out_features, bias=has_bias, dtype=dtype)
    copy_lh.load_state_dict(lh.state_dict())
    return copy_lh.to(device)


def _holdout_split(texts, eval_pct=0.2):
    n = len(texts)
    n_eval = max(min(int(n * eval_pct), n // 2), min(4, n // 2))
    if n_eval < 1:
        return texts, []
    return texts[:-n_eval], texts[-n_eval:]


def _streamed_eval_ppl(model_a, model_b, bridge, tok, eval_texts, device, max_len=128, batch_size=2):
    if not eval_texts:
        return float("inf")
    lm_head = _copy_lm_head(model_a, device)
    if lm_head is None:
        return float("inf")
    model_dtype = next(model_a.parameters()).dtype
    enc = tok(eval_texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    ids = enc.input_ids
    n = ids.shape[0]
    total_loss = 0.0
    with torch.no_grad():
        for b in range(0, n, batch_size):
            batch_ids = ids[b:b+batch_size]
            ha = StreamedForward(model_a, device)(batch_ids)
            hb = StreamedForward(model_b, device)(batch_ids)
            k = min(ha.shape[1], hb.shape[1])
            ha_gpu = ha[:, :k].float().to(device)
            hb_gpu = hb[:, :k].float().to(device)
            ids_gpu = batch_ids[:, :k].to(device)
            hf = bridge(ha_gpu, hb_gpu)
            logits = lm_head(hf.to(model_dtype))
            sl, ll = logits[..., :-1, :].contiguous(), ids_gpu[..., 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1), ignore_index=-100)
            total_loss += loss.item()
            del ha, hb, ha_gpu, hb_gpu, ids_gpu, hf, logits, sl, ll
            clean()
    lm_head.to("cpu")
    avg = total_loss / max(1, (n + batch_size - 1) // batch_size)
    return math.exp(avg)


def streamed_train_bridge_v2(model_a, model_b, tok, texts, device="cuda",
                              steps=10, lr=3e-4, weight_decay=0.01, max_len=128,
                              batch_size=2, eval_texts=None, bridge_type="linear"):
    """Approach 2: Train bridge with streamed forward passes each gradient step.

    Reports held-out PPL from eval_texts when provided.
    """
    if eval_texts is not None:
        train_texts = texts
    else:
        train_texts, eval_texts = texts, None
    if not train_texts:
        train_texts = texts

    print("="*60)
    print("  APPROACH 2: Bridge v2 (streamed forward each step)")
    print("="*60)

    d_a = utils.hidden_dim(model_a.config)
    d_b = utils.hidden_dim(model_b.config)
    if bridge_type == "mlp":
        bridge = merge_prod.MLPBridge(d_a, d_b).to(device)
    else:
        bridge = merge_prod.OptimalBridge(d_a, d_b).to(device)
    lm_head = _copy_lm_head(model_a, device)
    model_dtype = next(model_a.parameters()).dtype

    enc = tok(train_texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    ids, mask = enc.input_ids, enc.attention_mask

    opt = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    best = None
    best_loss = float("inf")
    n = ids.shape[0]

    torch.set_grad_enabled(True)
    bridge.train()

    for s in range(steps):
        opt.zero_grad()
        total_loss = 0.0

        for b in range(0, n, batch_size):
            batch_ids = ids[b:b+batch_size]
            ha = StreamedForward(model_a, device)(batch_ids)
            hb = StreamedForward(model_b, device)(batch_ids)

            k = min(ha.shape[1], hb.shape[1])
            ha_gpu = ha[:, :k].float().to(device)
            hb_gpu = hb[:, :k].float().to(device)
            ids_gpu = batch_ids[:, :k].to(device)

            hf = bridge(ha_gpu, hb_gpu)
            logits = lm_head(hf.to(model_dtype))
            sl, ll = logits[..., :-1, :].contiguous(), ids_gpu[..., 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1), ignore_index=-100)
            loss.backward()
            total_loss += loss.item()

            del ha, hb, ha_gpu, hb_gpu, ids_gpu, hf, logits, sl, ll
            clean()

        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        opt.step()
        sched.step()

        avg_loss = total_loss / max(1, (n + batch_size - 1) // batch_size)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best = copy.deepcopy(bridge.state_dict())

        print(f"    Step {s+1}/{steps}  loss={avg_loss:.4f}")

    bridge.load_state_dict(best if best else bridge.state_dict())
    bridge.eval()
    torch.set_grad_enabled(False)

    eval_ppl = _streamed_eval_ppl(model_a, model_b, bridge, tok, eval_texts, device, max_len, batch_size)
    if math.isfinite(eval_ppl):
        print(f"    Eval PPL (held-out): {eval_ppl:.1f}")
    else:
        print(f"    (no held-out eval texts)")
    bridge.eval_ppl = eval_ppl

    lm_head.to("cpu")
    clean()
    return bridge, eval_ppl


# ===========================================================================
# APPROACH 3 -- Bridge cached (streamed cache once, train bridge on GPU)
# ===========================================================================

def streamed_train_bridge_cached(model_a, model_b, tok, texts, device="cuda",
                                  steps=20, lr=3e-4, weight_decay=0.01,
                                  max_len=128, batch_size=2, eval_texts=None,
                                  bridge_type="linear"):
    """Approach 3: Cache hidden states via streaming ONCE, train bridge purely on GPU.

    Reports held-out PPL from eval_texts when provided.
    """
    if eval_texts is not None:
        train_texts = texts
    else:
        train_texts, eval_texts = texts, None
    if not train_texts:
        train_texts = texts

    print("="*60)
    print("  APPROACH 3: Bridge Cached (stream cache once, GPU train)")
    print("="*60)

    d_a = utils.hidden_dim(model_a.config)
    d_b = utils.hidden_dim(model_b.config)
    if bridge_type == "mlp":
        bridge = merge_prod.MLPBridge(d_a, d_b).to(device)
    else:
        bridge = merge_prod.OptimalBridge(d_a, d_b).to(device)

    print("  Caching hidden states (stream both models)...")
    enc = tok(train_texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    ids = enc.input_ids
    n = ids.shape[0]

    all_ha, all_hb, all_ids = [], [], []
    for b in range(0, n, batch_size):
        batch_ids = ids[b:b+batch_size]

        print(f"    Forward model A batch {b//batch_size + 1}...")
        ha = StreamedForward(model_a, device)(batch_ids)
        print(f"    Forward model B batch {b//batch_size + 1}...")
        hb = StreamedForward(model_b, device)(batch_ids)

        k = min(ha.shape[1], hb.shape[1])
        all_ha.append(ha[:, :k].float())
        all_hb.append(hb[:, :k].float())
        all_ids.append(batch_ids[:, :k])

    ha_full = torch.cat(all_ha, dim=0)
    hb_full = torch.cat(all_hb, dim=0)
    ids_full = torch.cat(all_ids, dim=0)
    print(f"  Cached: ha={list(ha_full.shape)}, hb={list(hb_full.shape)}")

    # Move to GPU for training
    ha_gpu = ha_full.to(device)
    hb_gpu = hb_full.to(device)
    ids_gpu = ids_full.to(device)
    lm_head = _copy_lm_head(model_a, device)
    model_dtype = next(model_a.parameters()).dtype

    opt = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    best = None; best_loss = float("inf")

    torch.set_grad_enabled(True)
    bridge.train()

    for s in range(steps):
        opt.zero_grad()
        hf = bridge(ha_gpu, hb_gpu)
        logits = lm_head(hf.to(model_dtype))
        sl, ll = logits[..., :-1, :].contiguous(), ids_gpu[..., 1:].contiguous()
        loss = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        opt.step()
        sched.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best = copy.deepcopy(bridge.state_dict())

        if (s+1) % 5 == 0:
            print(f"    Step {s+1}/{steps}  loss={loss.item():.4f}")

    bridge.load_state_dict(best if best else bridge.state_dict())
    bridge.eval()
    torch.set_grad_enabled(False)

    eval_ppl = _streamed_eval_ppl(model_a, model_b, bridge, tok, eval_texts, device, max_len, batch_size)
    if math.isfinite(eval_ppl):
        print(f"    Eval PPL (held-out): {eval_ppl:.1f}")
    else:
        print(f"    (no held-out eval texts)")
    bridge.eval_ppl = eval_ppl

    del ha_gpu, hb_gpu, ids_gpu, lm_head, ha_full, hb_full, ids_full
    clean()
    return bridge, eval_ppl


# ===========================================================================
# APPROACH 4 -- Full pipeline
# ===========================================================================

def streamed_merge_diff_arch(model_a, model_b, calib_texts=None,
                              save_name="streamed_merge", device="cuda",
                              tok=None, steps=10, lr=3e-4,
                              weight_decay=0.01, max_len=128,
                              bridge_type="linear"):
    """Approach 4: Full pipeline -- cache, train, save, eval."""
    print("="*60)
    print("  APPROACH 4: Full Pipeline (cached + save + eval)")
    print("="*60)

    if tok is None:
        tok = AutoTokenizer.from_pretrained(
            model_a.config._name_or_path or "distilgpt2"
        )
        tok.pad_token = tok.eos_token

    if calib_texts is None:
        calib_texts = merge_prod.load_texts(16)

    # Baseline PPL for model A
    enc = tok(calib_texts[:4], truncation=True, padding=True, max_length=64, return_tensors="pt")
    ids = enc.input_ids
    print("  Computing model A baseline PPL...")
    ha = StreamedForward(model_a, device)(ids)
    lm_head_copy = _copy_lm_head(model_a, device)
    if lm_head_copy is not None:
        logits = lm_head_copy(ha.to(dtype=next(model_a.parameters()).dtype).to(device))
        sl, ll = logits[..., :-1, :].contiguous(), ids[..., 1:].to(device).contiguous()
        loss_a = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1), ignore_index=-100)
        ppl_a = math.exp(loss_a.item())
        print(f"  Model A PPL: {ppl_a:.1f}")
    else:
        ppl_a = float("inf")

    # Train bridge (cached streaming)
    bridge, ppl_val = streamed_train_bridge_cached(
        model_a, model_b, tok, calib_texts, device=device,
        steps=steps, lr=lr, weight_decay=weight_decay, max_len=max_len,
        bridge_type=bridge_type,
    )

    # Save
    if save_name:
        save_dir = os.path.join(merge_prod.SAVE_DIR, save_name)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(bridge.state_dict(), os.path.join(save_dir, "bridge.pt"))
        tok.save_pretrained(save_dir)
        with open(os.path.join(save_dir, "bridge_config.json"), "w") as f:
            json.dump({
                "d_a": utils.hidden_dim(model_a.config),
                "d_b": utils.hidden_dim(model_b.config),
                "model_a": model_a.config._name_or_path or "unknown",
                "model_b": model_b.config._name_or_path or "unknown",
                "ppl_a": round(ppl_a, 1),
                "ppl_bridge": round(ppl_val, 1),
                "type": "diff_arch_streamed_bridge",
                "steps": steps, "lr": lr,
            }, f, indent=2)
        print(f"  Saved to {save_dir}/")

    clean()
    return bridge, ppl_val


# ===========================================================================
# GENERATION
# ===========================================================================

class StreamedGenerator:
    """Generate text via streamed forward passes."""

    def __init__(self, model_a, model_b, bridge, tokenizer, device="cuda",
                 mix_alpha=0.3, temp=0.9):
        self.model_a = model_a
        self.model_b = model_b
        self.bridge = bridge.to(device).eval()
        self.tok = tokenizer
        self.device = device
        self.dtype = next(model_a.parameters()).dtype
        self.lm_head = _copy_lm_head(model_a, device)
        self.mix_alpha = mix_alpha
        self.temp = temp

    @torch.no_grad()
    def generate(self, prompt, max_new=20, method="bridge"):
        ids = self.tok(prompt, return_tensors="pt").input_ids
        for _ in range(max_new):
            ha = StreamedForward(self.model_a, self.device)(ids)
            hb = StreamedForward(self.model_b, self.device)(ids)
            k = min(ha.shape[1], hb.shape[1])
            ha_gpu = ha[:, :k].float().to(self.device)
            hb_gpu = hb[:, :k].float().to(self.device)

            if method == "stitch":
                hf = self.bridge(ha_gpu, hb_gpu)
            else:
                h_bridge = self.bridge(ha_gpu, hb_gpu)
                hf = ha_gpu + self.mix_alpha * (h_bridge - ha_gpu)

            logits = self.lm_head(hf.to(self.dtype))[:, -1, :] / self.temp
            next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1).cpu()
            ids = torch.cat([ids, next_tok], dim=-1)
            del ha, hb, ha_gpu, hb_gpu, hf, logits
            clean()
        return self.tok.decode(ids[0], skip_special_tokens=True)
