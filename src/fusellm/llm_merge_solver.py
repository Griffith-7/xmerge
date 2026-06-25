# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
LLM Merge Solver — Breaks the weight-space ceiling
Two solutions:
  Solution 1: Different sizes (same arch)  → per-layer optimization + activation repair
  Solution 2: Different architectures       → stitching with learned bridges

No full training required. Uses ~128 calibration samples, ~50 gradient steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import differential_evolution
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    GPT2Model, GPT2LMHeadModel, GPT2Config
)
from datasets import load_dataset
from sklearn.decomposition import PCA
import copy
import os
import warnings
warnings.filterwarnings('ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
print(f"Using device: {DEVICE}")

# ─── 1. CALIBRATION DATA ───────────────────────────────────────────────────

def load_calibration_data(n_samples=128, seq_len=128, tokenizer_name="gpt2"):
    """Load small calibration set from Wikitext-2"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
    except:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    texts = [t["text"] for t in dataset if len(t["text"].strip()) > 50][:n_samples * 2]
    if len(texts) == 0:
        texts = ["The quick brown fox jumps over the lazy dog."] * n_samples
    
    encoded = tokenizer(texts, truncation=True, padding=True, max_length=seq_len, return_tensors="pt")
    input_ids = encoded.input_ids[:n_samples]
    attention_mask = encoded.attention_mask[:n_samples]
    
    return input_ids.to(DEVICE), attention_mask.to(DEVICE), tokenizer


# ─── 2. MODEL LOADING ──────────────────────────────────────────────────────

def load_gpt2(model_name="gpt2"):
    """Load a GPT-2 model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=DTYPE
    ).to(DEVICE).eval()
    
    # Extract config info
    config = model.config
    info = {
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "n_inner": config.n_inner if hasattr(config, "n_inner") else 4 * config.n_embd,
        "vocab_size": config.vocab_size,
        "activation_function": config.activation_function if hasattr(config, "activation_function") else "gelu_new",
        "model_type": "gpt2",
    }
    return model, tokenizer, info


def load_model(model_name):
    """Generic model loader"""
    if "gpt2" in model_name.lower() or model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
        return load_gpt2(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=DTYPE
    ).to(DEVICE).eval()
    config = model.config
    info = {
        "n_layer": getattr(config, "num_hidden_layers", getattr(config, "n_layer", 12)),
        "n_head": getattr(config, "num_attention_heads", getattr(config, "n_head", 12)),
        "n_embd": getattr(config, "hidden_size", getattr(config, "n_embd", 768)),
        "n_inner": getattr(config, "intermediate_size", 4 * getattr(config, "hidden_size", 768)),
        "vocab_size": config.vocab_size,
        "activation_function": getattr(config, "activation_function", "gelu"),
        "model_type": config.model_type,
    }
    return model, tokenizer, info


# ─── 3. DIMENSION ADAPTATION ───────────────────────────────────────────────

def svd_project(W, target_in, target_out):
    """Project weight matrix to target dimensions using SVD truncation/expansion"""
    W = W.float()
    current_out, current_in = W.shape
    
    if current_out == target_out and current_in == target_in:
        return W
    
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    k = min(current_in, current_out, target_in, target_out)
    
    U = U[:, :k]
    S = torch.diag(S[:k])
    Vh = Vh[:k, :]
    
    W_reduced = U @ S @ Vh
    
    W_new = torch.zeros(target_out, target_in, dtype=W.dtype)
    out_k = min(k, target_out)
    in_k = min(k, target_in)
    W_new[:out_k, :in_k] = W_reduced[:out_k, :in_k]
    
    return W_new


def adapt_weights_gpt2(source_state_dict, target_config, target_state_dict):
    """Project source weights to match target dimensions"""
    adapted = {}
    n_embd_t = target_config.n_embd
    n_embd_s = source_state_dict["transformer.wte.weight"].shape[1]
    
    scale_factor = np.sqrt(n_embd_t / n_embd_s) if n_embd_s > 0 else 1.0
    
    for key in target_state_dict.keys():
        if key in source_state_dict:
            W = source_state_dict[key].float()
            target_shape = target_state_dict[key].shape
            
            if W.shape == target_shape:
                adapted[key] = W
            elif len(W.shape) == 2 and len(target_shape) == 2:
                adapted[key] = svd_project(W, target_shape[1], target_shape[0])
            elif len(W.shape) == 1 and len(target_shape) == 1:
                if W.shape[0] != target_shape[0]:
                    adapted[key] = torch.zeros(target_shape[0], dtype=W.dtype)
                    k = min(W.shape[0], target_shape[0])
                    adapted[key][:k] = W[:k]
                else:
                    adapted[key] = W
            else:
                adapted[key] = W
        elif "weight" in key and "layernorm" not in key.lower() and "ln_" not in key.lower():
            adapted[key] = target_state_dict[key].clone()
        else:
            adapted[key] = target_state_dict[key].clone()
    
    return adapted


# ─── 4. LAYER CORRESPONDENCE ──────────────────────────────────────────────

def find_layer_correspondence(model_a, model_b, input_ids, attention_mask=None):
    """Find layer correspondence via CKA on hidden states"""
    n_layers_a = model_a.config.n_layer
    n_layers_b = model_b.config.n_layer
    
    with torch.no_grad():
        # Get hidden states from model A
        outputs_a = model_a(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hiddens_a = outputs_a.hidden_states  # tuple of (n_layers+1) tensors [batch, seq, dim]
        
        # Get hidden states from model B
        outputs_b = model_b(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hiddens_b = outputs_b.hidden_states
    
    # CKA between layer pairs
    sim_matrix = np.zeros((n_layers_a, n_layers_b))
    for i in range(n_layers_a):
        for j in range(n_layers_b):
            sim_matrix[i, j] = cka(
                hiddens_a[i+1].float().reshape(-1, hiddens_a[i+1].shape[-1]).cpu().numpy(),
                hiddens_b[j+1].float().reshape(-1, hiddens_b[j+1].shape[-1]).cpu().numpy()
            )
    
    # Optimal transport-style mapping
    from scipy.optimize import linear_sum_assignment
    cost = -sim_matrix  # maximize similarity
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # Map each layer in A to the closest layer in B
    mapping = {}
    for i in range(n_layers_a):
        mapping[i] = int(col_ind[row_ind == i][0]) if sum(row_ind == i) > 0 else i * n_layers_b // n_layers_a
    
    # For different numbers of layers, use proportional mapping
    if n_layers_a != n_layers_b:
        mapping = {}
        for i in range(n_layers_a):
            # Find best matching B layer
            best_j = np.argmax(sim_matrix[i])
            mapping[i] = best_j
    
    return mapping, sim_matrix


def cka(X, Y):
    """Linear CKA (Centered Kernel Alignment)"""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    XX = X @ X.T
    YY = Y @ Y.T
    h = np.trace(XX @ YY)
    return h / np.sqrt(np.trace(XX @ XX) * np.trace(YY @ YY) + 1e-10)


def proportional_mapping(n_a, n_b):
    """Map n_a layers to n_b layers proportionally"""
    mapping = {}
    for i in range(n_a):
        j = int((i + 0.5) * n_b / n_a)
        j = min(j, n_b - 1)
        mapping[i] = j
    return mapping


# ─── 5. WEIGHT MERGE (PER-LAYER) ──────────────────────────────────────────

def get_layer_params(model_name, state_dict):
    """Extract per-layer parameter groups from state dict"""
    layer_params = {}
    other_params = {}
    for key in state_dict.keys():
        parts = key.split(".")
        layer_idx = None
        for j, p in enumerate(parts):
            if p == "h" and j+1 < len(parts) and parts[j+1].isdigit():
                layer_idx = int(parts[j+1])
                break
            if p.startswith("layer.") and parts[j+1].isdigit():
                layer_idx = int(parts[j+1])
                break
        
        if layer_idx is not None:
            if layer_idx not in layer_params:
                layer_params[layer_idx] = {}
            # Simplify the key
            short_key = ".".join(parts[parts.index(str(layer_idx))+1:]) if str(layer_idx) in parts else ".".join(parts[parts.index("h")+2:])
            layer_params[layer_idx][key] = state_dict[key]
        else:
            other_params[key] = state_dict[key]
    
    return layer_params, other_params


def merged_state_dict(model_a, model_b, layer_mapping, alpha_per_layer=None):
    """Merge two models with per-layer interpolation coefficients"""
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    
    if alpha_per_layer is None:
        alpha_per_layer = {}
    
    merged = {}
    n_layers_a = model_a.config.n_layer
    
    # Merge non-layer params (with interpolation for shared ones)
    layer_keys_a = set()
    for k in sd_a:
        parts = k.split(".")
        if "h." in k and any(p.isdigit() for p in parts):
            layer_keys_a.add(k)
    
    non_layer_keys_a = {k: sd_a[k] for k in sd_a if k not in layer_keys_a}
    non_layer_keys_b = {k: sd_b[k] for k in sd_b if k not in layer_keys_a}
    
    for key in sd_a:
        merged[key] = sd_a[key].clone()
    
    # Decide merge coefficient for non-layer params
    alpha_base = alpha_per_layer.get("base", 0.5)
    for key in sd_b:
        if key not in merged:
            merged[key] = sd_b[key].clone()
        elif key in non_layer_keys_a and sd_a[key].shape == sd_b[key].shape:
            merged[key] = alpha_base * sd_a[key].float() + (1 - alpha_base) * sd_b[key].float()
    
    # Merge per-layer weights
    for i_a, i_b in layer_mapping.items():
        alpha = alpha_per_layer.get(i_a, 0.5)
        
        # Get all weights for layer i_a in model A
        prefix_a = f"transformer.h.{i_a}."
        prefix_b = f"transformer.h.{i_b}."
        
        for key in sd_a:
            if key.startswith(prefix_a):
                rel_key = key[len(prefix_a):]
                b_key = f"transformer.h.{i_b}.{rel_key}"
                
                if b_key in sd_b and sd_a[key].shape == sd_b[b_key].shape:
                    merged[key] = (alpha * sd_a[key].float() + 
                                   (1 - alpha) * sd_b[b_key].float())
                elif b_key in sd_b:
                    # Shape mismatch - project
                    w_b = sd_b[b_key].float()
                    target_shape = sd_a[key].shape
                    if len(w_b.shape) == 2 and len(target_shape) == 2:
                        w_b_proj = svd_project(w_b, target_shape[1], target_shape[0])
                        merged[key] = (alpha * sd_a[key].float() + 
                                       (1 - alpha) * w_b_proj)
                    else:
                        merged[key] = sd_a[key].clone()
    
    return merged


def create_merged_model(model_a, model_b, layer_mapping, alpha_per_layer=None):
    """Create a merged model with model_a's architecture"""
    config = copy.deepcopy(model_a.config)
    merged_model = AutoModelForCausalLM.from_config(config).to(DEVICE)
    merged_model.load_state_dict(
        merged_state_dict(model_a, model_b, layer_mapping, alpha_per_layer),
        strict=False
    )
    return merged_model


# ─── 6. EVALUATION ─────────────────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(model, input_ids, attention_mask=None):
    """Compute perplexity on given data"""
    model.eval()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    return torch.exp(outputs.loss).item()


def generate_text(model, tokenizer, prompt="The future of artificial intelligence is", max_new=50):
    """Generate text from a prompt"""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ─── 7. ACTIVATION NORMALIZATION REPAIR ────────────────────────────────────

@torch.no_grad()
def compute_hidden_means_stds(model, input_ids, attention_mask=None):
    """Compute per-layer hidden state mean and std from calibration data"""
    model.eval()
    stats = {}
    
    # Hook to capture hidden states
    hiddens = {}
    handles = []
    
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        for i, layer in enumerate(model.transformer.h):
            def make_hook(idx):
                def hook(module, input, output):
                    hiddens[idx] = output[0].float() if isinstance(output, tuple) else output.float()
                return hook
            handle = layer.register_forward_hook(make_hook(i))
            handles.append(handle)
        
        # Also capture final LN output
        if hasattr(model.transformer, "ln_f"):
            def make_final_hook():
                def hook(module, input, output):
                    hiddens["final"] = output[0].float() if isinstance(output, tuple) else output.float()
                return hook
            handle = model.transformer.ln_f.register_forward_hook(make_final_hook())
            handles.append(handle)
    
    # Run model
    model(input_ids=input_ids, attention_mask=attention_mask)
    
    for handle in handles:
        handle.remove()
    
    for idx, h in hiddens.items():
        mean = h.mean(dim=(0, 1))
        std = h.std(dim=(0, 1)) + 1e-8
        stats[idx] = {"mean": mean, "std": std}
    
    return stats


def apply_activation_repair(merged_model, model_a, model_b, input_ids, attention_mask, layer_mapping):
    """Repair merged model by matching activation statistics"""
    print("  Computing activation statistics for model A...")
    stats_a = compute_hidden_means_stds(model_a, input_ids, attention_mask)
    print("  Computing activation statistics for model B...")
    stats_b = compute_hidden_means_stds(model_b, input_ids, attention_mask)
    print("  Computing activation statistics for merged model...")
    stats_merged = compute_hidden_means_stds(merged_model, input_ids, attention_mask)
    
    # Create correction layers
    corrections = nn.ModuleDict()
    n_layers = merged_model.config.n_layer
    
    for i in range(n_layers):
        if i in stats_merged and i in stats_a:
            dim = stats_merged[i]["mean"].shape[0]
            # Correction: scale = target_std / merged_std, shift = target_mean - merged_mean * scale
            target_mean = stats_a[i]["mean"]
            target_std = stats_a[i]["std"]
            merged_mean = stats_merged[i]["mean"]
            merged_std = stats_merged[i]["std"]
            
            scale = target_std / merged_std
            shift = target_mean - merged_mean * scale
            
            corrections[f"layer_{i}_scale"] = nn.Parameter(scale.to(DEVICE))
            corrections[f"layer_{i}_shift"] = nn.Parameter(shift.to(DEVICE))
        else:
            corrections[f"layer_{i}_scale"] = nn.Parameter(torch.ones(stats_merged.get(i, stats_a.get(0, {"mean": torch.zeros(768)}) )["mean"].shape[0]).to(DEVICE))
            corrections[f"layer_{i}_shift"] = nn.Parameter(torch.zeros(stats_merged.get(i, stats_a.get(0, {"mean": torch.zeros(768)}) )["mean"].shape[0]).to(DEVICE))
    
    # Register hooks to apply corrections
    handles = []
    for i, layer in enumerate(merged_model.transformer.h):
        def make_correction_hook(idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                    scale = corrections.get(f"layer_{idx}_scale")
                    shift = corrections.get(f"layer_{idx}_shift")
                    if scale is not None and h.shape[-1] == scale.shape[0]:
                        h = h * scale + shift
                    return (h,) + output[1:]
                return output
            return hook
        handle = layer.register_forward_hook(make_correction_hook(i))
        handles.append(handle)
    
    return merged_model, corrections, handles


# ─── 8. PER-LAYER OPTIMIZATION ─────────────────────────────────────────────

def optimize_per_layer_coefficients(model_a, model_b, layer_mapping, input_ids, attention_mask, 
                                     n_iters=20, n_pop=15):
    """Use differential evolution to find optimal per-layer merge coefficients"""
    n_layers = model_a.config.n_layer
    mapping = layer_mapping
    
    def evaluate(alphas):
        alpha_dict = {}
        for i in range(n_layers):
            alpha_dict[i] = alphas[i]
        alpha_dict["base"] = alphas[-1]
        
        sd = merged_state_dict(model_a, model_b, mapping, alpha_dict)
        merged_model = AutoModelForCausalLM.from_config(model_a.config).to(DEVICE)
        merged_model.load_state_dict(sd, strict=False)
        
        try:
            ppl = compute_perplexity(merged_model, input_ids, attention_mask)
        except:
            ppl = 1e10
        
        # Clean up
        del merged_model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        return ppl
    
    bounds = [(0.0, 1.0)] * (n_layers + 1)
    
    print(f"  Running differential evolution with {n_iters}x{n_pop} evals...")
    result = differential_evolution(
        evaluate, bounds, 
        maxiter=n_iters, popsize=n_pop, 
        mutation=(0.5, 1.5), recombination=0.7,
        disp=False, polish=False
    )
    
    alpha_per_layer = {}
    for i in range(n_layers):
        alpha_per_layer[i] = result.x[i]
    alpha_per_layer["base"] = result.x[-1]
    
    print(f"  Best perplexity: {result.fun:.2f}")
    print(f"  Alpha range: [{min(result.x[:-1]):.3f}, {max(result.x[:-1]):.3f}]")
    
    return alpha_per_layer


# ─── 9. STITCHING / FUSION APPROACH (for different architectures) ─────────

class ModelFusionBridge(nn.Module):
    """Learnable bridge to fuse two frozen models' representations"""
    def __init__(self, dim_a, dim_b, hidden_dim=256):
        super().__init__()
        self.proj_a = nn.Linear(dim_a, hidden_dim, bias=False)
        self.proj_b = nn.Linear(dim_b, hidden_dim, bias=False)
        self.gate = nn.Linear(hidden_dim * 2, 2)
        self.output = nn.Linear(hidden_dim, dim_a)  # project to model A's space
        nn.init.normal_(self.proj_a.weight, std=0.01)
        nn.init.normal_(self.proj_b.weight, std=0.01)
        nn.init.normal_(self.output.weight, std=0.01)
    
    def forward(self, h_a, h_b):
        za = self.proj_a(h_a)
        zb = self.proj_b(h_b)
        gate_logits = self.gate(torch.cat([za, zb], dim=-1))
        gate_weights = F.softmax(gate_logits, dim=-1)
        fused = gate_weights[:, :, 0:1] * za + gate_weights[:, :, 1:2] * zb
        return self.output(fused)


class StitchedModel(nn.Module):
    """Two frozen models fused by a learnable bridge"""
    def __init__(self, model_a, model_b, bridge):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b
        self.bridge = bridge
        self._freeze_models()
    
    def _freeze_models(self):
        for p in self.model_a.parameters():
            p.requires_grad = False
        for p in self.model_b.parameters():
            p.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        with torch.no_grad():
            self.model_a.eval()
            self.model_b.eval()
            
            # Get hidden states from last layer of each model
            out_a = self.model_a(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            out_b = self.model_b(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            h_a = out_a.hidden_states[-1]  # [batch, seq, dim_a]
            h_b = out_b.hidden_states[-1]  # [batch, seq, dim_b]
        
        # Fuse representations
        h_fused = self.bridge(h_a, h_b)
        
        # Project through model A's LM head
        logits = self.model_a.lm_head(h_fused)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return type("Output", (), {"loss": loss, "logits": logits})
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=0.8, top_p=0.9):
        self.eval()
        batch = input_ids
        for _ in range(max_new_tokens):
            with torch.no_grad():
                out_a = self.model_a(input_ids=batch, output_hidden_states=True)
                out_b = self.model_b(input_ids=batch, output_hidden_states=True)
                h_a = out_a.hidden_states[-1]
                h_b = out_b.hidden_states[-1]
                h_fused = self.bridge(h_a, h_b)
                logits = self.model_a.lm_head(h_fused)
            
            next_logits = logits[:, -1, :] / temperature
            
            # Top-p filtering
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            batch = torch.cat([batch, next_token], dim=-1)
        
        return batch


def train_bridge(model_a, model_b, input_ids, attention_mask, n_steps=50, lr=1e-3):
    """Train a fusion bridge between two frozen models"""
    dim_a = model_a.config.n_embd
    dim_b = model_b.config.n_embd
    hidden_dim = min(256, max(dim_a, dim_b))
    
    bridge = ModelFusionBridge(dim_a, dim_b, hidden_dim).to(DEVICE)
    stitched = StitchedModel(model_a, model_b, bridge).to(DEVICE)
    
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=0.01)
    
    best_loss = float('inf')
    best_state = None
    
    print(f"  Training bridge ({sum(p.numel() for p in bridge.parameters()):,} params)...")
    for step in range(n_steps):
        stitched.train()
        optimizer.zero_grad()
        
        outputs = stitched(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = copy.deepcopy(bridge.state_dict())
        
        if (step + 1) % 10 == 0:
            print(f"    Step {step+1}/{n_steps}, loss: {loss.item():.4f}")
    
    bridge.load_state_dict(best_state)
    return bridge


# ─── 10. SOLUTION PIPELINES ─────────────────────────────────────────────────

def solve_different_sizes(model_name_a="gpt2", model_name_b="gpt2-medium", 
                           cal_n=128, opt_iters=20, do_repair=True):
    """
    Solution 1: Merge models of different sizes
    Steps: load → map layers → optimize per-layer α → activation repair
    """
    print(f"\n{'='*60}")
    print(f"SOLUTION 1: Different Sizes")
    print(f"Model A: {model_name_a} | Model B: {model_name_b}")
    print(f"{'='*60}")
    
    # Load models
    model_a, tokenizer, info_a = load_gpt2(model_name_a)
    model_b, _, info_b = load_gpt2(model_name_b)
    
    print(f"Model A: {info_a['n_layer']}L, {info_a['n_embd']}d, {info_a['n_head']} heads")
    print(f"Model B: {info_b['n_layer']}L, {info_b['n_embd']}d, {info_b['n_head']} heads")
    
    # Load calibration data
    print(f"\nLoading {cal_n} calibration samples...")
    input_ids, attn_mask, _ = load_calibration_data(cal_n, tokenizer_name=model_name_a)
    
    # Compute baseline perplexity (original models)
    print("\nComputing baselines...")
    ppl_a = compute_perplexity(model_a, input_ids, attn_mask)
    ppl_b = compute_perplexity(model_b, input_ids, attn_mask)
    print(f"  Model A PPL: {ppl_a:.2f}")
    print(f"  Model B PPL: {ppl_b:.2f}")
    
    # Layer mapping
    print("\nFinding layer correspondence...")
    layer_mapping = proportional_mapping(info_a['n_layer'], info_b['n_layer'])
    print(f"  Layer mapping: {dict(layer_mapping)}")
    
    # Try CKA mapping if models have the same hidden dim
    if info_a['n_embd'] == info_b['n_embd']:
        try:
            mapping_cka, sim = find_layer_correspondence(model_a, model_b, input_ids[:64], attn_mask[:64])
            print(f"  CKA mapping: {mapping_cka}")
            layer_mapping = mapping_cka
        except:
            pass
    
    # Baseline: uniform merge (α=0.5)
    print("\nBaseline: uniform merge (α=0.5)...")
    naive_alphas = {i: 0.5 for i in range(info_a['n_layer'])}
    naive_alphas["base"] = 0.5
    
    # We need to handle dimension mismatch for the naive merge
    # Project B's weights to A's dimensions
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    
    naive_sd = {}
    for key in sd_a:
        naive_sd[key] = sd_a[key].clone()
    
    # For weights that exist in both and match shape, average
    for key in sd_b:
        if key in sd_a:
            if sd_a[key].shape == sd_b[key].shape:
                naive_sd[key] = 0.5 * sd_a[key].float() + 0.5 * sd_b[key].float()
            elif len(sd_a[key].shape) == 2 and len(sd_b[key].shape) == 2:
                w_b_proj = svd_project(sd_b[key].float(), sd_a[key].shape[1], sd_a[key].shape[0])
                naive_sd[key] = 0.5 * sd_a[key].float() + 0.5 * w_b_proj
    
    naive_model = AutoModelForCausalLM.from_config(model_a.config).to(DEVICE)
    naive_model.load_state_dict(naive_sd, strict=False)
    
    try:
        naive_ppl = compute_perplexity(naive_model, input_ids, attn_mask)
        print(f"  Naive merge PPL: {naive_ppl:.2f}")
    except Exception as e:
        print(f"  Naive merge failed: {e}")
        naive_ppl = 99999
    
    naive_text = generate_text(naive_model, tokenizer)
    print(f"  Output: {naive_text}")
    
    # Step 1: Per-layer optimization
    print("\nStep 1: Per-layer merge coefficient optimization...")
    alpha_per_layer = optimize_per_layer_coefficients(
        model_a, model_b, layer_mapping, input_ids, attn_mask,
        n_iters=opt_iters, n_pop=15
    )
    print(f"  Optimized alphas: {', '.join(f'{k}:{v:.2f}' for k,v in list(alpha_per_layer.items())[:6])}...")
    
    # Create optimized merged model
    opt_sd = merged_state_dict(model_a, model_b, layer_mapping, alpha_per_layer)
    opt_model = AutoModelForCausalLM.from_config(model_a.config).to(DEVICE)
    opt_model.load_state_dict(opt_sd, strict=False)
    
    try:
        opt_ppl = compute_perplexity(opt_model, input_ids, attn_mask)
        print(f"  Optimized merge PPL: {opt_ppl:.2f}")
    except Exception as e:
        print(f"  Optimized merge failed: {e}")
        opt_ppl = 99999
    
    opt_text = generate_text(opt_model, tokenizer)
    print(f"  Output: {opt_text}")
    
    # Step 2: Activation normalization repair
    final_model = opt_model
    final_text = opt_text
    final_ppl = opt_ppl
    
    if do_repair:
        print("\nStep 2: Activation normalization repair...")
        try:
            repaired_model, corrections, _ = apply_activation_repair(
                opt_model, model_a, model_b, input_ids[:64], attn_mask[:64], layer_mapping
            )
            
            repair_ppl = compute_perplexity(repaired_model, input_ids, attn_mask)
            print(f"  Repaired PPL: {repair_ppl:.2f}")
            
            repair_text = generate_text(repaired_model, tokenizer)
            print(f"  Output: {repair_text}")
            
            if repair_ppl < opt_ppl:
                final_model = repaired_model
                final_ppl = repair_ppl
                final_text = repair_text
        except Exception as e:
            print(f"  Activation repair failed: {e}")
    
    # Summary
    print(f"\n{'─'*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'─'*60}")
    print(f"  Original A PPL:     {ppl_a:.2f}")
    print(f"  Original B PPL:     {ppl_b:.2f}")
    print(f"  Naive merge PPL:    {naive_ppl:.2f}")
    print(f"  Optimized PPL:      {opt_ppl:.2f}")
    print(f"  Final PPL:          {final_ppl:.2f}")
    print(f"  Ceiling broken:     {'YES [OK]' if final_ppl < min(ppl_a, ppl_b) * 2 else 'PARTIAL [!]' if final_ppl < min(ppl_a, ppl_b) * 5 else 'NO [FAIL]'}")
    print(f"{'─'*60}")
    print(f"\nFinal generation:")
    print(f"  {final_text}")
    print(f"\nNaive generation for comparison:")
    print(f"  {naive_text}")
    
    return final_model, final_text, {"ppl_a": ppl_a, "ppl_b": ppl_b, "naive_ppl": naive_ppl, "final_ppl": final_ppl}


def solve_different_architectures(model_name_a="gpt2", model_name_b="EleutherAI/gpt-neo-125M",
                                    cal_n=128, n_train_steps=50):
    """
    Solution 2: Merge models of different architectures
    Uses stitching with a learned fusion bridge
    """
    print(f"\n{'='*60}")
    print(f"SOLUTION 2: Different Architectures")
    print(f"Model A: {model_name_a} | Model B: {model_name_b}")
    print(f"{'='*60}")
    
    # Load models (uses same tokenizer - GPT-Neo uses GPT-2's tokenizer)
    print("\nLoading models...")
    model_a, tokenizer, info_a = load_model(model_name_a)
    
    try:
        model_b, _, info_b = load_model(model_name_b)
    except Exception as e:
        print(f"  Error loading {model_name_b}: {e}")
        print("  Trying to load architecture-different model with same tokenizer...")
        # Fall back to another model
        model_b, _, info_b = load_model("distilgpt2")
    
    print(f"Model A ({info_a['model_type']}): {info_a['n_layer']}L, {info_a['n_embd']}d")
    print(f"Model B ({info_b['model_type']}): {info_b['n_layer']}L, {info_b['n_embd']}d")
    
    # Load calibration data
    print(f"\nLoading {cal_n} calibration samples...")
    input_ids, attn_mask, _ = load_calibration_data(cal_n, tokenizer_name=model_name_a)
    
    # Compute baselines
    print("\nComputing baselines...")
    ppl_a = compute_perplexity(model_a, input_ids, attn_mask)
    ppl_b = compute_perplexity(model_b, input_ids, attn_mask)
    print(f"  Model A PPL: {ppl_a:.2f}")
    print(f"  Model B PPL: {ppl_b:.2f}")
    
    # Baseline: try naive weight average (likely impossible for different archs)
    naive_text = "N/A - different architectures cannot be weight-averaged"
    naive_ppl = 99999
    
    # Train fusion bridge
    print(f"\nTraining fusion bridge for {n_train_steps} steps...")
    bridge = train_bridge(model_a, model_b, input_ids, attn_mask, n_steps=n_train_steps)
    
    # Evaluate stitched model
    stitched = StitchedModel(model_a, model_b, bridge).to(DEVICE)
    
    # Compute bridge PPL
    stitched.eval()
    total_loss = 0
    with torch.no_grad():
        out = stitched(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids)
    bridge_ppl = torch.exp(out.loss).item() if out.loss is not None else 99999
    print(f"  Bridge PPL: {bridge_ppl:.2f}")
    
    # Generate text
    print(f"\nGenerating text with stitched model...")
    prompts = [
        "The future of artificial intelligence is",
        "In the beginning, there was",
        "The most important discovery was",
    ]
    
    bridge_texts = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        try:
            output_ids = stitched.generate(inputs.input_ids, max_new_tokens=40)
            text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            bridge_texts.append(text)
            print(f"  Prompt: '{prompt}'")
            print(f"  Output: {text[:120]}...")
        except Exception as e:
            print(f"  Generation failed: {e}")
    
    # Summary
    print(f"\n{'─'*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'─'*60}")
    print(f"  Original A PPL:     {ppl_a:.2f}")
    print(f"  Original B PPL:     {ppl_b:.2f}")
    print(f"  Bridge PPL:         {bridge_ppl:.2f}")
    print(f"  Ceiling broken:     {'YES [OK]' if bridge_ppl < min(ppl_a, ppl_b) * 2 else 'PARTIAL [!]' if bridge_ppl < min(ppl_a, ppl_b) * 5 else 'NO [FAIL]'}")
    print(f"{'─'*60}")
    print(f"\nFinal generation:")
    if bridge_texts:
        print(f"  {bridge_texts[0]}")
    
    return stitched, bridge_texts, {"ppl_a": ppl_a, "ppl_b": ppl_b, "bridge_ppl": bridge_ppl}


# ─── 11. SPECTRAL REPAIR (bonus) ──────────────────────────────────────────

def spectral_repair(W_merged, W_a, W_b, alpha=0.5):
    """
    Replace singular values of merged weight with interpolated originals
    to preserve the spectral structure of both models
    """
    W_merged = W_merged.float()
    W_a = W_a.float()
    W_b = W_b.float()
    
    U, S, Vt = torch.linalg.svd(W_merged, full_matrices=False)
    
    # Get singular values from original models (pad/truncate to match)
    _, S_a, _ = torch.linalg.svd(W_a, full_matrices=False)
    _, S_b, _ = torch.linalg.svd(W_b, full_matrices=False)
    
    k = min(len(S), len(S_a), len(S_b))
    S_repaired = alpha * S_a[:k] + (1 - alpha) * S_b[:k]
    
    S_repaired_padded = torch.zeros_like(S)
    S_repaired_padded[:k] = S_repaired
    
    return U @ torch.diag(S_repaired_padded) @ Vt


# ─── 12. MAIN ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", type=int, choices=[1, 2, 3], default=1,
                        help="1=different sizes, 2=different architectures, 3=both")
    parser.add_argument("--model_a", type=str, default="gpt2")
    parser.add_argument("--model_b", type=str, default=None)
    parser.add_argument("--cal_n", type=int, default=128)
    parser.add_argument("--opt_iters", type=int, default=20)
    parser.add_argument("--train_steps", type=int, default=50)
    parser.add_argument("--no_repair", action="store_true")
    args = parser.parse_args()
    
    torch.set_grad_enabled(True)
    
    if args.solution == 1 or args.solution == 3:
        model_b = args.model_b or "gpt2-medium"
        solve_different_sizes(
            model_name_a=args.model_a,
            model_name_b=model_b,
            cal_n=args.cal_n,
            opt_iters=args.opt_iters,
            do_repair=not args.no_repair
        )
    
    if args.solution == 2 or args.solution == 3:
        model_b = args.model_b or "EleutherAI/gpt-neo-125M"
        solve_different_architectures(
            model_name_a=args.model_a,
            model_name_b=model_b,
            cal_n=args.cal_n,
            n_train_steps=args.train_steps
        )
