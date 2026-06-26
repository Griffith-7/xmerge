"""xmerge CLI — merge, eval, serve, and manage model merges."""

import argparse, json, math, os, sys, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import merge_prod, utils

SAVE_DIR = merge_prod.SAVE_DIR
DEVICE = merge_prod.DEVICE
DTYPE = merge_prod.DTYPE


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def cmd_merge(args):
    cfg = _load_json(args.config)
    merge_name = cfg.get("name", "unnamed_merge")
    method = cfg.get("method", "bridge")
    train_cfg = cfg.get("training", {})
    cal_cfg = cfg.get("calibration", {})
    n_texts = cal_cfg.get("n_texts", 48)
    max_len = train_cfg.get("max_length", 128)
    steps = train_cfg.get("steps", 20)
    lr = train_cfg.get("lr", 3e-4)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    use_cached = train_cfg.get("use_cached", True)

    print(f"[xmerge] Loading models: {cfg['model_a']} + {cfg['model_b']}")
    t0 = time.time()
    ma = AutoModelForCausalLM.from_pretrained(cfg["model_a"], torch_dtype=DTYPE).to(DEVICE).eval()
    mb = AutoModelForCausalLM.from_pretrained(cfg["model_b"], torch_dtype=DTYPE).to(DEVICE).eval()
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", cfg["model_a"]))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print(f"[xmerge] Loading calibration texts ({n_texts})...")
    calib_texts = merge_prod.load_texts(n_texts)

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, merge_name)

    metrics = {
        "name": merge_name,
        "method": method,
        "model_a": cfg["model_a"],
        "model_b": cfg["model_b"],
        "config": cfg,
    }

    if method == "weight_blend":
        print(f"[xmerge] Running weight-blend merge...")
        merged, _ = merge_prod.merge_same_arch(ma, mb, calib_texts, save_name=merge_name)
        enc = tok(calib_texts[:16], truncation=True, padding=True, max_length=64, return_tensors="pt")
        ids, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)
        metrics["final_ppl"] = round(merge_prod.ppl(merged, ids, mask), 1)

    elif method == "bridge":
        print(f"[xmerge] {'Cached' if use_cached else 'Standard'} bridge training ({steps} steps)...")

        tok_b = AutoTokenizer.from_pretrained(cfg.get("tokenizer_b", cfg["model_b"]))
        if tok_b.pad_token is None:
            tok_b.pad_token = tok_b.eos_token

        token_map = None
        if cfg.get("cross_tokenizer", False):
            print("  Building cross-tokenizer map...")
            token_map = merge_prod.build_token_map(tok, tok_b)

        trainer = merge_prod.train_bridge_cached if use_cached else merge_prod.train_bridge_v2
        bridge = trainer(ma, mb, tok, calib_texts, token_map=token_map,
                         steps=steps, lr=lr, weight_decay=weight_decay, max_len=max_len)

        os.makedirs(save_path, exist_ok=True)
        torch.save(bridge.state_dict(), os.path.join(save_path, "bridge.pt"))
        tok.save_pretrained(save_path)
        info = {
            "type": "trained_bridge",
            "d_a": utils.hidden_dim(ma.config), "d_b": utils.hidden_dim(mb.config),
            "model_a": cfg["model_a"], "model_b": cfg["model_b"],
            "steps": steps, "lr": lr, "weight_decay": weight_decay,
            "has_token_map": token_map is not None,
            "use_cached": use_cached,
        }
        _save_json(os.path.join(save_path, "bridge_config.json"), info)
        metrics["bridge_info"] = info
        metrics["status"] = "saved"

    else:
        print(f"  Unknown method: {method}")
        sys.exit(1)

    # Save merge metrics
    metrics_path = os.path.join(save_path, "metrics.json")
    _save_json(metrics_path, metrics)
    print(f"[xmerge] Metrics saved to {metrics_path}")
    print(f"[xmerge] Done in {time.time() - t0:.1f}s")


def cmd_eval(args):
    if not os.path.exists(args.bridge_dir):
        print(f"Bridge dir not found: {args.bridge_dir}")
        sys.exit(1)

    info = _load_json(os.path.join(args.bridge_dir, "bridge_config.json"))
    print(f"[xmerge] Loading bridge from {args.bridge_dir}")
    print(f"  Model A: {info['model_a']}")
    print(f"  Model B: {info['model_b']}")

    ma = AutoModelForCausalLM.from_pretrained(info["model_a"], torch_dtype=DTYPE).to(DEVICE).eval()
    mb = AutoModelForCausalLM.from_pretrained(info["model_b"], torch_dtype=DTYPE).to(DEVICE).eval()
    tok = AutoTokenizer.from_pretrained(args.bridge_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bridge = merge_prod.OptimalBridge(info["d_a"], info["d_b"])
    state = torch.load(os.path.join(args.bridge_dir, "bridge.pt"), map_location=DEVICE, weights_only=True)
    bridge.load_state_dict(state)
    bridge.to(DEVICE).eval()

    prompts = args.prompts or merge_prod.EVAL_PROMPTS
    print(f"\n  {'='*55}")
    print(f"  Generating with {len(prompts)} prompts")
    print(f"  {'='*55}")
    token_map = None
    for prompt in prompts:
        text = merge_prod.generate_bridge(ma, mb, bridge, tok, prompt, token_map=token_map)
        print(f"\n  [{prompt}]")
        print(f"  -> {text}")

    if args.ppl:
        calib = merge_prod.load_texts(32)
        enc = tok(calib[:16], truncation=True, padding=True, max_length=64, return_tensors="pt")
        ids, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)
        pp_a = merge_prod.ppl(ma, ids, mask)
        lm_head = merge_prod._get_lm_head(ma)
        dtype = next(ma.parameters()).dtype
        with torch.no_grad():
            oa = ma(ids, attention_mask=mask, output_hidden_states=True)
            ob = mb(ids, attention_mask=mask, output_hidden_states=True)
            ha, hb = oa.hidden_states[-1].float(), ob.hidden_states[-1].float()
            k = min(ha.shape[1], hb.shape[1])
            hf = bridge(ha[:, :k], hb[:, :k])
            logits = lm_head(hf.to(dtype))
            sl, ll = logits[..., :-1, :].contiguous(), ids[:, :k][..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1), ignore_index=-100)
            bp = math.exp(loss.item())
        print(f"\n  PPL: A={pp_a:.1f}  Bridge={bp:.1f}")


def cmd_list(args):
    if not os.path.exists(SAVE_DIR):
        print("No saved merges found.")
        return
    entries = sorted(os.listdir(SAVE_DIR))
    if not entries:
        print("No saved merges found.")
        return
    print(f"{'Name':<30} {'Type':<20} {'PPL':<10} {'Models'}")
    print("-" * 80)
    for name in entries:
        path = os.path.join(SAVE_DIR, name)
        if not os.path.isdir(path):
            continue
        info_path = os.path.join(path, "bridge_config.json")
        merge_info_path = os.path.join(path, "merge_info.json")
        metrics_path = os.path.join(path, "metrics.json")

        if os.path.exists(info_path):
            info = _load_json(info_path)
            ppl_str = str(info.get("ppl_bridge", "?"))
            models = f"{info.get('model_a','?')} + {info.get('model_b','?')}"
            mtype = info.get("type", "bridge")
            print(f"{name:<30} {mtype:<20} {ppl_str:<10} {models}")
        elif os.path.exists(merge_info_path):
            info = _load_json(merge_info_path)
            ppl_str = str(info.get("final_ppl", "?"))
            models = "(same arch)"
            mtype = info.get("type", "merge")
            print(f"{name:<30} {mtype:<20} {ppl_str:<10} {models}")
        elif os.path.exists(metrics_path):
            info = _load_json(metrics_path)
            ppl_str = str(info.get("final_ppl", "?"))
            models = f"{info.get('model_a','?')} + {info.get('model_b','?')}"
            mtype = info.get("method", "?")
            print(f"{name:<30} {mtype:<20} {ppl_str:<10} {models}")


def cmd_clean(args):
    print(f"[xmerge] Cleaning GPU memory...")
    merge_prod.clean()
    print(f"  [OK]")


def main():
    parser = argparse.ArgumentParser(prog="xmerge", description="Merge LLMs across architectures and sizes")
    sub = parser.add_subparsers(dest="command", required=True)

    p_merge = sub.add_parser("merge", help="Run a merge from config file")
    p_merge.add_argument("--config", "-c", required=True, help="Path to JSON config file")

    p_eval = sub.add_parser("eval", help="Evaluate a saved bridge")
    p_eval.add_argument("--bridge-dir", "-b", required=True, help="Path to saved bridge directory")
    p_eval.add_argument("--prompts", "-p", nargs="*", default=None, help="Custom prompts")
    p_eval.add_argument("--ppl", action="store_true", help="Compute PPL")

    p_list = sub.add_parser("list", help="List saved merges")

    p_clean = sub.add_parser("clean", help="Clear GPU memory cache")

    args = parser.parse_args()

    if args.command == "merge":
        cmd_merge(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "clean":
        cmd_clean(args)


if __name__ == "__main__":
    main()
