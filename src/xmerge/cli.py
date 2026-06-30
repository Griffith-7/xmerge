"""xmerge CLI — merge, eval, serve, and manage model merges."""

import argparse
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import merge_prod, utils

logger = logging.getLogger("xmerge.cli")

SAVE_DIR = merge_prod.SAVE_DIR
DEVICE = merge_prod.DEVICE


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def cmd_merge(args: argparse.Namespace) -> None:
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
    bridge_type = train_cfg.get("bridge_type", "linear")

    logger.info("Loading models: %s + %s", cfg["model_a"], cfg["model_b"])
    t0 = time.time()
    try:
        ma = AutoModelForCausalLM.from_pretrained(cfg["model_a"]).to(DEVICE).eval()
        mb = AutoModelForCausalLM.from_pretrained(cfg["model_b"]).to(DEVICE).eval()
    except Exception as e:
        logger.error("Failed to load models: %s", e)
        sys.exit(1)

    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", cfg["model_a"]))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    logger.info("Models loaded in %.1fs", time.time() - t0)

    logger.info("Loading calibration texts (%d)...", n_texts)
    calib_texts = merge_prod.load_texts(n_texts)

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, merge_name)

    metrics: Dict[str, Any] = {
        "name": merge_name,
        "method": method,
        "model_a": cfg["model_a"],
        "model_b": cfg["model_b"],
        "config": cfg,
    }

    if method == "weight_blend":
        logger.info("Running weight-blend merge...")
        merged, _ = merge_prod.merge_same_arch(ma, mb, calib_texts, save_name=merge_name)
        enc = tok(calib_texts[:16], truncation=True, padding=True, max_length=64, return_tensors="pt")
        ids = enc.input_ids.to(DEVICE)
        mask = enc.attention_mask.to(DEVICE)
        metrics["final_ppl"] = round(merge_prod.ppl(merged, ids, mask), 1)

    elif method == "bridge":
        logger.info("%s bridge training (%d steps, type=%s)...",
                     "Cached" if use_cached else "Standard", steps, bridge_type)

        tok_b = AutoTokenizer.from_pretrained(cfg.get("tokenizer_b", cfg["model_b"]))
        if tok_b.pad_token is None:
            tok_b.pad_token = tok_b.eos_token

        token_map = None
        if cfg.get("cross_tokenizer", False):
            logger.info("Building cross-tokenizer map...")
            token_map = utils.build_token_map(tok, tok_b)

        trainer = merge_prod.train_bridge_cached if use_cached else merge_prod.train_bridge_v2
        bridge = trainer(
            ma, mb, tok, calib_texts,
            token_map=token_map,
            steps=steps, lr=lr,
            weight_decay=weight_decay,
            max_len=max_len,
            bridge_type=bridge_type,
        )

        os.makedirs(save_path, exist_ok=True)
        torch.save(bridge.state_dict(), os.path.join(save_path, "bridge.pt"))
        tok.save_pretrained(save_path)
        info = {
            "type": "trained_bridge",
            "d_a": utils.hidden_dim(ma.config),
            "d_b": utils.hidden_dim(mb.config),
            "model_a": cfg["model_a"],
            "model_b": cfg["model_b"],
            "steps": steps,
            "lr": lr,
            "weight_decay": weight_decay,
            "bridge_type": bridge_type,
            "has_token_map": token_map is not None,
            "use_cached": use_cached,
        }
        _save_json(os.path.join(save_path, "bridge_config.json"), info)
        metrics["bridge_info"] = info
        metrics["status"] = "saved"

    else:
        logger.error("Unknown method: %s", method)
        sys.exit(1)

    # Save metrics
    metrics_path = os.path.join(save_path, "metrics.json")
    _save_json(metrics_path, metrics)
    logger.info("Done in %.1fs. Results saved to %s", time.time() - t0, save_path)


def cmd_eval(args: argparse.Namespace) -> None:
    if not os.path.exists(args.bridge_dir):
        logger.error("Bridge dir not found: %s", args.bridge_dir)
        sys.exit(1)

    info = _load_json(os.path.join(args.bridge_dir, "bridge_config.json"))
    logger.info("Loading bridge from %s", args.bridge_dir)
    logger.info("  Model A: %s", info["model_a"])
    logger.info("  Model B: %s", info["model_b"])

    try:
        ma = AutoModelForCausalLM.from_pretrained(info["model_a"]).to(DEVICE).eval()
        mb = AutoModelForCausalLM.from_pretrained(info["model_b"]).to(DEVICE).eval()
    except Exception as e:
        logger.error("Failed to load models: %s", e)
        sys.exit(1)

    tok = AutoTokenizer.from_pretrained(args.bridge_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    d_a = info.get("d_a", utils.hidden_dim(ma.config))
    d_b = info.get("d_b", utils.hidden_dim(mb.config))
    bridge_type = info.get("bridge_type", "linear")

    if bridge_type == "mlp":
        bridge = merge_prod.MLPBridge(d_a, d_b)
    else:
        bridge = merge_prod.OptimalBridge(d_a, d_b)

    state = torch.load(
        os.path.join(args.bridge_dir, "bridge.pt"),
        map_location=DEVICE,
        weights_only=True,
    )
    bridge.load_state_dict(state)
    bridge.to(DEVICE).eval()

    prompts = args.prompts or merge_prod.EVAL_PROMPTS
    logger.info("Generating with %d prompts", len(prompts))
    for prompt in prompts:
        text = merge_prod.generate_bridge(ma, mb, bridge, tok, prompt)
        print(f"\n[{prompt}]")
        print(f"  -> {text}")

    if args.ppl:
        calib = merge_prod.load_texts(32)
        enc = tok(calib[:16], truncation=True, padding=True, max_length=64, return_tensors="pt")
        ids = enc.input_ids.to(DEVICE)
        mask = enc.attention_mask.to(DEVICE)
        pp_a = merge_prod.ppl(ma, ids, mask)
        lm_head = merge_prod._get_lm_head(ma)
        dtype = next(ma.parameters()).dtype
        with torch.no_grad():
            oa = ma(ids, attention_mask=mask, output_hidden_states=True)
            ob = mb(ids, attention_mask=mask, output_hidden_states=True)
            ha = oa.hidden_states[-1].float()
            hb = ob.hidden_states[-1].float()
            k = min(ha.shape[1], hb.shape[1])
            hf = bridge(ha[:, :k], hb[:, :k])
            if lm_head:
                logits = lm_head(hf.to(dtype))
                sl = logits[..., :-1, :].contiguous()
                ll = ids[:, :k][..., 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    sl.view(-1, sl.size(-1)), ll.view(-1), ignore_index=-100
                )
                bp = math.exp(loss.item())
            else:
                bp = float("inf")
        logger.info("PPL: A=%.1f  Bridge=%.1f", pp_a, bp)


def cmd_list(args: argparse.Namespace) -> None:
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


def cmd_clean(args: argparse.Namespace) -> None:
    logger.info("Cleaning GPU memory...")
    merge_prod.clean()
    logger.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="xmerge",
        description="Merge LLMs across architectures and sizes — representation-level merging",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_merge = sub.add_parser("merge", help="Run a merge from config file")
    p_merge.add_argument("--config", "-c", required=True, help="Path to JSON config file")

    p_eval = sub.add_parser("eval", help="Evaluate a saved bridge")
    p_eval.add_argument("--bridge-dir", "-b", required=True, help="Path to saved bridge directory")
    p_eval.add_argument("--prompts", "-p", nargs="*", default=None, help="Custom prompts")
    p_eval.add_argument("--ppl", action="store_true", help="Compute PPL")

    sub.add_parser("list", help="List saved merges")

    sub.add_parser("clean", help="Clear GPU memory cache")

    args = parser.parse_args()
    _setup_logging(args.verbose if hasattr(args, "verbose") else False)

    commands = {
        "merge": cmd_merge,
        "eval": cmd_eval,
        "list": cmd_list,
        "clean": cmd_clean,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
