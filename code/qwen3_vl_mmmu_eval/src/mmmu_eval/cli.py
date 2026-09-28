"""One CLI for generate -> score -> report, with immutable run manifests."""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import time
import warnings

from .data import SUBJECTS
from .parse import DEFAULT_REGEX
from .utils import (digest, file_hash, git_info, output_lock, package_versions,
                    read_json, source_hashes, write_json)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "mmmu_eval.yaml"
OVERRIDES = {
    "model_path": str, "model_revision": str, "backend": str,
    "max_model_len": int, "max_new_tokens": int, "temperature": float,
    "top_p": float, "top_k": int, "repetition_penalty": float,
    "presence_penalty": float, "seed": int, "min_pixels": int, "max_pixels": int,
    "limit_mm_per_prompt": json.loads, "gpu_memory_utilization": float,
}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage", choices=("all", "generate", "score", "report"), default="all")
    result.add_argument("--config", help="YAML config; saved run config is used by score/report")
    result.add_argument("--output_dir", default=os.environ.get("MMMU_OUTPUT_DIR"),
                        required="MMMU_OUTPUT_DIR" not in os.environ)
    result.add_argument("--data_root", default=os.environ.get("MMMU_DATA_ROOT", "./data"))
    result.add_argument("--require_gpu", default=None)
    result.add_argument("--limit_per_subject", type=int, default=None)
    for name, kind in OVERRIDES.items():
        result.add_argument(f"--{name}", type=kind, default=argparse.SUPPRESS)
    return result


def validate_config(config):
    fixed = {"dtype": "bfloat16", "dataset_path": "MMMU/MMMU",
             "dataset_revision": "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68",
             "split": "validation", "subjects": SUBJECTS, "expected_per_subject": 30,
             "expected_total": 900, "backend": "vllm", "open_question_mode": "qwen_mc"}
    for key, value in fixed.items():
        if config.get(key) != value:
            raise ValueError(f"Fixed protocol field {key} must be {value!r}")
    p = config["prompt"]
    if (p["role"], p["system"], p["content_order"]) != ("user", None, "images_first"):
        raise ValueError("Prompt requires one user message, no system, images first")
    expected_parser = {"version": "qwen_rules_regex_v1", "order": ["option", "text", "regex", "unparsed"],
                       "regex": DEFAULT_REGEX, "regex_selection": "last_valid_match",
                       "unparsed_is_incorrect": True, "refusal_is_incorrect": True}
    if config["parser"] != expected_parser:
        raise ValueError("Parser configuration differs from implemented qwen_rules_regex_v1")
    for key in ("max_model_len", "max_new_tokens", "min_pixels", "max_pixels"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if config["max_new_tokens"] >= config["max_model_len"]:
        raise ValueError("max_model_len must exceed max_new_tokens")
    if config["min_pixels"] > config["max_pixels"]:
        raise ValueError("min_pixels must not exceed max_pixels")
    if not 0 < config["gpu_memory_utilization"] <= 1 or not 0 < config["top_p"] <= 1:
        raise ValueError("gpu_memory_utilization and top_p must be in (0, 1]")
    if config["temperature"] < 0 or (config["top_k"] != -1 and config["top_k"] < 1):
        raise ValueError("Invalid temperature/top_k")
    if config["repetition_penalty"] <= 0 or not -2 <= config["presence_penalty"] <= 2:
        raise ValueError("Invalid repetition/presence penalty")
    limit = config["limit_mm_per_prompt"]
    if not isinstance(limit, dict) or set(limit) != {"image"} or not isinstance(limit["image"], int) or limit["image"] < 1:
        raise ValueError('limit_mm_per_prompt must be {"image": positive integer}')


def resolve_config(args, old=None):
    import yaml
    if args.config or old is None:
        path = Path(args.config) if args.config else DEFAULT_CONFIG
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        config = copy.deepcopy(old["config"])
    for key in OVERRIDES:
        if hasattr(args, key):
            config[key] = getattr(args, key)
    saved_local = (old is not None and old.get("local_model_files") is not None
                   and config["model_path"] == old["config"]["model_path"])
    if Path(config["model_path"]).is_dir() or saved_local:
        if saved_local and not Path(config["model_path"]).is_dir() and args.stage in ("all", "generate"):
            raise ValueError("Local checkpoint directory is missing; restore it before generating")
        if config["model_revision"]:
            warnings.warn("Local model directory: ignoring --model_revision")
        if not saved_local:
            config["model_path"] = str(Path(config["model_path"]).resolve())
        config["model_revision"] = None
    elif not config["model_revision"] or not re.fullmatch(r"[0-9a-f]{40}", config["model_revision"]):
        raise ValueError("Remote model requires an immutable 40-character commit revision")
    validate_config(config)
    if args.limit_per_subject is None and old:
        args.limit_per_subject = old["limit_per_subject"]
    if args.limit_per_subject is not None and not 1 <= args.limit_per_subject <= 30:
        raise ValueError("limit_per_subject must be between 1 and 30")
    return config


def local_model_fingerprint(config):
    root = Path(config["model_path"])
    if not root.is_dir():
        return None
    # Hash weights as well as processor assets, so overwriting a checkpoint
    # in place cannot silently mix outputs. Cost: one sequential read per run.
    extensions = {".safetensors", ".bin", ".json", ".model", ".txt", ".jinja", ".py"}
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix in extensions)
    if not any(path.suffix in (".safetensors", ".bin") for path in paths):
        raise ValueError("Local checkpoint must contain full model weights (merge LoRA adapters first)")
    return {str(path.relative_to(root)): file_hash(path) for path in paths}


def run(args):
    output_dir = Path(args.output_dir)
    with output_lock(output_dir):
        meta_path = output_dir / "run_meta.json"
        old = read_json(meta_path) if meta_path.exists() else None
        if old is None and args.stage in ("score", "report"):
            raise ValueError("Missing run_meta.json; generate first")
        config = resolve_config(args, old)
        code = source_hashes()
        identity = {"config": config, "source_hashes": code, "limit_per_subject": args.limit_per_subject}
        fingerprint = digest(identity)
        if old and old["run_fingerprint"] != fingerprint:
            raise ValueError("Model/config/code/subset changed. Use a NEW output_dir; mixed runs are forbidden.")
        if not old and (output_dir / "raw_outputs.jsonl").exists():
            raise ValueError("Raw outputs exist without a run manifest; refusing to mix data")
        protocol = {key: value for key, value in config.items() if key not in ("model_path", "model_revision")}
        meta = old or {"config": config, "source_hashes": code, "run_fingerprint": fingerprint,
                       "protocol_hash": digest({"config": protocol, "source_hashes": code}),
                       "cli_args": vars(args).copy(), "git": git_info(), "packages": package_versions(),
                       "partial": args.limit_per_subject is not None, "limit_per_subject": args.limit_per_subject,
                       "gpu": None, "peak_vram_mib": None, "generation_seconds": 0,
                       "created_unix": time.time(), "invocations": []}
        meta["invocations"].append({"unix_time": time.time(), "args": vars(args).copy()})
        if args.stage in ("all", "generate"):
            local = local_model_fingerprint(config)
            if old and old.get("local_model_files") != local:
                raise ValueError("Local checkpoint contents changed; use a new output_dir")
            meta["local_model_files"] = local
            for session in meta.get("generation_sessions", []):
                if session["status"] == "running":
                    session["status"] = "process_terminated"
        write_json(meta_path, meta)
        write_json(output_dir / "resolved_config.json", config)
        try:
            if args.stage in ("all", "generate"):
                from .generate import generate
                generate(config, args, meta)
            if args.stage in ("all", "score"):
                from .score import score
                score(config, args, meta)
            if args.stage in ("all", "report"):
                from .report import report
                report(config, args, meta)
        except BaseException as exc:
            meta["last_error"] = {"type": type(exc).__name__, "message": str(exc), "unix_time": time.time()}
            write_json(meta_path, meta)
            raise


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
