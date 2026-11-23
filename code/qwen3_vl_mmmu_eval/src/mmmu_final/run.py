"""Final evaluation runner: one model x one set (mmmu_test | pro_standard | pro_vision).

The ID file decides the questions: configs/final_eval_ids.json (450 samples) or
configs/final_eval_ids_full.json (full splits: MMMU test 10,500, MMMU-Pro 1,730).
Generation runs subject by subject in chunks of --chunk_size questions; every finished chunk
is appended to raw_outputs.jsonl, so a resumed run regenerates only unsaved questions
(question-level resume; a torn last line is dropped). Each request carries the same seed, so
chunking does not change the sampling settings. The preflight (prompt + input token count per
question) is cached in input_lengths.json and reused on resume when its hash matches run_meta.

Reuses the validation protocol pieces without modifying src/mmmu_eval: final_16k.yaml sampling,
resolution and vLLM settings, mmmu_eval.generate.prepare_inputs_for_vllm/count_input_tokens,
subject-level resume (recover_subjects) and the output lock.

max_model_len is a capacity limit, not a sampling setting: it is max(final_16k value,
largest input + max_new_tokens rounded up to 1024) so multi-image MMMU-Pro inputs never
overflow. The value used is recorded in run_meta.json.

Usage (Colab):
  PYTHONPATH=<project>/src python -m mmmu_final.run --set pro_vision --model_tag submission \
     --model_path /content/model --output_root <drive>/final_eval --data_root <drive>/cache \
     --require_gpu A100-SXM4-40GB
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import time

import yaml

from mmmu_eval.data import SUBJECTS
from mmmu_eval.utils import (VRAMMonitor, append_subject, check_gpu, digest, file_hash, output_lock,
                             package_versions, read_jsonl, write_json, write_jsonl)
from .scoring import score_rows, summarize
from .sets import SETS, build_set_messages, get_row, load_set

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT / "configs" / "final_16k.yaml"
DEFAULT_IDS = PROJECT / "configs" / "final_eval_ids.json"
EXPECTED = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.0, "presence_penalty": 1.5,
            "min_pixels": 1310720, "max_pixels": 5242880, "max_new_tokens": 16384, "max_model_len": 24576}
BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
BASE_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


def final_source_hashes():
    here = Path(__file__).resolve().parent
    files = sorted(here.glob("*.py")) + [PROJECT / "src/mmmu_eval" / n for n in
                                         ("data.py", "prompt.py", "parse.py", "parse_v2.py", "generate.py", "utils.py")]
    return {f"{p.parent.name}/{p.name}": file_hash(p) for p in files}


def model_fingerprint(path):
    root = Path(path)
    if not root.is_dir():
        return None
    exts = {".safetensors", ".json", ".jinja", ".txt", ".model", ".py"}
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in exts)
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("Local model directory must contain full merged weights")
    return {str(p.relative_to(root)): file_hash(p) for p in files}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=SETS)
    ap.add_argument("--model_tag", required=True, help="e.g. baseline / submission (output folder name)")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_revision", default=None, help="40-char commit; required for a Hub model id")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--ids", default=str(DEFAULT_IDS))
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--require_gpu", default=None)
    ap.add_argument("--stage", choices=("all", "generate", "score"), default="all")
    ap.add_argument("--chunk_size", type=int, default=256, help="questions per vLLM call / save unit")
    return ap.parse_args(argv)


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for k, v in EXPECTED.items():
        if cfg[k] != v:
            raise ValueError(f"{path}: {k}={cfg[k]} differs from the validation protocol ({v})")
    return cfg


def prepare(row, set_name, cfg, processor, vision_fn=None):
    from mmmu_eval.generate import prepare_inputs_for_vllm
    return prepare_inputs_for_vllm(build_set_messages(row, set_name, cfg["min_pixels"], cfg["max_pixels"]),
                                   processor, vision_fn)


def preflight(set_name, index, datasets, cfg, processor, out_dir, vision_fn=None):
    from mmmu_eval.generate import count_input_tokens
    lengths = {}
    for subject in SUBJECTS:
        for i in index[subject]:
            row = get_row(set_name, datasets[subject], i, subject)
            if len(row["images"]) > cfg["limit_mm_per_prompt"]["image"]:
                raise ValueError(f"{row['id']}: {len(row['images'])} images exceed limit_mm_per_prompt")
            req = prepare(row, set_name, cfg, processor, vision_fn)
            lengths[row["id"]] = {"subject": subject, "num_images": len(row["images"]),
                                  "input_tokens": count_input_tokens(req, processor),
                                  "prompt_sha256": digest(req["prompt"])}
        print(f"Preflight {set_name}: {subject}", flush=True)
    write_json(out_dir / "input_lengths.json", lengths)
    longest = max(v["input_tokens"] for v in lengths.values())
    mml = max(cfg["max_model_len"], int(math.ceil((longest + cfg["max_new_tokens"]) / 1024) * 1024))
    return lengths, longest, mml


def recover_rows(path, expected, validate_row):
    """Keep every valid saved question (any subject, any chunk); rewrite the file atomically."""
    if not Path(path).exists():
        return set()
    allowed = {i: s for s, ids in expected.items() for i in ids}
    kept, seen = [], set()
    for row in read_jsonl(path, recover_tail=True):
        ident = row["id"]
        if allowed.get(ident) != row["subject"]:
            raise ValueError(f"Unexpected saved question: {row['subject']}/{ident}")
        if ident in seen:
            raise ValueError(f"Duplicate saved question: {ident}")
        validate_row(row)
        seen.add(ident); kept.append(row)
    write_jsonl(path, kept)
    return seen


def generate(args, cfg, ids_spec, out_dir, meta, loader=None, llm_factory=None, processor=None, vision_fn=None):
    meta_path = out_dir / "run_meta.json"
    gpu = check_gpu(args.require_gpu) if llm_factory is None else "TEST"
    if meta.get("gpu") and meta["gpu"] != gpu:
        raise ValueError(f"GPU changed ({meta['gpu']} -> {gpu}); resume on the same GPU type")
    meta["gpu"] = gpu
    index, datasets, expected = load_set(args.set, ids_spec, args.data_root, loader)
    if processor is None:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model_path, revision=args.model_revision)
    lp = out_dir / "input_lengths.json"
    if meta.get("input_lengths_sha256") and lp.is_file() and file_hash(lp) == meta["input_lengths_sha256"]:
        lengths = json.loads(lp.read_text(encoding="utf-8"))
        if set(lengths) != {i for v in expected.values() for i in v}:
            raise ValueError("cached input_lengths.json does not match the ID file")
        longest = max(v["input_tokens"] for v in lengths.values())
        mml = max(cfg["max_model_len"], int(math.ceil((longest + cfg["max_new_tokens"]) / 1024) * 1024))
        print(f"Preflight: reusing cached input_lengths.json ({len(lengths)} questions)", flush=True)
    else:
        lengths, longest, mml = preflight(args.set, index, datasets, cfg, processor, out_dir, vision_fn)
        meta["input_lengths_sha256"] = file_hash(lp)
    if meta.get("max_model_len_used") not in (None, mml):
        raise ValueError(f"max_model_len changed ({meta['max_model_len_used']} -> {mml})")
    meta.update(max_input_tokens=longest, max_model_len_used=mml, status="preflight")
    write_json(meta_path, meta)
    print(f"Preflight: {len(lengths)} questions, longest input {longest} tokens -> max_model_len {mml}", flush=True)

    def validate_saved(row):
        e = lengths[row["id"]]
        if row["seed"] != args.seed or digest(row["prompt_text"]) != e["prompt_sha256"] or row["input_tokens"] != e["input_tokens"]:
            raise ValueError(f"Saved output provenance mismatch: {row['id']}")
    raw = out_dir / "raw_outputs.jsonl"
    done = recover_rows(raw, expected, validate_saved)
    total = sum(len(v) for v in expected.values())
    if len(done) == total:
        print("All questions complete; skipping generation", flush=True)
        meta["raw_outputs_sha256"] = file_hash(raw); meta["status"] = "generated"
        write_json(meta_path, meta)
        return
    sample_kwargs = {k: cfg[k] for k in ("temperature", "top_p", "top_k", "repetition_penalty", "presence_penalty")}
    sample_kwargs.update(seed=args.seed, max_tokens=cfg["max_new_tokens"], n=1, stop_token_ids=[])
    if llm_factory is None:
        from vllm import LLM, SamplingParams
        llm = LLM(model=args.model_path, revision=args.model_revision, dtype="bfloat16", max_model_len=mml,
                  gpu_memory_utilization=cfg["gpu_memory_utilization"], limit_mm_per_prompt=cfg["limit_mm_per_prompt"],
                  seed=args.seed, generation_config="vllm")
        meta["sampling_params_repr"] = repr(SamplingParams(**sample_kwargs))
        make_params = lambda: SamplingParams(**sample_kwargs)
    else:
        llm, make_params = llm_factory(), (lambda: dict(sample_kwargs))
    print(f"Resume: {len(done)}/{total} questions already saved; generating {total - len(done)}", flush=True)
    started = time.monotonic(); session = {"started_unix": time.time(), "completed_subjects": [], "status": "running",
                                           "start_saved": len(done), "generated": 0, "output_tokens": 0}
    meta.setdefault("generation_sessions", []).append(session); meta["status"] = "generating"; write_json(meta_path, meta)
    monitor = VRAMMonitor(out_dir / "vram_samples.jsonl") if llm_factory is None else None
    try:
        if monitor: monitor.__enter__()
        n_saved = len(done)
        for subject in SUBJECTS:
            todo = [i for ident, i in zip(expected[subject], index[subject]) if ident not in done]
            if not todo:
                continue
            chunks = [todo[k:k + args.chunk_size] for k in range(0, len(todo), args.chunk_size)]
            for ci, chunk in enumerate(chunks, 1):
                rows = [get_row(args.set, datasets[subject], i, subject) for i in chunk]
                reqs = [prepare(r, args.set, cfg, processor, vision_fn) for r in rows]
                outs = llm.generate(reqs, sampling_params=[make_params() for _ in rows])
                if len(outs) != len(rows):
                    raise ValueError(f"{subject}: missing outputs")
                saved = []
                for row, req, o in zip(rows, reqs, outs):
                    norm = lambda p: re.sub(r"(<\|image_pad\|>)+", "<|image_pad|>", p)
                    if o.prompt is not None and norm(o.prompt) != norm(req["prompt"]):
                        raise ValueError(f"vLLM returned unexpected prompt/order: {row['id']}")
                    if len(o.prompt_token_ids) != lengths[row["id"]]["input_tokens"]:
                        raise ValueError(f"{row['id']}: preflight/vLLM token mismatch")
                    res = o.outputs[0]
                    if res.finish_reason not in ("stop", "length"):
                        raise ValueError(f"{row['id']}: finish_reason {res.finish_reason}")
                    saved.append({k: row[k] for k in ("id", "subject", "question_type", "answer", "options", "origin")}
                                 | {"set": args.set, "prompt_text": req["prompt"], "num_images": len(row["images"]),
                                    "response": res.text, "input_tokens": len(o.prompt_token_ids),
                                    "output_tokens": len(res.token_ids), "finish_reason": res.finish_reason,
                                    "truncated": res.finish_reason == "length", "gpu": gpu, "seed": args.seed})
                append_subject(raw, saved)
                n_saved += len(saved); session["generated"] += len(saved)
                session["output_tokens"] += sum(r["output_tokens"] for r in saved)
                el = time.monotonic() - started; session["elapsed_seconds"] = el
                if ci == len(chunks):
                    session["completed_subjects"].append(subject)
                write_json(meta_path, meta)
                rate = session["generated"] / el if el else 0.0
                eta = (total - n_saved) / rate / 3600 if rate else float("nan")
                print(f"Saved {args.set} {subject} chunk {ci}/{len(chunks)}: {len(saved)} | total {n_saved}/{total} | "
                      f"{session['output_tokens'] / el:.0f} out tok/s, {3600 * rate:.0f} q/h | ETA {eta:.1f} h", flush=True)
        session["status"] = "complete"
        meta["raw_outputs_sha256"] = file_hash(raw); meta["status"] = "generated"
    except BaseException as exc:
        session["status"] = "failed"; session["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if monitor:
            monitor.__exit__(None, None, None); meta["peak_vram_mib"] = monitor.peak
        session["elapsed_seconds"] = time.monotonic() - started
        meta["generation_seconds"] = sum(s.get("elapsed_seconds", 0) for s in meta["generation_sessions"])
        write_json(meta_path, meta)


def score(args, cfg, ids_spec, out_dir, meta):
    raw = out_dir / "raw_outputs.jsonl"
    if not meta.get("raw_outputs_sha256") or file_hash(raw) != meta["raw_outputs_sha256"]:
        raise ValueError("raw_outputs.jsonl missing or changed; finish generation first")
    from .sets import expected_ids
    expected, _ = expected_ids(args.set, ids_spec)
    by_id = {r["id"]: r for r in read_jsonl(raw)}
    order = [i for s in SUBJECTS for i in expected[s]]
    if set(by_id) != set(order) or len(by_id) != len(order):
        raise ValueError(f"Expected the {len(order)} fixed IDs, got {len(by_id)}")
    scored = score_rows(args.set, [by_id[i] for i in order], cfg["parser"]["regex"])
    summary = summarize(args.set, scored, SUBJECTS)
    summary.update(model_tag=args.model_tag, gpu=meta["gpu"], seed=args.seed, max_model_len_used=meta.get("max_model_len_used"),
                   ids_sha256=meta["ids_sha256"], run_fingerprint=meta["run_fingerprint"])
    write_jsonl(out_dir / "scored.jsonl", scored)
    write_json(out_dir / "scores.json", summary)
    meta["status"] = "scored"; meta["scores_sha256"] = file_hash(out_dir / "scores.json")
    write_json(out_dir / "run_meta.json", meta)
    print(f"Scored {args.set} [{args.model_tag}]: {summary['correct']}/{summary['n']} = {100*summary['accuracy']:.2f}% | "
          f"trunc {summary['trunc_pct']:.1f}%", flush=True)
    return summary


def main(argv=None, **test_hooks):
    args = parse_args(argv)
    if not Path(args.model_path).is_dir():
        if args.model_path != BASE_MODEL or args.model_revision != BASE_REVISION:
            raise ValueError(f"Hub model must be {BASE_MODEL}@{BASE_REVISION}")
    cfg = load_config(args.config)
    ids_spec = json.loads(Path(args.ids).read_text(encoding="utf-8"))
    out_dir = Path(args.output_root) / args.model_tag / args.set
    out_dir.mkdir(parents=True, exist_ok=True)
    with output_lock(out_dir):
        meta_path = out_dir / "run_meta.json"
        old = json.loads(meta_path.read_text()) if meta_path.exists() else None
        identity = {"set": args.set, "model_path": args.model_path, "model_revision": args.model_revision,
                    "model_files": model_fingerprint(args.model_path), "seed": args.seed,
                    "config": {k: cfg[k] for k in sorted(cfg) if k not in ("model_path", "model_revision")},
                    "ids_sha256": file_hash(args.ids), "sources": final_source_hashes()}
        fp = digest(identity)
        if old and old["run_fingerprint"] != fp:
            raise ValueError("Model/config/IDs/code changed since this folder was started. Use a new --output_root.")
        if not old and (out_dir / "raw_outputs.jsonl").exists():
            raise ValueError("raw_outputs.jsonl exists without run_meta.json; refusing to mix data")
        meta = old or {"run_fingerprint": fp, "identity": identity, "ids_sha256": identity["ids_sha256"],
                       "packages": package_versions(), "created_unix": time.time(), "invocations": []}
        meta["invocations"].append({"unix_time": time.time(), "argv": vars(args)})
        write_json(meta_path, meta)
        if args.stage in ("all", "generate"):
            generate(args, cfg, ids_spec, out_dir, meta, **test_hooks)
        if args.stage in ("all", "score"):
            return score(args, cfg, ids_spec, out_dir, meta)


if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
