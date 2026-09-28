import re
"""GPU inference, full preflight, subject-level crash recovery."""
import os
from pathlib import Path
import time
import warnings

from .data import load_data, normalize_row
from .prompt import build_messages
from .utils import (VRAMMonitor, append_subject, check_gpu, digest, file_hash,
                    package_versions, read_jsonl, write_json, write_jsonl)


def prepare_inputs_for_vllm(messages, processor, vision_fn=None):
    if vision_fn is None:
        from qwen_vl_utils import process_vision_info
        vision_fn = process_vision_info
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, kwargs = vision_fn(
        messages, image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True, return_video_metadata=True)
    mm_data = {}
    if images is not None:
        mm_data["image"] = images
    if videos is not None:
        mm_data["video"] = videos
    return {"prompt": text, "multi_modal_data": mm_data, "mm_processor_kwargs": kwargs}


def count_input_tokens(request, processor):
    # Qwen processor expands each image placeholder using image_grid_thw and
    # merge_size. Plain tokenizer.encode(prompt) would miss these image tokens.
    mm = request["multi_modal_data"]
    if "video" in mm:
        raise ValueError("MMMU protocol expects images only")
    encoded = processor(text=[request["prompt"]], images=mm.get("image"),
                        **request["mm_processor_kwargs"])
    return len(encoded["input_ids"][0])


def prepare(row, config, processor):
    return prepare_inputs_for_vllm(build_messages(
        row, config["min_pixels"], config["max_pixels"], config["prompt"]), processor)


def preflight(datasets, config, processor, output_dir):
    lengths, overflow = {}, []
    for subject, ds in datasets.items():
        for item in ds:
            row = normalize_row(item, subject)
            if len(row["images"]) > config["limit_mm_per_prompt"]["image"]:
                raise ValueError(f"{row['id']}: exceeds limit_mm_per_prompt; images are never dropped")
            request = prepare(row, config, processor)
            count = count_input_tokens(request, processor)
            entry = {"id": row["id"], "subject": subject, "num_images": len(row["images"]),
                     "input_tokens": count, "prompt_sha256": digest(request["prompt"])}
            lengths[row["id"]] = entry
            if count + config["max_new_tokens"] > config["max_model_len"]:
                overflow.append(entry)
            del request, row
        print(f"Preflight: {subject} ({len(ds)} questions)", flush=True)
    write_json(Path(output_dir) / "input_lengths.json", lengths)
    write_json(Path(output_dir) / "overflow_report.json", overflow)
    if overflow:
        print("id\tsubject\tnum_images\tinput_tokens", flush=True)
        for row in overflow:
            print("\t".join(str(row[key]) for key in ("id", "subject", "num_images", "input_tokens")), flush=True)
        raise ValueError(f"{len(overflow)} inputs overflow max_model_len. Increase --max_model_len "
                         "explicitly and use a new output directory; no questions were skipped or resized.")
    return lengths


def recover_subjects(path, expected, validate_row=None):
    rows = read_jsonl(path, recover_tail=True)
    grouped = {subject: [] for subject in expected}
    seen = set()
    for row in rows:
        subject, ident = row["subject"], row["id"]
        if subject not in expected or ident not in expected[subject]:
            raise ValueError(f"Unexpected saved question: {subject}/{ident}")
        if ident in seen:
            raise ValueError(f"Duplicate saved question: {ident}")
        seen.add(ident)
        if validate_row:
            validate_row(row)
        grouped[subject].append(row)
    completed, kept = set(), []
    for subject, saved in grouped.items():
        if {row["id"] for row in saved} == set(expected[subject]):
            completed.add(subject)
            kept.extend(saved)
        elif saved:
            warnings.warn(f"Discarding incomplete subject {subject}: {len(saved)} rows; regenerating")
    # Atomic replacement also fixes a final JSON object whose newline was torn.
    write_jsonl(path, kept)
    return completed


def generate(config, args, meta):
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    output_dir = Path(args.output_dir)
    meta_path = output_dir / "run_meta.json"
    gpu = check_gpu(args.require_gpu)
    versions = package_versions()
    if meta.get("generation_packages") and meta["generation_packages"] != versions:
        raise ValueError("Package versions changed; resume requires the original environment")
    if meta.get("gpu") and meta["gpu"] != gpu:
        raise ValueError("GPU changed; use the original GPU type for resume")
    meta.update(gpu=gpu, generation_packages=versions, packages=versions, status="preflight")
    write_json(meta_path, meta)
    from transformers import AutoConfig, AutoProcessor
    revision = config["model_revision"]
    model_config = AutoConfig.from_pretrained(config["model_path"], revision=revision)
    if getattr(model_config, "quantization_config", None):
        raise ValueError("Quantized checkpoints are forbidden; use full bfloat16 weights")
    processor = AutoProcessor.from_pretrained(config["model_path"], revision=revision)
    if isinstance(processor.chat_template, str):
        from .utils import atomic_text
        atomic_text(output_dir / "chat_template.jinja", processor.chat_template)
    meta["processor"] = {"class": type(processor).__name__,
                         "image_processor": processor.image_processor.to_dict(),
                         "chat_template_sha256": digest(processor.chat_template)}
    datasets, stats = load_data(config, args.data_root)
    n = args.limit_per_subject or 30
    expected = {subject: list(ds["id"][:n]) for subject, ds in datasets.items()}
    meta.update(data_counts=stats, evaluated_counts={subject: n for subject in datasets},
                evaluated_total=n * len(datasets), expected_ids=expected)
    write_json(meta_path, meta)
    # Even a debug run validates lengths for the complete 900-question dataset.
    lengths = preflight(datasets, config, processor, output_dir)
    def validate_saved(row):
        entry = lengths[row["id"]]
        if (row["seed"] != config["seed"] or digest(row["prompt_text"]) != entry["prompt_sha256"]
                or row["input_tokens"] != entry["input_tokens"]):
            raise ValueError(f"Saved output provenance mismatch: {row['id']}")
    completed = recover_subjects(output_dir / "raw_outputs.jsonl", expected, validate_saved)
    if len(completed) == len(datasets):
        print("All subjects complete; skipping generation", flush=True)
        meta["status"] = "generated"
        meta["generation_seconds"] = sum(x.get("elapsed_seconds", 0) for x in meta.get("generation_sessions", []))
        meta["raw_outputs_sha256"] = file_hash(output_dir / "raw_outputs.jsonl")
        write_json(meta_path, meta)
        return
    from vllm import LLM, SamplingParams
    started = time.monotonic()
    session = {"started_unix": time.time(), "completed_subjects": [], "status": "running"}
    meta.setdefault("generation_sessions", []).append(session)
    monitor = VRAMMonitor(output_dir / "vram_samples.jsonl")
    meta["status"] = "generating"
    write_json(meta_path, meta)
    try:
        with monitor:
            llm = LLM(model=config["model_path"], revision=revision, dtype="bfloat16",
                      max_model_len=config["max_model_len"],
                      gpu_memory_utilization=config["gpu_memory_utilization"],
                      limit_mm_per_prompt=config["limit_mm_per_prompt"], seed=config["seed"],
                      generation_config="vllm")
            # generation_config=vllm prevents model generation_config.json from
            # silently supplying unrecorded defaults. Save resolved SamplingParams.
            sample_kwargs = {key: config[key] for key in (
                "temperature", "top_p", "top_k", "repetition_penalty", "presence_penalty", "seed")}
            sample_kwargs.update(max_tokens=config["max_new_tokens"], n=1, stop_token_ids=[])
            meta["sampling_params_repr"] = repr(SamplingParams(**sample_kwargs))
            for subject, ds in datasets.items():
                if subject in completed:
                    print(f"Resume: skipping {subject}", flush=True)
                    continue
                rows = [normalize_row(ds[index], subject) for index in range(n)]
                requests = [prepare(row, config, processor) for row in rows]
                outputs = llm.generate(requests, sampling_params=[SamplingParams(**sample_kwargs) for _ in rows])
                if len(outputs) != len(rows):
                    raise ValueError(f"{subject}: missing generated outputs")
                saved = []
                for row, request, output in zip(rows, requests, outputs):
                    # vLLM may return the multimodal-processed prompt (expanded
                    # <|image_pad|> runs) or None; compare after collapsing pads.
                    # Order/input identity is enforced by the token-count check below.
                    def _norm(p):
                        return re.sub(r"(<\|image_pad\|>)+", "<|image_pad|>", p)
                    if output.prompt is not None and _norm(output.prompt) != _norm(request["prompt"]):
                        print("EXPECTED:", repr(request["prompt"][:300]), flush=True)
                        print("GOT     :", repr(output.prompt[:300]), flush=True)
                        raise ValueError(f"vLLM returned unexpected prompt/order: {row['id']}")
                    actual = len(output.prompt_token_ids)
                    if actual != lengths[row["id"]]["input_tokens"]:
                        raise ValueError(f"{row['id']}: preflight/vLLM token mismatch "
                                         f"{lengths[row['id']]['input_tokens']} != {actual}")
                    if len(output.outputs) != 1:
                        raise ValueError(f"{row['id']}: expected one completion")
                    result = output.outputs[0]
                    if result.finish_reason not in ("stop", "length"):
                        raise ValueError(f"{row['id']}: unexpected finish_reason {result.finish_reason}")
                    saved.append({key: row[key] for key in ("id", "subject", "question_type", "answer", "options")}
                                 | {"prompt_text": request["prompt"], "num_images": len(row["images"]),
                                    "response": result.text, "input_tokens": actual,
                                    "output_tokens": len(result.token_ids), "finish_reason": result.finish_reason,
                                    "truncated": result.finish_reason == "length", "gpu": gpu,
                                    "seed": config["seed"]})
                append_subject(output_dir / "raw_outputs.jsonl", saved)
                session["completed_subjects"].append(subject)
                session["elapsed_seconds"] = time.monotonic() - started
                meta["peak_vram_mib"] = max(meta.get("peak_vram_mib") or 0, monitor.peak or 0) or None
                write_json(meta_path, meta)
                print(f"Saved {subject}: {len(saved)}", flush=True)
                del rows, requests, outputs, saved
        if monitor.error:
            raise RuntimeError(f"VRAM monitoring failed: {monitor.error}")
        session["status"] = "complete"
        meta["status"] = "generated"
        meta["raw_outputs_sha256"] = file_hash(output_dir / "raw_outputs.jsonl")
    except BaseException as exc:
        session["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        meta["status"] = session["status"]
        session["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        session["elapsed_seconds"] = time.monotonic() - started
        meta["generation_seconds"] = sum(x.get("elapsed_seconds", 0) for x in meta["generation_sessions"])
        meta["generation_time_note"] = "Includes engine startup; SIGKILL sessions retain last checkpoint elapsed time. Excludes preflight."
        meta["peak_vram_mib"] = max(meta.get("peak_vram_mib") or 0, monitor.peak or 0) or None
        meta["vram_note"] = "Device-total peak across nvidia-smi rows; vLLM preallocates gpu_memory_utilization of VRAM."
        meta["vram_monitor_error"] = monitor.error
        write_json(meta_path, meta)
