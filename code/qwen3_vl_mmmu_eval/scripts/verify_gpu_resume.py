#!/usr/bin/env python3
"""Linux GPU acceptance test: kill a 30-item smoke run, resume, verify skips."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mmmu_eval.utils import read_json, read_jsonl, write_json, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True, help="Must be a NEW directory")
    parser.add_argument("--data_root", default=os.environ.get("MMMU_DATA_ROOT", "./data"))
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--timeout_seconds", type=int, default=7200)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new output directory for the kill/resume acceptance test")
    out.mkdir(parents=True, exist_ok=True)
    command = ["bash", str(ROOT / "scripts" / "run_mmmu_eval.sh"), "--output_dir", str(out),
               "--data_root", args.data_root, "--require_gpu", "A100", "--limit_per_subject", "1"]
    if args.max_model_len is not None:
        command += ["--max_model_len", str(args.max_model_len)]
    raw = out / "raw_outputs.jsonl"
    started = time.monotonic()
    with (out / "interrupted_run.log").open("w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"Run exited before interruption; see {out / 'interrupted_run.log'}")
                if raw.exists() and b"\n" in raw.read_bytes():
                    # Stop the process group first to obtain a stable snapshot,
                    # then kill all vLLM workers as well as the parent process.
                    os.killpg(proc.pid, signal.SIGSTOP)
                    before = read_jsonl(raw, recover_tail=True)
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=30)
                    if not 0 < len(before) < 30:
                        raise RuntimeError("Could not interrupt between subjects; repeat in a new directory")
                    break
                if time.monotonic() - started > args.timeout_seconds:
                    raise TimeoutError("Timed out waiting for first completed subject")
                time.sleep(0.2)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=30)
    print(f"Killed process group after {len(before)} completed subjects; resuming identical command", flush=True)
    # Allow the driver to reclaim memory from killed vLLM workers.
    time.sleep(5)
    with (out / "resumed_run.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                       timeout=args.timeout_seconds)
    after = read_jsonl(raw)
    saved = {row["id"]: row for row in after}
    if len(saved) != 30 or len(after) != 30:
        raise AssertionError("Expected exactly 30 unique smoke outputs")
    log = (out / "resumed_run.log").read_text()
    for row in before:
        assert saved[row["id"]] == row, f"Completed response changed: {row['id']}"
        assert f"Resume: skipping {row['subject']}" in log
    meta = read_json(out / "run_meta.json")
    assert meta["status"] == "complete" and meta["partial"] and "A100" in meta["gpu"]
    write_json(out / "gpu_resume_verified.json", {
        "verified": True, "unix_time": time.time(), "command": command,
        "subjects_preserved": [row["subject"] for row in before],
        "raw_outputs_sha256": file_hash(raw), "protocol_hash": meta["protocol_hash"],
        "run_fingerprint": meta["run_fingerprint"], "gpu": meta["gpu"],
    })
    print(f"GPU smoke + kill/resume verified: {out / 'results_table.md'}")


if __name__ == "__main__":
    main()
