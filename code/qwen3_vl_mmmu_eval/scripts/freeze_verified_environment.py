#!/usr/bin/env python3
"""Write exact installed pins only after recorded A100 smoke/resume success."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mmmu_eval.utils import atomic_text, file_hash, package_versions, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output", default="requirements.txt")
    args = parser.parse_args()
    root = Path(args.run_dir)
    meta = read_json(root / "run_meta.json")
    evidence = read_json(root / "gpu_resume_verified.json")
    scores = read_json(root / "scores.json")
    if not (meta["status"] == "complete" and meta["limit_per_subject"] == 1
            and scores["total"] == 30 and "A100" in meta["gpu"] and evidence["verified"]):
        raise ValueError("Successful A100 30-question smoke/resume evidence required")
    if evidence["raw_outputs_sha256"] != file_hash(root / "raw_outputs.jsonl"):
        raise ValueError("Resume evidence does not match current raw outputs")
    if evidence["run_fingerprint"] != meta["run_fingerprint"] or evidence["protocol_hash"] != meta["protocol_hash"]:
        raise ValueError("Resume evidence belongs to another evaluation protocol")
    if meta["generation_packages"] != package_versions():
        raise ValueError("Current packages differ from tested GPU environment")
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    # Refuse local paths/editable installs: a portable exact-version lock only.
    lines = [line for line in freeze.splitlines() if line and not line.startswith("#")]
    if any("==" not in line or " @ " in line or line.startswith("-e") for line in lines):
        raise ValueError("Environment contains non-version pins; use a clean venv before freezing")
    atomic_text(args.output, "# A100 smoke and forced kill/resume verified.\n" + "\n".join(lines) + "\n")
    write_json(root / "environment_verified.json", {
        "requirements_sha256": file_hash(args.output), "packages": package_versions(),
        "gpu": meta["gpu"], "protocol_hash": meta["protocol_hash"],
    })
    print(f"Wrote verified exact-version requirements: {args.output}")


if __name__ == "__main__":
    main()
