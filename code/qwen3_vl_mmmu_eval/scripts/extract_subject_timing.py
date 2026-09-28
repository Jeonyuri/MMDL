#!/usr/bin/env python3
"""Extract per-subject generation wall time from a run notebook's captured stdout.

generate.py batches each subject's 30 questions into one vLLM call; vLLM's own
tqdm progress bar prints "Processed prompts: 100%|...|[MM:SS<00:00, ...]" right
before generate.py prints "Saved <subject>: 30". run_meta.json only stores the
run TOTAL (generation_seconds), not a per-subject breakdown, so this recovers
it from the notebook's saved cell output instead of any dedicated field.

Usage: python scripts/extract_subject_timing.py notebooks/runs/<run>.ipynb [--out out.json]
"""
import argparse
import json
import re
from pathlib import Path

PROGRESS_RE = re.compile(r"^Processed prompts:.*100%.*\[(\d+):(\d+)(?::(\d+))?<")
SAVED_RE = re.compile(r"^Saved ([A-Za-z_]+): (\d+)$")


def parse_elapsed(line):
    m = PROGRESS_RE.match(line)
    if not m:
        return None
    a, b, c = m.groups()
    return (int(a) * 3600 + int(b) * 60 + int(c)) if c is not None else (int(a) * 60 + int(b))


def extract(notebook_path):
    nb = json.load(open(notebook_path, encoding="utf-8"))
    text = ""
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                text += "".join(out.get("text", []))
    last_progress = None
    rows = []
    for line in text.split("\n"):
        if PROGRESS_RE.match(line):
            last_progress = line
        elif SAVED_RE.match(line):
            subject, n = SAVED_RE.match(line).groups()
            rows.append({"subject": subject, "data_num": int(n),
                        "generation_seconds": parse_elapsed(last_progress) if last_progress else None})
            last_progress = None
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = extract(args.notebook)
    missing = [r["subject"] for r in rows if r["generation_seconds"] is None]
    total = sum(r["generation_seconds"] for r in rows if r["generation_seconds"] is not None)
    print(f"{len(rows)} subjects extracted, {len(missing)} missing timing: {missing}")
    print(f"sum of per-subject vLLM batch time: {total}s ({total/60:.1f} min)")
    print("note: excludes one-time engine startup / between-subject overhead "
          "(compare against run_meta.json's generation_seconds)")
    out_path = Path(args.out) if args.out else Path(args.notebook).with_name("subject_timing.json")
    json.dump({"rows": rows, "sum_generation_seconds": total,
               "source_notebook": str(args.notebook),
               "method": "parsed from vLLM tqdm 'Processed prompts' lines preceding each "
                         "'Saved <subject>: N' line in the notebook's captured stdout"},
              open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
