#!/usr/bin/env python3
"""Print the report's §1 numbers and §5 per-subject table for one run (CPU only).

Usage: PYTHONPATH=src python scripts/make_report_table.py outputs/final_seed3407_16k
Reads scored.jsonl/run_meta.json; recomputes both the original Qwen-rule parser and the v2 fallback parser.
"""
import json
import sys
from collections import Counter
from pathlib import Path

from mmmu_eval.parse import DEFAULT_REGEX, extract_answer
from mmmu_eval.parse_v2 import extract_answer_v2


def ok(parsed, gold):
    return parsed["method"] not in ("unparsed", "refused") and parsed["extracted"] == gold


def main(run_dir):
    run = Path(run_dir)
    rows = [json.loads(line) for line in open(run / "scored.jsonl", encoding="utf-8")]
    meta = json.load(open(run / "run_meta.json", encoding="utf-8"))
    cfg = json.load(open(run / "resolved_config.json", encoding="utf-8"))
    regex = cfg.get("parser", {}).get("regex", DEFAULT_REGEX)
    for r in rows:
        c, g = r["scoring_choices"], r["scoring_answer"]
        r["orig_ok"] = ok(extract_answer(r["response"], dict(c), regex), g)
        r["v2_parsed"] = extract_answer_v2(r["response"], dict(c), regex, r["question_type"], r["truncated"])
        r["v2_ok"] = ok(r["v2_parsed"], g)
    subjects = cfg["subjects"]
    print(f"| No. | Subject | Data Num | Acc (v2) | Acc (Qwen 원본 규칙) |\n|---|---|---|---|---|")
    v2s, os_ = [], []
    for i, s in enumerate(subjects, 1):
        sub = [r for r in rows if r["subject"] == s]
        a2, a0 = 100 * sum(r["v2_ok"] for r in sub) / len(sub), 100 * sum(r["orig_ok"] for r in sub) / len(sub)
        v2s.append(a2); os_.append(a0)
        print(f"| {i} | {s} | {len(sub)} | {a2:.2f} | {a0:.2f} |")
    print(f"| | **Overall (macro avg)** | **{len(rows)}** | **{sum(v2s) / len(v2s):.2f}** | **{sum(os_) / len(os_):.2f}** |")
    n = len(rows)
    print("\n--- §1/§7 numbers ---")
    print(f"GPU: {meta.get('gpu')} | peak_vram_mib(device total, vLLM preallocated): {meta.get('peak_vram_mib')} "
          f"| generation minutes: {meta.get('generation_seconds', 0) / 60:.0f}")
    print(f"packages: {meta.get('packages')}")
    print(f"truncated: {sum(r['truncated'] for r in rows)} ({100 * sum(r['truncated'] for r in rows) / n:.1f}%)")
    print(f"unparsed orig: {sum(not r['orig_ok'] and r['method'] == 'unparsed' for r in rows)} | "
          f"unparsed v2: {sum(r['v2_parsed']['method'] == 'unparsed' for r in rows)} | "
          f"v2 methods: {dict(Counter(r['v2_parsed']['method'] for r in rows))}")
    print(f"mean output tokens: {sum(r['output_tokens'] for r in rows) / n:.0f}")
    print(f"mean acc mc/open (v2): "
          f"{100 * sum(r['v2_ok'] for r in rows if r['question_type'] == 'multiple-choice') / sum(r['question_type'] == 'multiple-choice' for r in rows):.2f} / "
          f"{100 * sum(r['v2_ok'] for r in rows if r['question_type'] == 'open') / sum(r['question_type'] == 'open' for r in rows):.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
