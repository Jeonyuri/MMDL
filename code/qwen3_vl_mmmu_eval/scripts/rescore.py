#!/usr/bin/env python3
"""Re-score saved runs with the original Qwen-rule parser and the v2 fallback parser.

CPU only; reads outputs/<run>/scored.jsonl (response, truncated, scoring_* fields) and
writes summaries under --out-dir. Original outputs are never modified.
Usage: PYTHONPATH=src python scripts/rescore.py [--outputs-dir outputs] [--out-dir reports/rescore]
"""
import argparse
import json
import random
from collections import Counter
from math import comb
from pathlib import Path

from mmmu_eval.parse import DEFAULT_REGEX, extract_answer
from mmmu_eval.parse_v2 import extract_answer_v2, fallback_letter

RUNS = [
    "baseline_seed42_4k", "baseline_seed42_8k", "baseline_seed42_16k",
    "baseline_seed42_high_max", "baseline_seed42_high_min",
    "baseline_seed3407_baseline", "baseline_seed3407_prompt",
    "baseline_seed3407_detailed_prompt", "baseline_seed3407_presence0",
    "final_seed3407_16k",  # skipped automatically if the folder does not exist yet
]
FINAL_RUN = "final_seed3407_16k"
# audit sample sizes: kept separate so the final run always gets its own dedicated
# sample regardless of how big its fallback pool is relative to the other 9 runs
# (a single pooled draw across all runs could under-sample it by chance).
PRE_RECOVERED_SAMPLE = 30
FINAL_RECOVERED_SAMPLE = 8
TRUNCATED_EXCLUDED_SAMPLE = 10
PAIRS = [
    ("baseline_seed42_8k", "baseline_seed42_4k"), ("baseline_seed42_8k", "baseline_seed42_16k"),
    ("baseline_seed42_4k", "baseline_seed42_16k"),
    ("baseline_seed42_8k", "baseline_seed42_high_max"), ("baseline_seed42_8k", "baseline_seed42_high_min"),
    ("baseline_seed42_8k", "baseline_seed3407_baseline"),
    ("baseline_seed3407_baseline", "baseline_seed3407_prompt"),
    ("baseline_seed3407_baseline", "baseline_seed3407_detailed_prompt"),
    ("baseline_seed3407_baseline", "baseline_seed3407_presence0"),
    ("baseline_seed3407_prompt", "baseline_seed3407_detailed_prompt"),
    ("baseline_seed3407_baseline", "final_seed3407_16k"),
    ("baseline_seed42_16k", "final_seed3407_16k"),
]


def is_correct(parsed, gold):
    return parsed["method"] not in ("unparsed", "refused") and parsed["extracted"] == gold


def load(outputs_dir, run):
    rows = [json.loads(line) for line in open(outputs_dir / run / "scored.jsonl", encoding="utf-8")]
    cfg = json.load(open(outputs_dir / run / "resolved_config.json", encoding="utf-8"))
    regex = cfg["parser"]["regex"] if "parser" in cfg else DEFAULT_REGEX
    out = {}
    for row in rows:
        choices, gold = row["scoring_choices"], row["scoring_answer"]
        orig = extract_answer(row["response"], dict(choices), regex)
        v2 = extract_answer_v2(row["response"], dict(choices), regex, row["question_type"], row["truncated"])
        out[row["id"]] = {
            "row": row, "orig": orig, "v2": v2,
            "orig_ok": is_correct(orig, gold), "v2_ok": is_correct(v2, gold),
            "stored_ok": row["correct"],
        }
    return out


def mcnemar(a, b, key):
    gain = sum((not a[i][key]) and b[i][key] for i in a)
    loss = sum(a[i][key] and (not b[i][key]) for i in a)
    n, k = gain + loss, min(gain, loss)
    p = min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
    return gain, loss, p


def pct(x, n):
    return 100 * x / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--out-dir", default="reports/rescore")
    ap.add_argument("--audit-seed", type=int, default=0)
    ap.add_argument("--runs", nargs="*", default=None, help="run folder names (default: the built-in list, existing ones only)")
    args = ap.parse_args()
    outputs_dir, out_dir = Path(args.outputs_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [r for r in (args.runs or RUNS) if (outputs_dir / r / "scored.jsonl").exists()]
    print("runs:", runs)
    data, summary = {}, {}
    for run in runs:
        d = data[run] = load(outputs_dir, run)
        n = len(d)
        mismatch = sum(v["orig_ok"] != v["stored_ok"] for v in d.values())
        trunc = [v for v in d.values() if v["row"]["truncated"]]
        summary[run] = {
            "n": n, "sanity_mismatch_vs_stored": mismatch,
            "acc_orig": pct(sum(v["orig_ok"] for v in d.values()), n),
            "acc_v2": pct(sum(v["v2_ok"] for v in d.values()), n),
            "trunc_pct": pct(len(trunc), n),
            "unparsed_orig_pct": pct(sum(v["orig"]["method"] == "unparsed" for v in d.values()), n),
            "unparsed_v2_pct": pct(sum(v["v2"]["method"] == "unparsed" for v in d.values()), n),
            "unparsed_v2_notrunc": sum(v["v2"]["method"] == "unparsed" and not v["row"]["truncated"] for v in d.values()),
            "unparsed_v2_notrunc_open": sum(v["v2"]["method"] == "unparsed" and not v["row"]["truncated"]
                                            and v["row"]["question_type"] == "open" for v in d.values()),
            "fallback_used": sum(v["v2"]["method"] == "fallback" for v in d.values()),
            "fallback_correct": sum(v["v2"]["method"] == "fallback" and v["v2_ok"] for v in d.values()),
            "mean_output_tokens": sum(v["row"]["output_tokens"] for v in d.values()) / n,
            "methods_v2": dict(Counter(v["v2"]["method"] for v in d.values())),
        }
        s = summary[run]
        s["acc_v2_open"] = pct(sum(v["v2_ok"] for v in d.values() if v["row"]["question_type"] == "open"),
                               sum(v["row"]["question_type"] == "open" for v in d.values()))
        s["acc_v2_mc"] = pct(sum(v["v2_ok"] for v in d.values() if v["row"]["question_type"] == "multiple-choice"),
                             sum(v["row"]["question_type"] == "multiple-choice" for v in d.values()))
        assert mismatch == 0, f"{run}: recomputed original parser differs from stored scores"

    tests = []
    for a, b in PAIRS:
        if a not in data or b not in data:
            continue
        for key in ("orig_ok", "v2_ok"):
            g, l, p = mcnemar(data[a], data[b], key)
            tests.append({"from": a, "to": b, "parser": key[:-3], "gain": g, "loss": l, "p": p})

    # audit sample: recovered (non-truncated) fallback answers + truncated responses the patterns would have matched.
    # "recovered" is split into pre-experiment runs vs the final run and sampled separately (see PRE/FINAL_RECOVERED_SAMPLE
    # above) so the final run's manual-review count in the report is always exactly reproducible from this file, not a
    # side effect of pool-size weighting in one pooled random draw.
    rng = random.Random(args.audit_seed)
    recovered_pre, recovered_final, trunc_hits = [], [], []
    for run, d in data.items():
        for ident, v in d.items():
            row = v["row"]
            if v["v2"]["method"] == "fallback":
                (recovered_final if run == FINAL_RUN else recovered_pre).append((run, ident))
            elif (row["truncated"] and v["orig"]["method"] == "unparsed" and row["question_type"] == "multiple-choice"
                  and fallback_letter(row["response"], row["scoring_choices"])):
                trunc_hits.append((run, ident))
    audit = []
    groups = [("recovered", recovered_pre, PRE_RECOVERED_SAMPLE), ("recovered", recovered_final, FINAL_RECOVERED_SAMPLE),
              ("truncated_excluded", trunc_hits, TRUNCATED_EXCLUDED_SAMPLE)]
    for label, pool, k in groups:
        for run, ident in rng.sample(pool, min(k, len(pool))):
            v = data[run][ident]
            audit.append({"kind": label, "run": run, "id": ident, "gold": v["row"]["scoring_answer"],
                          "v2_extracted": v["v2"]["extracted"], "v2_correct": v["v2_ok"],
                          "truncated": v["row"]["truncated"], "tail": v["row"]["response"][-400:]})
    with open(out_dir / "audit_sample.jsonl", "w", encoding="utf-8") as f:
        for a in audit:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    recovered = recovered_pre + recovered_final
    json.dump({"runs": summary, "mcnemar": tests, "recovered_total": len(recovered),
               "recovered_pre_total": len(recovered_pre), "recovered_final_total": len(recovered_final),
               "recovered_pre_sampled": min(PRE_RECOVERED_SAMPLE, len(recovered_pre)),
               "recovered_final_sampled": min(FINAL_RECOVERED_SAMPLE, len(recovered_final)),
               "truncated_excluded_total": len(trunc_hits)},
              open(out_dir / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    lines = ["# Rescore summary (orig = Qwen rule parser, v2 = orig + conservative fallback)", "",
             "| run | acc orig | acc v2 | Δ | trunc% | unparsed% orig | unparsed% v2 | fallback used/correct | v2 unparsed (not truncated) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for run, s in summary.items():
        lines.append(f"| {run} | {s['acc_orig']:.2f} | {s['acc_v2']:.2f} | {s['acc_v2'] - s['acc_orig']:+.2f} | "
                     f"{s['trunc_pct']:.1f} | {s['unparsed_orig_pct']:.1f} | {s['unparsed_v2_pct']:.1f} | "
                     f"{s['fallback_used']}/{s['fallback_correct']} | {s['unparsed_v2_notrunc']} "
                     f"(open {s['unparsed_v2_notrunc_open']}) |")
    lines += ["", "## McNemar (paired, exact two-sided)", "",
              "| from → to | parser | gain | loss | p |", "|---|---|---|---|---|"]
    for t in tests:
        lines.append(f"| {t['from']} → {t['to']} | {t['parser']} | {t['gain']} | {t['loss']} | {t['p']:.3g} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out_dir}/summary.md, summary.json, audit_sample.jsonl "
          f"(recovered pool: pre-experiments {len(recovered_pre)} (sampled {min(PRE_RECOVERED_SAMPLE, len(recovered_pre))}), "
          f"final run {len(recovered_final)} (sampled {min(FINAL_RECOVERED_SAMPLE, len(recovered_final))}), "
          f"truncated-excluded pool {len(trunc_hits)} (sampled {min(TRUNCATED_EXCLUDED_SAMPLE, len(trunc_hits))}))")


if __name__ == "__main__":
    main()
