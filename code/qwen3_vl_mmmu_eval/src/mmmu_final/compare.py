"""Compare baseline vs submission on the final-evaluation sets (CPU only).

Reads <output_root>/<tag>/<set>/scored.jsonl for both tags, checks both runs used the same
GPU type, seed, ID file and settings, and writes final_comparison.json / .md.
Interpretation (pre-registered 2026-11-19): descriptive only; no model is chosen here.
One run's net-change SD is about +/-7.5 questions on 450 items (scales with sqrt(n)),
so |net| < 2 SD is "no demonstrated difference".
Usage: PYTHONPATH=src python -m mmmu_final.compare --output_root <dir> [--baseline baseline --submission submission]
"""
import argparse
import json
from math import comb
from pathlib import Path

from mmmu_eval.utils import read_jsonl, write_json
from .sets import SETS

NAMES = {"mmmu_test": "MMMU test", "pro_standard": "MMMU-Pro standard(10)", "pro_vision": "MMMU-Pro vision"}


def mcnemar(a, b):
    gain = sum(1 for i in a if not a[i] and b[i]); loss = sum(1 for i in a if a[i] and not b[i])
    n, k = gain + loss, min(gain, loss)
    return gain, loss, (min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0)


def compare(output_root, base_tag="baseline", sub_tag="submission"):
    root = Path(output_root); report = {"sets": {}}
    for s in SETS:
        pa, pb = root / base_tag / s, root / sub_tag / s
        if not ((pa / "scores.json").is_file() and (pb / "scores.json").is_file()):
            report["sets"][s] = {"status": "missing"}; continue
        A, B = json.loads((pa / "scores.json").read_text()), json.loads((pb / "scores.json").read_text())
        for k in ("gpu", "seed", "ids_sha256"):
            if A[k] != B[k]:
                raise ValueError(f"{s}: {k} differs between runs ({A[k]} vs {B[k]}); not comparable")
        ra = {r["id"]: r for r in read_jsonl(pa / "scored.jsonl")}; rb = {r["id"]: r for r in read_jsonl(pb / "scored.jsonl")}
        assert set(ra) == set(rb) and len(ra) == A["n"] == B["n"]
        g, l, p = mcnemar({i: ra[i]["correct"] for i in ra}, {i: rb[i]["correct"] for i in ra})
        net = B["correct"] - A["correct"]
        two_sd = round(15 * (A["n"] / 450) ** 0.5)
        entry = {"name": f"{NAMES[s]} {A['n']}", "n": A["n"], "gpu": A["gpu"], "baseline": A["correct"], "submission": B["correct"],
                 "baseline_acc": A["accuracy"], "submission_acc": B["accuracy"], "net": net, "gain": g, "loss": l, "mcnemar_p": p,
                 "trunc_pct": {"baseline": A["trunc_pct"], "submission": B["trunc_pct"]},
                 "median_tokens": {"baseline": A["median_output_tokens"], "submission": B["median_output_tokens"]},
                 "interpretation": (f"no demonstrated difference (|net| < {two_sd}, ~2 SD)" if abs(net) < two_sd else
                                    ("submission higher beyond ~2 SD" if net > 0 else "submission lower beyond ~2 SD"))}
        if "reference_official_with_guess" in A:
            entry["reference_official_with_guess"] = {"baseline": A["reference_official_with_guess"]["correct"],
                                                      "submission": B["reference_official_with_guess"]["correct"]}
            entry["by_origin"] = {"baseline": A["by_origin"], "submission": B["by_origin"]}
        report["sets"][s] = entry
    lines = ["| set | GPU | baseline | submission | net | +/− | p | trunc% (base→sub) |", "|---|---|---|---|---|---|---|---|"]
    for s, e in report["sets"].items():
        if e.get("status") == "missing":
            lines.append(f"| {NAMES[s]} | — | — | — | — | — | — | — |"); continue
        lines.append(f"| {e['name']} | {e['gpu'].replace('NVIDIA ', '')} | {e['baseline']} ({100*e['baseline_acc']:.2f}%) | "
                     f"{e['submission']} ({100*e['submission_acc']:.2f}%) | {e['net']:+d} | +{e['gain']}/−{e['loss']} | {e['mcnemar_p']:.3f} | "
                     f"{e['trunc_pct']['baseline']:.1f}→{e['trunc_pct']['submission']:.1f} |")
    md = "\n".join(lines)
    write_json(root / "final_comparison.json", report)
    (root / "final_comparison.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--submission", default="submission")
    a = ap.parse_args()
    compare(a.output_root, a.baseline, a.submission)
