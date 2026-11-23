"""CPU scoring for the final-evaluation sets.

mmmu_test: identical to the validation protocol (Qwen rule parser + v2 conservative fallback;
open questions scored as qwen_mc {"A": gold, "B": "Other Answers"}).

MMMU-Pro (pre-registered 2026-11-19):
  primary   = official MMMU-Pro parser (official_pro.parse_multi_choice_response) with
              no random guess; truncated responses count as wrong (same rule as v2).
  reference = exact official behaviour: random.Random(42) guess when nothing parses,
              applied in ID-file order, truncation ignored. Only for comparison with
              published numbers; not used for any decision.
"""
from collections import Counter
import random
import statistics

from mmmu_eval.parse import DEFAULT_REGEX
from mmmu_eval.parse_v2 import extract_answer_v2
from .official_pro import get_multi_choice_info, parse_multi_choice_response


def score_rows(set_name, rows, regex=DEFAULT_REGEX):
    scored = []
    guess_rng = random.Random(42)
    for row in rows:
        if set_name == "mmmu_test":
            if row["question_type"] == "open":
                choices, gold = {"A": row["answer"], "B": "Other Answers"}, "A"
            else:
                choices, gold = dict(zip("ABCDEFGHI", row["options"])), row["answer"]
            parsed = extract_answer_v2(row["response"], dict(choices), regex, row["question_type"], row["truncated"])
            correct = parsed["method"] not in ("unparsed", "refused") and parsed["extracted"] == gold
            scored.append(row | {"extracted": parsed["extracted"], "method": parsed["method"], "correct": correct,
                                 "scoring_answer": gold, "scoring_choices": choices})
        else:
            index2ans, all_choices = get_multi_choice_info(row["options"])
            if row["truncated"]:
                pred, method = None, "truncated"
            else:
                pred = parse_multi_choice_response(row["response"], all_choices, index2ans, rng=None)
                method = "official" if pred is not None else "unparsed"
            official = parse_multi_choice_response(row["response"], all_choices, index2ans, rng=guess_rng)
            scored.append(row | {"extracted": pred, "method": method, "correct": pred == row["answer"],
                                 "official_with_guess": official, "correct_official_with_guess": official == row["answer"]})
    return scored


def summarize(set_name, scored, subjects):
    n = len(scored)
    toks = [r["output_tokens"] for r in scored]
    out = {"set": set_name, "n": n, "correct": sum(r["correct"] for r in scored),
           "accuracy": sum(r["correct"] for r in scored) / n,
           "truncated": sum(r["truncated"] for r in scored), "trunc_pct": 100 * sum(r["truncated"] for r in scored) / n,
           "mean_output_tokens": statistics.mean(toks), "median_output_tokens": statistics.median(toks),
           "methods": dict(Counter(r["method"] for r in scored)),
           "per_subject": {s: sum(r["correct"] for r in scored if r["subject"] == s) for s in subjects}}
    if set_name == "mmmu_test":
        out["question_type"] = {k: {"n": sum(r["question_type"] == k for r in scored),
                                    "correct": sum(r["correct"] for r in scored if r["question_type"] == k)}
                                for k in ("multiple-choice", "open")}
    else:
        out["reference_official_with_guess"] = {"correct": sum(r["correct_official_with_guess"] for r in scored),
                                                "accuracy": sum(r["correct_official_with_guess"] for r in scored) / n}
        out["by_origin"] = {o: {"n": sum(r["origin"] == o for r in scored),
                                "correct": sum(r["correct"] for r in scored if r["origin"] == o)}
                            for o in sorted({r["origin"] for r in scored})}
    return out
