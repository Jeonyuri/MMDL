"""Deterministic CPU-only scoring of complete saved runs."""
from collections import Counter
import math
from pathlib import Path

from .parse import extract_answer
from .utils import file_hash, read_jsonl, write_json, write_jsonl


def validate_rows(rows, meta):
    expected = meta["expected_ids"]
    by_subject = {subject: {} for subject in expected}
    for row in rows:
        subject, ident = row["subject"], row["id"]
        if subject not in expected or ident not in expected[subject]:
            raise ValueError(f"Unexpected question: {subject}/{ident}")
        if ident in by_subject[subject]:
            raise ValueError(f"Duplicate question: {ident}")
        if row["question_type"] not in ("open", "multiple-choice"):
            raise ValueError(f"Invalid question type: {ident}")
        if row["finish_reason"] not in ("stop", "length"):
            raise ValueError(f"Invalid finish reason: {ident}")
        if not isinstance(row["response"], str) or row["output_tokens"] < 0:
            raise ValueError(f"Invalid response: {ident}")
        by_subject[subject][ident] = row
    for subject, ids in expected.items():
        if set(by_subject[subject]) != set(ids):
            raise ValueError(f"{subject}: missing outputs; expected {len(ids)}, got {len(by_subject[subject])}")
        expected_count = meta["limit_per_subject"] or 30
        if len(ids) != expected_count:
            raise ValueError(f"{subject}: invalid manifest count")
    return [by_subject[subject][ident] for subject, ids in expected.items() for ident in ids]


def score(config, args, meta):
    output_dir = Path(args.output_dir)
    raw = output_dir / "raw_outputs.jsonl"
    if not raw.exists():
        raise ValueError("Missing raw_outputs.jsonl; generate first")
    if not meta.get("raw_outputs_sha256") or file_hash(raw) != meta["raw_outputs_sha256"]:
        raise ValueError("Raw output checksum missing or mismatched; finish/resume generation first")
    rows = validate_rows(read_jsonl(raw), meta)
    scored = []
    for row in rows:
        if row["question_type"] == "open":
            choices, gold = {"A": row["answer"], "B": "Other Answers"}, "A"
        else:
            choices = dict(zip("ABCDEFGHI", row["options"]))
            gold = row["answer"]
            if not choices or gold not in choices:
                raise ValueError(f"Invalid options/answer: {row['id']}")
        parsed = extract_answer(row["response"], choices, config["parser"]["regex"])
        correct = parsed["method"] not in ("unparsed", "refused") and parsed["extracted"] == gold
        scored.append(row | parsed | {"correct": correct, "scoring_answer": gold,
                                      "scoring_choices": choices, "open_question_mode": config["open_question_mode"]})
    subjects = []
    for subject in config["subjects"]:
        subset = [row for row in scored if row["subject"] == subject]
        correct = sum(row["correct"] for row in subset)
        subjects.append({"subject": subject, "data_num": len(subset), "correct": correct,
                         "accuracy": correct / len(subset),
                         "mean_output_tokens": sum(row["output_tokens"] for row in subset) / len(subset)})
    macro = sum(row["accuracy"] for row in subjects) / len(subjects)
    micro = sum(row["correct"] for row in scored) / len(scored)
    assert math.isclose(macro, micro, rel_tol=0, abs_tol=1e-12), "Macro/micro mismatch"
    types = {}
    for kind in ("multiple-choice", "open"):
        subset = [row for row in scored if row["question_type"] == kind]
        correct = sum(row["correct"] for row in subset)
        types[kind] = {"count": len(subset), "correct": correct,
                       "accuracy": correct / len(subset) if subset else None}
    scores = {"subjects": subjects, "overall_macro": macro, "overall_micro": micro,
              "total": len(scored), "question_type": types,
              "method_counts": dict(Counter(row["method"] for row in scored)),
              "length_finish_ratio": sum(row["finish_reason"] == "length" for row in scored) / len(scored),
              "open_question_mode": config["open_question_mode"], "partial": meta["partial"],
              "protocol_hash": meta["protocol_hash"], "raw_outputs_sha256": file_hash(raw)}
    write_jsonl(output_dir / "scored.jsonl", scored)
    write_json(output_dir / "scores.json", scores)
    meta["scores_sha256"] = file_hash(output_dir / "scores.json")
    meta["status"] = "scored"
    write_json(output_dir / "run_meta.json", meta)
    print(f"Scored {len(scored)} questions: macro={100 * macro:.2f}%", flush=True)
    return scores
