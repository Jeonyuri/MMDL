"""Pinned per-config loading; never filter the full validation dataset."""
import ast
from collections import Counter

SUBJECTS = "Accounting Agriculture Architecture_and_Engineering Art Art_Theory Basic_Medical_Science Biology Chemistry Clinical_Medicine Computer_Science Design Diagnostics_and_Laboratory_Medicine Economics Electronics Energy_and_Power Finance Geography History Literature Manage Marketing Materials Math Mechanical_Engineering Music Pharmacy Physics Psychology Public_Health Sociology".split()


def parse_options(value):
    options = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(options, list) or len(options) > 9 or not all(isinstance(x, str) for x in options):
        raise ValueError(f"Invalid options: {value!r}")
    return options


def normalize_row(row, subject):
    options = parse_options(row["options"])
    kind = row["question_type"]
    if kind not in ("multiple-choice", "open"):
        raise ValueError(f"Unknown question_type: {kind}")
    if (kind == "open" and options) or (kind == "multiple-choice" and not options):
        raise ValueError(f"Options/question_type mismatch: {row['id']}")
    answer = str(row["answer"])
    if kind == "multiple-choice" and answer not in "ABCDEFGHI"[:len(options)]:
        raise ValueError(f"Invalid gold answer: {row['id']}: {answer}")
    return {"id": str(row["id"]), "subject": subject, "question": row["question"],
            "question_type": kind, "answer": answer, "options": options,
            "images": [row[f"image_{i}"].convert("RGB") for i in range(1, 8)
                       if row.get(f"image_{i}") is not None]}


def load_data(config, data_root, loader=None):
    if loader is None:
        from datasets import load_dataset
        loader = load_dataset
    datasets, counts, kinds, seen = {}, {}, Counter(), set()
    for subject in config["subjects"]:
        ds = loader(config["dataset_path"], subject, split=config["split"],
                    revision=config["dataset_revision"], cache_dir=str(data_root))
        if len(ds) != 30:
            raise ValueError(f"{subject}: expected 30 questions, got {len(ds)}")
        # Keep Arrow-backed rows lazy: decoding all 900 PIL images wastes Colab RAM.
        datasets[subject] = ds
        counts[subject] = len(ds)
        for ident, kind in zip(ds["id"], ds["question_type"]):
            if ident in seen:
                raise ValueError(f"Duplicate dataset ID: {ident}")
            if kind not in ("multiple-choice", "open"):
                raise ValueError(f"Unknown question_type: {kind}")
            seen.add(ident)
            kinds[kind] += 1
        print(f"{subject}: {len(ds)}", flush=True)
    if sum(counts.values()) != 900:
        raise ValueError(f"Expected 900 questions: {counts}")
    stats = {"per_subject": counts, "total": sum(counts.values()), "question_type": dict(kinds)}
    print(f"Dataset counts: {stats}", flush=True)
    return datasets, stats
