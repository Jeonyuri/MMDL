"""Final-evaluation sets: MMMU test and MMMU-Pro (standard 10 options / vision).

Question IDs are fixed in an ID file frozen before evaluation:
  configs/final_eval_ids.json      - 450-question samples (15 per subject, 2026-11-19)
  configs/final_eval_ids_full.json - full test splits (MMMU test 10,500; MMMU-Pro 1,730; 2026-11-21)
Prompts:
  - mmmu_test: exactly the validation protocol (mmmu_eval.prompt.build_messages, images first).
  - pro_standard: official MMMU-Pro CoT text "{question}\\n{A. opt ...}\\n{cot standard}".
  - pro_vision: official MMMU-Pro CoT vision instruction only; the question is inside the image.
  For both Pro settings images go first (same content order as our MMMU protocol), all
  non-null image_1..image_7 in index order, and <image n> tags stay in the text (as in
  our MMMU protocol). The official transformers script instead puts text first and
  replaces tags with "[image]"; we keep one image layout across all evaluations.
"""
import ast
from collections import Counter

from mmmu_eval.data import SUBJECTS, normalize_row
from mmmu_eval.prompt import build_messages
from .official_pro import COT_PROMPTS, parse_options

SETS = ("mmmu_test", "pro_standard", "pro_vision")
PRO_CONFIGS = {"pro_standard": "standard (10 options)", "pro_vision": "vision"}
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def block_total(block):
    """Number of questions the ID block defines (450 for the samples, full split size otherwise)."""
    return block["total"] if block.get("per_subject") is None else block["per_subject"] * len(SUBJECTS)


def expected_ids(set_name, ids_spec):
    block = ids_spec["mmmu_test"] if set_name == "mmmu_test" else ids_spec["mmmu_pro"]
    ids = block["ids"]
    if sorted(ids) != sorted(SUBJECTS):
        raise ValueError(f"{set_name}: ID file must list IDs for each of 30 subjects")
    if block.get("per_subject") is not None:
        bad = [s for s, v in ids.items() if len(v) != block["per_subject"]]
    else:
        bad = [s for s, v in ids.items() if not v or len(v) != block["counts"][s]]
    flat = [i for v in ids.values() for i in v]
    if bad or len(flat) != len(set(flat)) or len(flat) != block_total(block):
        raise ValueError(f"{set_name}: ID file counts are inconsistent (subjects {bad[:3]}, total {len(flat)})")
    return {subject: list(ids[subject]) for subject in SUBJECTS}, block


def _pro_options(value):
    options = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(options, list) or not 2 <= len(options) <= len(LETTERS) or not all(isinstance(x, str) for x in options):
        raise ValueError(f"Invalid MMMU-Pro options: {value!r}")
    return [str(x) for x in options]


def normalize_pro_row(row, set_name, subject):
    options = _pro_options(row["options"])
    answer = str(row["answer"])
    if answer not in LETTERS[:len(options)]:
        raise ValueError(f"Invalid gold answer: {row['id']}: {answer}")
    if set_name == "pro_standard":
        images = [row[f"image_{i}"].convert("RGB") for i in range(1, 8) if row.get(f"image_{i}") is not None]
        question = row["question"]
    else:
        images = [row["image"].convert("RGB")]
        question = None
    if not images:
        raise ValueError(f"{row['id']}: no image")
    return {"id": str(row["id"]), "subject": subject, "question": question, "question_type": "multiple-choice",
            "answer": answer, "options": options, "images": images,
            "origin": str(row["id"]).split("_")[0]}   # 'validation' = derived from MMMU validation


def pro_text(row, set_name):
    if set_name == "pro_standard":
        return f"{row['question']}\n{parse_options(row['options'])}\n{COT_PROMPTS['standard']}"
    return COT_PROMPTS["vision"]


def build_set_messages(row, set_name, min_pixels, max_pixels):
    if set_name == "mmmu_test":
        return build_messages(row, min_pixels, max_pixels)
    content = [{"type": "image", "image": image, "min_pixels": min_pixels, "max_pixels": max_pixels}
               for image in row["images"]]
    content.append({"type": "text", "text": pro_text(row, set_name)})
    return [{"role": "user", "content": content}]


def load_set(set_name, ids_spec, data_root, loader=None):
    """Returns ({subject: [Arrow row index...]}, {subject: dataset}, block). Rows stay lazy."""
    if set_name not in SETS:
        raise ValueError(set_name)
    if loader is None:
        from datasets import load_dataset
        loader = load_dataset
    expected, block = expected_ids(set_name, ids_spec)
    datasets, index = {}, {}
    if set_name == "mmmu_test":
        for subject in SUBJECTS:
            ds = loader(block["dataset"], subject, split="test", revision=block["revision"], cache_dir=str(data_root))
            pos = {ident: i for i, ident in enumerate(ds["id"])}
            missing = [i for i in expected[subject] if i not in pos]
            if missing:
                raise ValueError(f"{subject}: IDs not found in MMMU test: {missing[:3]}")
            datasets[subject], index[subject] = ds, [pos[i] for i in expected[subject]]
    else:
        ds = loader(block["dataset"], PRO_CONFIGS[set_name], split="test", revision=block["revision"], cache_dir=str(data_root))
        if len(ds) != 1730:
            raise ValueError(f"{set_name}: expected 1730 rows, got {len(ds)}")
        pos = {ident: i for i, ident in enumerate(ds["id"])}
        subj = ds["subject"]
        for subject in SUBJECTS:
            missing = [i for i in expected[subject] if i not in pos]
            if missing:
                raise ValueError(f"{subject}: IDs not found in MMMU-Pro: {missing[:3]}")
            bad = [i for i in expected[subject] if subj[pos[i]] != subject]
            if bad:
                raise ValueError(f"{subject}: subject mismatch for {bad[:3]}")
            datasets[subject], index[subject] = ds, [pos[i] for i in expected[subject]]
    counts = Counter({s: len(v) for s, v in index.items()})
    if sum(counts.values()) != block_total(block):
        raise ValueError(f"{set_name}: expected {block_total(block)} questions, got {sum(counts.values())}")
    return index, datasets, expected


def get_row(set_name, ds, i, subject):
    row = ds[i]
    if set_name == "mmmu_test":
        out = normalize_row(row, subject)
        out["origin"] = "test"
        return out
    return normalize_pro_row(row, set_name, subject)
