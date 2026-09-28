"""Markdown report without GPU imports or dataset downloads."""
from pathlib import Path

from .utils import atomic_text, file_hash, read_json, write_json


def report(config, args, meta):
    output_dir = Path(args.output_dir)
    scores_path = output_dir / "scores.json"
    if not scores_path.exists() or file_hash(scores_path) != meta.get("scores_sha256"):
        raise ValueError("Missing or modified scores.json; run --stage score first")
    scores = read_json(scores_path)
    lines = []
    if scores["partial"]:
        lines += ["> **PARTIAL RUN** — debug subset; not a full MMMU-val score.", ""]
    lines += ["| No. | Subject | Data Num | Acc |", "|---|---|---|---|"]
    for index, item in enumerate(scores["subjects"], 1):
        lines.append(f"| {index} | {item['subject']} | {item['data_num']} | {100 * item['accuracy']:.2f} |")
    ours = 100 * scores["overall_macro"]
    reference = config["official_reference_percent"]
    lines += [f"| | **Overall (macro avg)** | **{scores['total']}** | **{ours:.2f}** |", "",
              "Overall = mean(30개 과목 accuracy)", "",
              "| 공식 | 우리 결과 | Δ (percentage points) |", "|---|---|---|",
              f"| {reference:.1f} | {ours:.2f} | {ours - reference:+.2f} |", "",
              "공식 수치는 참고용입니다. judge 제거 등 평가 절차 차이가 있으므로 동일 프로토콜 점수로 해석하지 않습니다.",
              f"주관식 채점: `{config['open_question_mode']}`. 잘림 비율: {100 * scores['length_finish_ratio']:.2f}%.",
              f"Protocol SHA256: `{meta['protocol_hash']}`", ""]
    atomic_text(output_dir / "results_table.md", "\n".join(lines))
    meta["status"] = "complete"
    write_json(output_dir / "run_meta.json", meta)
    print(f"Report: {output_dir / 'results_table.md'}", flush=True)
