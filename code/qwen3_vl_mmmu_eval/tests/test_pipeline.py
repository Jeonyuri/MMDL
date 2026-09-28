from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from mmmu_eval.cli import DEFAULT_CONFIG, parser, resolve_config, run
from mmmu_eval.data import SUBJECTS
from mmmu_eval.generate import count_input_tokens, preflight, recover_subjects, prepare_inputs_for_vllm
from mmmu_eval.report import report
from mmmu_eval.score import score
from mmmu_eval.utils import file_hash, read_json, read_jsonl, write_json, write_jsonl
from test_data import FakeDataset


@pytest.fixture
def config():
    return yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def fixture_run(tmp_path, config, n=1):
    rows, expected = [], {}
    for subject in SUBJECTS:
        expected[subject] = [f"{subject}_{i}" for i in range(n)]
        for ident in expected[subject]:
            rows.append({"id": ident, "subject": subject, "question_type": "open", "answer": "42",
                         "options": [], "response": "42", "output_tokens": 1, "finish_reason": "stop"})
    raw = tmp_path / "raw_outputs.jsonl"
    write_jsonl(raw, rows)
    meta = {"expected_ids": expected, "limit_per_subject": n if n != 30 else None,
            "partial": n != 30, "protocol_hash": "fixture", "raw_outputs_sha256": file_hash(raw)}
    return SimpleNamespace(output_dir=str(tmp_path)), meta


def test_cpu_score_and_report(tmp_path, config):
    args, meta = fixture_run(tmp_path, config)
    scores = score(config, args, meta)
    assert scores["total"] == 30 and scores["overall_macro"] == scores["overall_micro"] == 1
    assert scores["method_counts"] == {"text": 30}
    assert read_jsonl(tmp_path / "scored.jsonl")[0]["scoring_choices"] == {"A": "42", "B": "Other Answers"}
    report(config, args, meta)
    table = (tmp_path / "results_table.md").read_text(encoding="utf-8")
    assert "PARTIAL RUN" in table and "| 30 | Sociology | 1 | 100.00 |" in table
    assert "**30** | **100.00**" in table


def test_full_900_and_unparsed(tmp_path, config):
    args, meta = fixture_run(tmp_path, config, 30)
    rows = read_jsonl(tmp_path / "raw_outputs.jsonl")
    rows[0].update(response="unknown", finish_reason="length", output_tokens=2048)
    write_jsonl(tmp_path / "raw_outputs.jsonl", rows)
    meta["raw_outputs_sha256"] = file_hash(tmp_path / "raw_outputs.jsonl")
    scores = score(config, args, meta)
    assert scores["overall_micro"] == 899 / 900
    assert scores["method_counts"]["unparsed"] == 1
    assert scores["length_finish_ratio"] == 1 / 900


@pytest.mark.parametrize("mode", ["missing", "duplicate", "unknown"])
def test_bad_raw_fails(tmp_path, config, mode):
    args, meta = fixture_run(tmp_path, config)
    rows = read_jsonl(tmp_path / "raw_outputs.jsonl")
    if mode == "missing":
        rows.pop()
    elif mode == "duplicate":
        rows.append(rows[0])
    else:
        rows[0]["id"] = "wrong"
    write_jsonl(tmp_path / "raw_outputs.jsonl", rows)
    meta["raw_outputs_sha256"] = file_hash(tmp_path / "raw_outputs.jsonl")
    with pytest.raises(ValueError):
        score(config, args, meta)


def test_recovery_partial_and_torn_tail(tmp_path):
    path = tmp_path / "raw_outputs.jsonl"
    write_jsonl(path, [{"id": "a1", "subject": "A"}, {"id": "a2", "subject": "A"},
                       {"id": "b1", "subject": "B"}])
    with path.open("ab") as handle:
        handle.write(b'{"id": "b2')
    with pytest.warns(UserWarning):
        complete = recover_subjects(path, {"A": ["a1", "a2"], "B": ["b1", "b2"]})
    assert complete == {"A"}
    assert [row["id"] for row in read_jsonl(path)] == ["a1", "a2"]


def test_midfile_corruption_fails(tmp_path):
    path = tmp_path / "raw_outputs.jsonl"
    path.write_text('{invalid}\n{"id":"a","subject":"A"}\n')
    with pytest.raises(ValueError, match="Corrupt"):
        recover_subjects(path, {"A": ["a"]})


def test_preflight_checks_entire_dataset_before_failure(tmp_path, config, monkeypatch):
    import mmmu_eval.generate as module
    datasets = {subject: FakeDataset(subject) for subject in SUBJECTS}
    calls = []
    monkeypatch.setattr(module, "prepare", lambda row, *args: {"prompt": row["id"]})
    def counter(request, processor):
        calls.append(request["prompt"])
        return 8000 if request["prompt"] == "Accounting_0" else 100
    monkeypatch.setattr(module, "count_input_tokens", counter)
    with pytest.raises(ValueError, match="Increase --max_model_len"):
        preflight(datasets, config, None, tmp_path)
    assert len(calls) == 900
    overflow = read_json(tmp_path / "overflow_report.json")
    assert len(overflow) == 1 and overflow[0]["input_tokens"] == 8000
    assert not (tmp_path / "raw_outputs.jsonl").exists()


def test_vision_preparation_and_expanded_token_count():
    class Processor:
        image_processor = SimpleNamespace(patch_size=16)
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return "<image>Q"
        def __call__(self, **kwargs):
            assert kwargs == {"text": ["<image>Q"], "images": ["image"], "fps": []}
            return {"input_ids": [list(range(1234))]}
    def vision(messages, **kwargs):
        assert kwargs == {"image_patch_size": 16, "return_video_kwargs": True, "return_video_metadata": True}
        return ["image"], None, {"fps": []}
    processor = Processor()
    request = prepare_inputs_for_vllm([], processor, vision)
    assert count_input_tokens(request, processor) == 1234


def test_config_defaults_and_overrides(config):
    args = parser().parse_args(["--output_dir", "out", "--seed", "5", "--limit_mm_per_prompt", '{"image":9}'])
    actual = resolve_config(args)
    assert actual["seed"] == 5 and actual["limit_mm_per_prompt"] == {"image": 9}
    assert config["max_model_len"] == 9048 and config["dtype"] == "bfloat16"


def test_local_revision_ignored(tmp_path):
    args = parser().parse_args(["--output_dir", "out", "--model_path", str(tmp_path)])
    with pytest.warns(UserWarning, match="ignoring"):
        assert resolve_config(args)["model_revision"] is None


def test_score_config_without_local_checkpoint(config, tmp_path):
    config["model_path"] = str(tmp_path / "checkpoint_not_present")
    config["model_revision"] = None
    old = {"config": config, "local_model_files": {"model.safetensors": "test-hash"}, "limit_per_subject": None}
    args = parser().parse_args(["--output_dir", "out", "--stage", "score"])
    assert resolve_config(args, old) == config
    args.stage = "generate"
    with pytest.raises(ValueError, match="missing"):
        resolve_config(args, old)


def test_cli_score_has_no_gpu_imports(tmp_path, config):
    # Make a CLI manifest through the same public entry point, mocking only
    # generation. Then run scoring in an isolated process where GPU imports fail.
    import unittest.mock
    args = parser().parse_args(["--output_dir", str(tmp_path), "--limit_per_subject", "1", "--stage", "generate"])
    with unittest.mock.patch("mmmu_eval.generate.generate"):
        run(args)
    meta = read_json(tmp_path / "run_meta.json")
    _, fixture_meta = fixture_run(tmp_path, config)
    meta.update({key: value for key, value in fixture_meta.items() if key != "protocol_hash"})
    write_json(tmp_path / "run_meta.json", meta)
    code = '''
import sys
class BlockGPU:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','vllm','transformers','qwen_vl_utils','datasets'}:
            raise RuntimeError('GPU/dataset import forbidden: ' + fullname)
sys.meta_path.insert(0, BlockGPU())
from mmmu_eval.cli import main
main()
'''
    import os
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    for stage in ("score", "report"):
        result = subprocess.run([sys.executable, "-c", code, "--stage", stage, "--output_dir", str(tmp_path)],
                                env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    with pytest.raises(ValueError, match="changed"):
        run(parser().parse_args(["--output_dir", str(tmp_path), "--stage", "score", "--seed", "99"]))
