"""CPU integration doubles verify orchestration, not GPU model numerics."""
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from mmmu_eval.cli import parser, run
from mmmu_eval.generate import recover_subjects
from mmmu_eval.utils import read_json, read_jsonl
from test_data import FakeDataset


def test_mock_engine_end_to_end_and_resume(tmp_path, monkeypatch):
    import mmmu_eval.generate as generation
    from mmmu_eval.data import SUBJECTS
    calls = []

    class Processor:
        chat_template = "test template"
        image_processor = SimpleNamespace(patch_size=16, to_dict=lambda: {"patch_size": 16})
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            assert kwargs["revision"] == "ebb281ec70b05090aa6165b016eac8ec08e71b17"
            return cls()
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"][-1]["text"]
        def __call__(self, **kwargs):
            return {"input_ids": [[1, 2, 3]]}

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class LLM:
        def __init__(self, **kwargs):
            assert kwargs["dtype"] == "bfloat16"
            assert kwargs["max_model_len"] == 9048 and kwargs["seed"] == 42
        def generate(self, requests, sampling_params):
            calls.append(len(requests))
            assert all(p.kwargs["seed"] == 42 and p.kwargs["max_tokens"] == 2048 for p in sampling_params)
            return [SimpleNamespace(prompt=r["prompt"], prompt_token_ids=[1, 2, 3], outputs=[
                SimpleNamespace(text="42", token_ids=[1], finish_reason="stop")]) for r in requests]

    class Monitor:
        peak = 36000
        error = None
        def __init__(self, *args):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=Processor, AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace())))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=SamplingParams))
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", SimpleNamespace(
        process_vision_info=lambda *a, **k: (None, None, {})))
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=lambda path, subject, **k: FakeDataset(subject)))
    monkeypatch.setattr(generation, "check_gpu", lambda required: "MOCK A100 (no real GPU)")
    monkeypatch.setattr(generation, "VRAMMonitor", Monitor)
    args = parser().parse_args(["--output_dir", str(tmp_path), "--limit_per_subject", "1"])
    run(args)
    assert calls == [1] * 30
    rows = read_jsonl(tmp_path / "raw_outputs.jsonl")
    assert len(rows) == 30 and all(row["input_tokens"] == 3 for row in rows)
    assert len(read_json(tmp_path / "input_lengths.json")) == 900
    assert read_json(tmp_path / "run_meta.json")["status"] == "complete"
    run(args)
    assert calls == [1] * 30  # no extra inference
    assert read_jsonl(tmp_path / "raw_outputs.jsonl") == rows


def test_real_process_kill_recovery(tmp_path):
    """Kill a CPU writer mid-subject and recover completed durable rows."""
    raw = tmp_path / "raw_outputs.jsonl"
    marker = tmp_path / "ready"
    code = '''
import os, sys, time
from pathlib import Path
from mmmu_eval.utils import append_subject
raw, marker = map(Path, sys.argv[1:])
append_subject(raw, [{"id":"a1","subject":"A"},{"id":"a2","subject":"A"}])
append_subject(raw, [{"id":"b1","subject":"B"}])
with raw.open('ab') as handle:
    handle.write(b'{"id":"b2')
    handle.flush()
    os.fsync(handle.fileno())
marker.touch()
time.sleep(30)
'''
    import os
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    proc = subprocess.Popen([sys.executable, "-c", code, str(raw), str(marker)], env=env)
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            if proc.poll() is not None:
                raise AssertionError("Writer exited before test interruption")
            time.sleep(0.05)
        assert marker.exists(), "Writer startup timed out"
    finally:
        proc.kill()
        proc.wait(timeout=5)
    with pytest.warns(UserWarning):
        complete = recover_subjects(raw, {"A": ["a1", "a2"], "B": ["b1", "b2"]})
    assert complete == {"A"}
    assert [row["id"] for row in read_jsonl(raw)] == ["a1", "a2"]
