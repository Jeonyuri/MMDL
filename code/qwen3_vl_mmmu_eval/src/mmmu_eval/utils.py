"""CPU-safe persistence, provenance, GPU checks and monitoring."""
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import warnings

PACKAGES = ("torch", "vllm", "transformers", "qwen-vl-utils", "datasets", "Pillow", "PyYAML")


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path, recover_tail=False):
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            if recover_tail and index == len(lines) - 1 and not line.endswith(b"\n"):
                warnings.warn(f"Recovering interrupted final JSONL write at line {index + 1}")
                break
            raise ValueError(f"Corrupt JSONL at {path}:{index + 1}") from None
    return rows


def write_jsonl(path, rows):
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def append_subject(path, rows):
    with Path(path).open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def source_hashes():
    return {path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))}


def package_versions():
    result = {}
    for package in PACKAGES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def git_info():
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        try:
            return subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}", *args],
                                           cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git("status", "--porcelain", "--untracked-files=normal")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


@contextlib.contextmanager
def output_lock(output_dir):
    """OS advisory lock: automatically released even after SIGKILL."""
    path = Path(output_dir) / ".run.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            handle.write(b"0")
            handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError(f"Another process is using {output_dir}") from None
    try:
        yield
    finally:
        handle.close()


def check_gpu(require_gpu=None):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Generation requires a CUDA GPU; score/report run on CPU.")
    name = torch.cuda.get_device_name()
    print(f"GPU: {name}", flush=True)
    if require_gpu and require_gpu not in name:
        raise RuntimeError(f"Required GPU {require_gpu!r}; found {name!r}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU must support bfloat16; no fp16/quantization fallback.")
    return name


class VRAMMonitor:
    """nvidia-smi device-total usage, not allocator usage. vLLM preallocates VRAM.

    On multi-GPU hosts peak is the largest single-device value across all visible
    nvidia-smi rows (including other processes); the full samples are saved too.
    """
    def __init__(self, path, interval=5):
        self.path, self.interval = Path(path), interval
        self.peak, self.error = None, None
        self.stop = threading.Event()

    def poll(self):
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True, timeout=10)
            values = [int(value.strip()) for value in output.splitlines() if value.strip()]
            if not values:
                raise ValueError("nvidia-smi returned no GPU memory values")
            self.peak = max(self.peak or 0, *values)
            append_subject(self.path, [{"unix_time": time.time(), "memory_used_mib": values}])
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            self.error = str(exc)
            warnings.warn(f"VRAM monitoring failed: {exc}")
            self.stop.set()

    def loop(self):
        while not self.stop.wait(self.interval):
            self.poll()

    def __enter__(self):
        self.poll()
        if self.error:
            raise RuntimeError(f"VRAM monitoring unavailable: {self.error}")
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=15)
        self.poll()
