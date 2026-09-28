#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
if [[ "$SCRIPT_DIR" == "${BASH_SOURCE[0]}" ]]; then SCRIPT_DIR=.; fi
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
exec "${PYTHON_BIN:-python}" -m mmmu_eval.cli "$@"
