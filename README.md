# MMDL — Assignment 1: Qwen3-VL-4B-Instruct MMMU-val Baseline

## 구조

```
MMDL/
├── README.md                 (이 파일)
├── reports/
│   └── mmmu_baseline.md      제출 보고서 (§1~§8, SUBMISSION_TEMPLATE.md 양식)
├── code/
│   └── qwen3_vl_mmmu_eval/   평가 파이프라인 전체 (src, scripts, configs, tests) + 실행 결과·로그
└── results/
    └── qwen3_vl_mmmu_eval/   핵심 결과 요약 (점수표, config, 재채점/검수 근거)
```

## 한 커맨드로 재현하기

`code/qwen3_vl_mmmu_eval/`에서 실행한다. `MODEL_PATH`/`DATA_ROOT`/`OUTPUT_ROOT`를 채우면 생성(A100 필요) → v2 재채점(CPU) → §5 표 생성까지 한 번에 돈다.

```bash
cd code/qwen3_vl_mmmu_eval

MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct" MODEL_REVISION="ebb281ec70b05090aa6165b016eac8ec08e71b17" \
DATA_ROOT="/path/to/mmmu_data" OUTPUT_ROOT="/path/to/new_outputs" RUN="final_seed3407_16k"; \
bash scripts/run_mmmu_eval.sh \
  --model_path "$MODEL_PATH" \
  --model_revision "$MODEL_REVISION" \
  --data_root "$DATA_ROOT" \
  --config configs/final_16k.yaml \
  --seed 3407 \
  --output_dir "$OUTPUT_ROOT/$RUN" \
  --require_gpu A100 \
  --stage all \
&& PYTHONPATH=src python scripts/rescore.py --outputs-dir "$OUTPUT_ROOT" --out-dir reports/rescore \
&& PYTHONPATH=src python scripts/make_report_table.py "$OUTPUT_ROOT/$RUN"
```

**나중에 파인튜닝된 체크포인트로 다시 돌리려면 `MODEL_PATH`(로컬 디렉터리 경로) 한 줄만 바꾸면 된다.** 로컬 경로를 주면 `MODEL_REVISION`은 코드가 자동으로 무시한다. 로컬 체크포인트는 LoRA adapter만이 아니라 원본과 병합한 전체 가중치 + tokenizer/processor/chat template여야 한다(`code/qwen3_vl_mmmu_eval/README.md` 참고).

세부 실행 환경(GPU, 의존성 버전, 소요 시간), 프롬프트, 생성 설정, 채점 방식, 결과, 공식 수치 대비 격차 분석은 [`reports/mmmu_baseline.md`](reports/mmmu_baseline.md)에 있다.

## 설치

```bash
cd code/qwen3_vl_mmmu_eval
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
pytest tests/
```

자세한 내용은 [`code/qwen3_vl_mmmu_eval/README.md`](code/qwen3_vl_mmmu_eval/README.md) 참고.
