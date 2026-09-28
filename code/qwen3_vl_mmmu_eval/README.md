# Qwen3-VL-4B-Instruct MMMU-val 평가

평가 프로토콜을 고정해 baseline과 fine-tuned 모델을 같은 방식으로 비교하는 파이프라인입니다. 과목별 30문항, 30과목 총 900문항을 각각 Hugging Face config에서 로드합니다. 공식 67.4에 점수를 맞추는 구현이 아닙니다.

## 검증 상태

- CPU 검증 완료: `pytest tests/` **49 passed** (Windows, Python 3.12). 프롬프트, 파서, 데이터 검증, 전체 길이 검사, 실제 CPU 프로세스 강제 종료 후 파일 복구, 모의 엔진 통합 실행, GPU import를 차단한 별도 프로세스의 채점·보고서를 검증했습니다. 모의 엔진 테스트는 실제 모델 추론 검증을 대신하지 않습니다.
- `pip check`, Python 소스 컴파일, 노트북 JSON/코드 셀 문법, Bash 문법 및 셸 진입점 `--help` 검증을 통과했습니다.
- **실제 A100 추론·900문항 채점 완료**: 최종 실행(`outputs/final_seed3407_16k`, A100-SXM4-40GB, 약 114분)에서 900문항을 생성·채점했다. v2 파서 기준 63.89%(v1 파서 기준 48.89%). 근거와 재현 커맨드는 `reports/mmmu_baseline.md` §1·§5 참고. 강제 종료 후 재개(resume) 기능은 사전 실험 단계에서 확인했으나, 이번 최종 실행에서는 중단 없이 단일 세션으로 끝났다.
- `requirements.txt`의 `==` 버전은 **GPU 검증 후보**입니다. 검증 완료 버전으로 주장하지 않습니다. Colab에서 아래 검증을 통과한 뒤 제공한 스크립트로 실제 설치 버전을 고정해야 합니다.
- 기본 해상도로 900문항 중 길이 초과가 있으면 스모크도 생성 전에 종료됩니다. 이 경우 명세에 따라 사용자가 `--max_model_len`을 명시적으로 조정해야 합니다. 기본 9048이나 이미지 해상도를 자동으로 바꾸지 않습니다.

## 설치

일반 Linux + NVIDIA GPU에서 실행합니다. 권장 시작 환경은 Python 3.11/3.12와 A100입니다. vLLM 추론은 Windows 네이티브에서 지원하는 실행 경로로 제공하지 않습니다. 설치 대상 GPU 드라이버는 설치하는 PyTorch/vLLM CUDA wheel과 호환되어야 합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
pytest tests/
```

GPU가 없는 채점·개발 환경에서는 아래만 설치합니다.

```bash
python -m pip install -r requirements-cpu.txt
pytest tests/
```

Colab에서 실제로 실행한 노트북(실행 로그)은 `notebooks/runs/`에 있습니다. 개인 Drive 절대경로가 박혀 있어 그대로 실행되지는 않으며, 재현은 아래 CLI 커맨드를 쓰십시오. 자세한 설명은 `notebooks/README.md`를 참고하십시오.

## 한 커맨드 실행

```bash
bash scripts/run_mmmu_eval.sh \
  --model_path Qwen/Qwen3-VL-4B-Instruct \
  --model_revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --data_root "${MMMU_DATA_ROOT:-./data}" \
  --config configs/mmmu_eval.yaml \
  --seed 42 \
  --output_dir outputs/baseline_seed42 \
  --require_gpu A100
```

`--stage all`이 기본이며 generate → score → report를 실행합니다. `PYTHON_BIN`으로 Python 실행 파일을, `MMMU_DATA_ROOT`로 데이터 캐시를, `MMMU_OUTPUT_DIR`로 출력 경로 기본값을 지정할 수 있습니다. 다른 작업 디렉터리에서도 셸 스크립트의 절대/상대 경로로 호출할 수 있습니다.

생성 파라미터는 모두 YAML 기본값과 같은 이름의 CLI로 덮어쓸 수 있습니다. 예: `--max_new_tokens 2048 --temperature 0.7 --top_p 0.8 --top_k 20 --repetition_penalty 1.0 --presence_penalty 1.5 --seed 42 --min_pixels 1003520 --max_pixels 4014080 --limit_mm_per_prompt '{"image":10}' --gpu_memory_utilization 0.9 --max_model_len 9048`. `--backend vllm`만 지원하며 dtype은 bfloat16으로 고정합니다. 양자화/fp16 옵션은 없습니다.

데이터 revision `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`, split, 30과목 목록과 30/900 개수는 변경할 수 없습니다. 모델 기본 revision도 요청한 SHA입니다. 원격 모델 revision은 불변 commit SHA만 받습니다.

## 스모크와 실제 재개 검증

```bash
bash scripts/run_mmmu_eval.sh --limit_per_subject 1 \
  --require_gpu A100 --output_dir outputs/smoke_seed42
```

스모크는 각 과목의 첫 문항, 총 30문항을 생성하지만 원본 900개 개수 및 **900문항 전체 입력 길이**를 검증합니다. `--limit_per_subject`를 사용하면 30을 지정한 경우에도 `partial: true`와 `PARTIAL RUN`이 기록됩니다.

강제 종료 후 재개까지 자동 검증하려면 **새 출력 폴더**로 실행합니다. 이 테스트는 실제 GPU를 사용하며, 자신이 실행한 프로세스 그룹만 종료합니다.

```bash
python scripts/verify_gpu_resume.py \
  --data_root "${MMMU_DATA_ROOT:-./data}" \
  --output_dir outputs/gpu_resume_check
python scripts/freeze_verified_environment.py \
  --run_dir outputs/gpu_resume_check --output requirements.txt
```

첫 과목 저장 후 자식 vLLM worker를 포함한 프로세스 그룹을 강제 종료하고, 동일 커맨드로 재실행합니다. 완료된 과목의 응답이 그대로 유지되고 skip 로그가 있는지, 30개 고유 응답과 표가 완성되었는지 확인해 `gpu_resume_verified.json`을 만듭니다. 두 번째 스크립트는 이 증거와 현재 패키지 버전 일치를 확인하고 `pip check` 후 `pip freeze`의 정확한 버전들을 기록합니다. baseline과 fine-tuned 실행 모두 이 검증된 requirements를 사용하세요.

길이 초과 시 `overflow_report.json`을 보고 사용자가 결정한 `--max_model_len`을 추가합니다. 재개 검증 스크립트에도 같은 옵션이 있습니다. 이렇게 변경한 프로토콜은 baseline과 fine-tuned 양쪽에 똑같이 적용해야 합니다.

## 재개와 실패 처리

중단 후 **같은 커맨드, 같은 출력 폴더**로 다시 실행합니다. 과목 하나를 생성한 뒤 JSONL append + flush + fsync합니다. 완전한 과목은 건너뛰고, 일부만 저장된 과목은 기존 줄을 제거한 뒤 다시 생성합니다. 쓰다가 끊긴 마지막 JSON 줄만 경고와 함께 복구합니다. 중간 줄 손상, 중복 ID, 예상하지 않은 ID는 오류입니다. OS 파일 잠금은 프로세스가 죽으면 해제됩니다.

모델, 설정, 소스 코드, subset이 바뀐 폴더에 결과를 섞을 수 없습니다. 패키지 버전과 GPU 모델 변경도 생성 재개 시 실패합니다. 로컬 체크포인트는 가중치와 processor 파일의 SHA256을 검사하므로 같은 경로의 파일을 덮어써도 감지합니다. 설정을 바꾸려면 새 `output_dir`를 사용하세요.

입력 길이는 HF processor의 이미지 토큰 확장 후 `input_ids`로 계산합니다. 실제 vLLM의 `prompt_token_ids` 개수와 다르면 해당 과목 저장 전에 실패합니다. 길이 초과 항목은 표와 `overflow_report.json`으로 남깁니다. 문항을 버리거나 입력을 잘라 맞추지 않습니다.

GPU VRAM은 5초마다 `nvidia-smi`로 폴링합니다. `peak_vram_mib`는 장치 전체 사용량의 최고치이고, 여러 GPU가 조회되면 그중 최대 단일 장치 값입니다. 다른 프로세스 사용량도 포함됩니다. vLLM은 `gpu_memory_utilization`만큼 선점하므로 높게 보일 수 있습니다. 표본은 `vram_samples.jsonl`에 저장합니다. 생성 시간은 engine 시작을 포함하고 길이 검사 시간은 제외하며 재개 세션을 합산합니다. SIGKILL 당시 저장되지 않은 시간/VRAM은 정확히 복구할 수 없어 마지막 저장값과 세션 상태를 보존합니다.

요청 단위 seed와 엔진 seed를 모두 42로 지정합니다. 이 설정이 하드웨어·커널·패키지가 달라도 비트 단위 동일성을 보장하지는 않습니다. 비교 시 동일 GPU, 검증된 패키지, 코드 및 config를 사용하세요.

## CPU에서 다시 채점·보고서 생성

```bash
bash scripts/run_mmmu_eval.sh --stage score --output_dir outputs/baseline_seed42
bash scripts/run_mmmu_eval.sh --stage report --output_dir outputs/baseline_seed42
```

이 단계는 torch/vLLM/transformers/datasets를 import하지 않으며 GPU, 데이터 다운로드, 모델 파일이 필요 없습니다. 저장한 config와 manifest를 자동으로 사용합니다. `--stage score`는 채점 파일만, `--stage report`는 표만 갱신합니다. 전체 실행은 두 단계를 연속 처리합니다.

## 결과 확인

| 파일 | 내용 |
|---|---|
| `results_table.md` | 30과목 정확도, Overall macro, 공식 67.4 대비 percentage-point 차이 |
| `scores.json` | macro/micro, 유형별 정확도, 파싱 방법별 개수, 잘림 비율, 과목별 평균 출력 토큰 |
| `raw_outputs.jsonl` | 렌더링된 전체 프롬프트, 원문 응답, 토큰 수, 종료 이유/잘림, 정답, 선택지, GPU/seed |
| `scored.jsonl` | 원문 필드와 extracted/method/correct, 주관식 채점용 choices |
| `run_meta.json` | 전체 config, CLI 이력, 코드 SHA256, git commit/dirty, 패키지, GPU, VRAM, 시간, 데이터 개수, partial |
| `resolved_config.json` | CLI 적용 후 최종 설정 |
| `input_lengths.json` | 전체 900문항 입력 길이 및 프롬프트 해시 |
| `overflow_report.json` | 초과 문항 목록, 없으면 빈 리스트 |
| `chat_template.jinja` | 실제 processor chat template |

정확도는 소수점 둘째 자리 퍼센트로 표시합니다. Overall은 30과목 accuracy 단순 평균이며 micro와 `1e-12` 이내 같은지 assert합니다. parser 실패·거절은 오답입니다. 주관식은 생성 때 선택지를 추가하지 않고 채점 때만 `A=정답 문자열`, `B=Other Answers`, gold `A`로 취급합니다. 이 방식은 의미 기반 주관식 채점이 아니며 문자열 포함 규칙의 한계를 그대로 가집니다.

## Fine-tuned 체크포인트로 재평가하는 방법

full checkpoint에 모델 가중치와 tokenizer/processor/chat template 자산을 함께 저장하세요. LoRA adapter만 있는 폴더는 지원하지 않으며 먼저 원본 모델과 병합해 전체 체크포인트를 내보내야 합니다. processor와 chat template는 baseline 것을 유지해야 프롬프트가 같습니다.

```bash
bash scripts/run_mmmu_eval.sh \
  --model_path "${FINETUNED_MODEL_PATH}" \
  --data_root "${MMMU_DATA_ROOT:-./data}" \
  --config configs/mmmu_eval.yaml \
  --seed 42 --output_dir outputs/finetuned_seed42 --require_gpu A100
```

모델 경로와 새 결과 폴더만 바꾸고 다른 평가 설정은 baseline과 같게 둡니다. 로컬 디렉터리면 model revision은 경고 후 무시합니다. 변경했던 `max_model_len` 등이 있다면 양쪽에 동일하게 전달하세요. `protocol_hash`, 패키지 버전, GPU, `processor.chat_template_sha256`, `processor.image_processor`, 각 문항 `prompt_text`/`input_tokens`를 비교해 평가 조건이 같은지 확인합니다. `protocol_hash`는 모델 경로·revision을 제외한 config와 코드를 반영하며 processor 차이는 별도 필드로 확인해야 합니다.

## 공식 구현과의 차이

- GPT judge 및 무작위 답 선택을 제거했습니다. official option → text 규칙 뒤 regex fallback을 사용하며 마지막 **유효한** 매치만 선택합니다. 실패는 `unparsed`, 거절/`Z`는 `refused`입니다.
- regex는 `answer is (X)`, `answer: X`, `**X**`를 지원합니다. 답 문자는 대문자 A–I 중 해당 문항 선택지에 존재하는 문자만 인정합니다. 앞 단계가 성공하면 regex는 실행하지 않습니다.
- 요청마다 `SamplingParams(seed=seed)`를 지정합니다.
- 전체 일괄 생성 대신 과목별 생성·저장과 재개를 사용합니다.
- 공식 TSV 대신 요청한 pinned `MMMU/MMMU`의 30개 HF config를 로드합니다.
- 사용자 명세에 따라 hint/CoT 추가와 `</think>` 이후 응답 잘라내기를 하지 않습니다. 원문 전체를 채점합니다.
- BF16, 모델/데이터 revision, 전체 길이 검사, provenance 및 입력 길이 교차 검증을 추가했습니다.
- `generation_config="vllm"`으로 모델 저장소의 숨은 생성 기본값 병합을 막고 SamplingParams 설정을 기록합니다. EOS 등 나머지 엔진 동작은 고정된 vLLM/tokenizer 버전에 따릅니다.

공식 파서의 대소문자 구분 거절 문구, 구두점 제거, distinct choice 개수, 긴 문장의 관사 `A` 예외, 선택지 텍스트의 substring 비교는 유지했습니다. 사용한 공식 코드의 링크는 아래와 같습니다. 프로젝트 내 구현은 해시로 기록되므로 upstream main의 이후 변화가 기존 실행을 바꾸지 않습니다.

- [Qwen MMMU prompt / input preparation](https://github.com/QwenLM/Qwen3-VL/blob/main/evaluation/mmmu/run_mmmu.py)
- [Qwen rule parser](https://github.com/QwenLM/Qwen3-VL/blob/main/evaluation/mmmu/eval_utils.py)
- [Qwen open-question preprocessing](https://github.com/QwenLM/Qwen3-VL/blob/main/evaluation/mmmu/dataset_utils.py)
- [Qwen VL generation recipe script](https://github.com/QwenLM/Qwen3-VL/blob/main/evaluation/mmmu/infer_instruct.sh)
- [Pinned model card, Generation Hyperparameters (VL)](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/ebb281ec70b05090aa6165b016eac8ec08e71b17/README.md)

## 파일 역할

`data.py`는 원본 데이터 검증, `prompt.py`는 메시지 생성, `parse.py`는 결정적 답 추출, `generate.py`는 사전 검사·추론·재개, `score.py`는 CPU 채점, `report.py`는 표, `utils.py`는 저장·출처·GPU 모니터링, `cli.py`는 설정과 단계 실행을 담당합니다. 모델·캐시·대용량 결과는 `.gitignore`로 제외하며 결과 표와 scores만 선택적으로 커밋할 수 있습니다.
