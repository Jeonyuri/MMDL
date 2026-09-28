# MMMU-val Baseline Evaluation Report — Qwen3-VL-4B-Instruct

- **팀명**: x64
- **팀원**: 전유리(팀장), 김소정, 임나경, 정세은
- **작성일**: 2026-09-28
- **재현 커맨드**: 아래 §1 참고 (생성 `scripts/run_mmmu_eval.sh` → 재채점 `scripts/rescore.py`). 실제 제출 실행은 Colab A100에서 노트북 셀로 수행했다(실행 로그: `notebooks/runs/seed3407_final_16k.ipynb`; 이전 실행에 쓰던 노트북 셀을 config/output_dir만 바꿔 재사용했다). 9개 사전 실험의 실행 로그도 `notebooks/runs/`에 있다(`notebooks/README.md` 참고).

---

## 1. 환경 / 재현성

| 항목 | 값 |
|---|---|
| 모델 checkpoint | `Qwen/Qwen3-VL-4B-Instruct` (ebb281ec70b05090aa6165b016eac8ec08e71b17), bf16 |
| 데이터 | `MMMU/MMMU` validation (98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68), 30과목 각각 로드, 900문항 무필터 |
| 추론 백엔드 | vLLM 0.11.0 (torch 2.8.0, transformers 4.57.1, qwen-vl-utils 0.0.14, datasets 4.2.0) — 최종 실행 `run_meta.json`의 `packages`와 동일. 과목별 30문항을 배치로 묶어 한 번에 추론하는 vLLM의 연속 배치(continuous batching)가 필요해 선택했다(§5.2의 과목별 생성 시간도 이 배치 단위로 측정됨) |
| 사용 GPU | NVIDIA A100-SXM4-40GB (Google Colab Pro+) |
| 실측 peak VRAM | 38.1 GiB (39,022 MiB, `run_meta.json`의 `peak_vram_mib`). 단 vLLM이 `gpu_memory_utilization=0.9`만큼 미리 잡는 값이라 **실제 필요량이 아닌 장치 전체 사용량**이다(8k 실행들도 동일하게 약 38 GiB) |
| 생성 단계 소요 시간 | 약 114분(1시간 54분, `run_meta.json`의 `generation_seconds`). 엔진 시작(engine startup) 포함, preflight 제외(`generation_time_note`). 8k baseline(약 57~63분)의 약 2배. Drive 마운트·패키지 설치 등 노트북 전체 경과 시간은 별도 |
| 의존성 | [requirements.txt](../code/qwen3_vl_mmmu_eval/requirements.txt) (CPU 전용 테스트/채점: [requirements-cpu.txt](../code/qwen3_vl_mmmu_eval/requirements-cpu.txt)) |
| 실행 커맨드 | 아래 코드 블록 |

**`code/qwen3_vl_mmmu_eval/`(저장소 루트 기준)에서 실행한다.** 아래 `MODEL_PATH`/`DATA_ROOT`/`OUTPUT_ROOT`를 자기 환경의 실제 경로로 바꿔서 **한 번에 실행**한다(예시 경로가 그대로 들어 있어 형식을 참고할 수 있다).

**나중에 파인튜닝된 체크포인트로 재현하려면 `MODEL_PATH` 한 줄만 바꾸면 된다** — 나머지 파이프라인(재채점, §5 표 생성 포함)은 그대로다. `MODEL_PATH`가 로컬 디렉터리면 `MODEL_REVISION`은 코드가 자동으로 무시한다(`cli.py`). 로컬 체크포인트는 LoRA adapter만이 아니라 원본과 병합한 전체 가중치+tokenizer/processor/chat template여야 한다(`README.md` 참고).

```bash
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

## 2. 프롬프트

**실제 모델에 들어간 프롬프트 전문** (Qwen3-VL chat template 적용 후, 이미지가 텍스트 앞에 온다):

```
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Question: {question}
Options:
A. {option_A}
B. {option_B}
...(선택지 개수만큼, 최대 I까지)
Please select the correct answer from the options above.<|im_end|>
<|im_start|>assistant
```

- 이미지가 여러 장이면 `<|vision_start|><|image_pad|><|vision_end|>`가 그 수만큼 앞에 반복된다. system 프롬프트는 없다. 이미지당 `min_pixels`/`max_pixels`를 지정한다(§3.2).
- 주관식(open, 53문항)은 선택지가 없어 `Question: {question}`만 들어가고 지시문은 붙지 않는다(`prompt.py`의 `render_text`).
- **출처**: Qwen 공식 평가 툴킷의 MMMU 프롬프트(`QwenLM/Qwen3-VL` `evaluation/mmmu`의 `build_mmmu_prompt`) 사양을 따라 직접 구현했다(`src/mmmu_eval/prompt.py`). https://github.com/QwenLM/Qwen3-VL/tree/main/evaluation/mmmu — 저장소 페이지 요약으로 사양을 확인했으며, 원문 코드를 줄 단위·commit 단위로 대조하지는 않았다.
- **선택 이유**: 공식 구현을 참고한 기준 프롬프트를 쓰는 것이 격차 분석의 출발점으로 가장 깨끗하다. 풀이형/상세형 지시문도 시험했으나(§8), 형식 오류를 보정한 채점(v2) 기준으로 정확도 향상이 없어(baseline 62.89, 풀이형 59.22, 상세형 62.11) 공식 구현을 참고한 기준 프롬프트를 유지했다.
- 위 전문은 **최종 실행 자신의** `outputs/final_seed3407_16k/raw_outputs.jsonl` 첫 행의 실제 `prompt_text`(chat template 적용 후)를 직접 읽어 확인한 것이다. `resolved_config.json`의 `prompt.instruction`도 기존 문구(`Please select the correct answer from the options above.`)임을 함께 확인했다.

## 3. 생성(Decoding) 설정

### 3.1 Sampling recipe

| 파라미터 | 값 |
|---|---|
| `do_sample` | True — vLLM `SamplingParams`엔 이 이름의 필드가 없다(HuggingFace `generate()` 관용 표기). `temperature=0.7 > 0`이라 실질적으로 샘플링(비-greedy)임을 의미한다 |
| `temperature` | 0.7 |
| `top_p` | 0.8 |
| `top_k` | 20 |
| `repetition_penalty` | 1.0 |
| `presence_penalty` | 1.5 |
| `seed` | 3407 |

- **출처**: Qwen3-VL-4B-Instruct 모델 카드 "VL" 섹션의 권장 셸 예시를 그대로 따랐다: `top_p=0.8 top_k=20 temperature=0.7 repetition_penalty=1.0 presence_penalty=1.5 out_seq_length=16384, greedy=false`(모델 카드 원문 재확인 완료). https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- "Text"(순수 텍스트) 섹션의 권장값(top_p 1.0, top_k 40, temperature 1.0, presence_penalty 2.0, out_seq_length 32768)은 쓰지 않았다 — 우리 과제는 이미지가 포함된 VL 태스크이기 때문이다. **seed 3407은 모델 카드에 없는, 우리가 직접 고정한 값이다**(모델 카드 전체에 특정 seed 값 명시가 없음을 확인했다).
- presence_penalty를 0으로 낮춘 실험(§8)에서 v2 기준 -4.0점(p=0.001), 잘림 비율 12.6%→24.0%로 악화해 권장값을 유지했다.

### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 값 |
|---|---|
| `max_new_tokens` | 16384 (`max_model_len` 24576) |
| 이미지 해상도 처리 | `min_pixels`=1,310,720 (1280×32×32, 이미지당 1,280 토큰), `max_pixels`=5,242,880 (5120×32×32, 이미지당 5,120 토큰) |

**선택 근거**: 8192로는 응답의 12~15%가 잘렸고(잘린 응답 정답률 10~20%), 16384로 늘리면 seed 42 기준 v2 정확도가 61.56→64.22(+2.7점, McNemar p=0.018)로 올랐다. 16384는 모델 카드의 VL 권장 out_seq_length와도 같다. 비용은 A100 40GB에서 8k 약 1시간 대비 약 2시간으로 늘지만 Colab 컴퓨팅 유닛 안에서 감당 가능했다. `max_model_len`은 입력 최대 약 6.7k + 출력 16.4k = 23.1k < 24576이어서 입력 잘림이 없다. 해상도는 max_pixels/min_pixels를 각각 올려도 유의한 차이를 확인하지 못해(p=0.25/0.72) 기본값을 유지했다. 4k는 잘림이 19%로 늘고 정확도가 하락했다(-2.7점, p=0.022).

## 4. 채점(파싱) 방식

- 사용한 파서/로직:
  - **v1 파서**: Qwen 공식 평가 툴킷 `evaluation/mmmu/eval_utils.py`의 규칙 로직(`can_infer_option`, `can_infer_text`)을 옮기고 정규식 폴백(`answer is X`, `**X**`)을 더한 뒤 **GPT judge 단계만 제거**했다(`src/mmmu_eval/parse.py`). `can_infer_option`/`can_infer_text`는 upstream 세부 동작(대소문자 구분 거절 문구, 관사 A 가드, 표준 Z 처리)을 그대로 옮겼고, 정규식 폴백은 이를 보완하는 자체 단계다 — "그대로 포팅"이 아니라 **Qwen 규칙 기반 파서**로 표현한다.
  - **v2 파서(주 지표)**: 원본 결과를 유지하되 원본이 unparsed이고 응답이 잘리지 않았고 객관식일 때만 명시적 답 선언 패턴을 추가로 적용하는 자체 구현(`src/mmmu_eval/parse_v2.py`, `scripts/rescore.py`).
- 동작 방식 요약:
  1. `option`: 구두점을 공백으로 바꾼 뒤 선택지 글자(A~I)가 정확히 하나만 나오면 채택. 단 "A"가 있고 토큰이 3개를 넘으면 실패(관사 A 방지). 거절 문구가 있으면 refused(오답).
  2. `text`: 선택지 텍스트가 응답에 정확히 하나만 포함되면 채택.
  3. `regex`: `answer is X`, `**X**` 패턴의 마지막 유효 매치.
  4. (v2 전용) `fallback`: `**C. text**`, `**Option A**`, `final/correct answer is B`, `\boxed{B}` 중 마지막 유효 매치.
  5. 모두 실패하면 unparsed → 오답. 잘린 응답과 주관식에는 fallback을 적용하지 않는다.
- 주관식: 정답 문자열을 선택지 A, `Other Answers`를 B로 두는 pseudo-MC(2지선다)로 변환한 뒤 **v1 파서 전체 파이프라인**(`option`→`text`→`regex`, 위와 동일한 `extract_answer`)을 그대로 적용해 판정한다(`qwen_mc`, `score.py`). v2 fallback은 `question_type != multiple-choice`이면 건너뛰므로 주관식엔 적용되지 않는다. 공식 GPT judge는 사용하지 않았다.
- 검증: v1 파서를 재계산한 점수가 저장된 점수와 10개 실행(9개 사전 실험 + 최종) 모두 일치했다. v2가 복구한 응답을 두 단계로 수기 검수했다: (1) 사전 실험 9개에서 무작위 30건, (2) 최종 실행(`final_seed3407_16k`)에서 별도로 8건. 두 표본 모두 오추출 0건이었고, 오답으로 남은 건은 모델이 실제로 틀린 경우였다(표본은 `reports/rescore/audit_sample.jsonl`에 있다).
- 최종 실행에서 v1 파서 unparsed 273건 중 194건을 v2가 fallback으로 추가 추출했고 그중 135건이 정답이었다(추출 실패 273→79, 정답 수 440→575/900).

## 5. 결과

최종 실행 `final_seed3407_16k` (seed 3407, max_new_tokens 16384, 공식 구현을 참고한 기준 프롬프트). `Acc (v2)`가 주 지표이고 v1 파서 점수를 병기한다.

### 5.1 카테고리별 요약

MMMU의 공식 6개 상위 카테고리로 30개 과목을 묶었다(Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, Tech & Engineering). 카테고리 평균은 소속 과목 정확도의 평균이다(과목당 30문항으로 동일해 micro와 같다). 생성 시간은 §5.2의 과목별 시간을 카테고리로 합산한 값이다.

| Category | Subjects | Data Num | Acc (v2) | Acc (v1 parser) | 생성 시간 |
|---|---|---|---|---|---|
| Art & Design | 4 | 120 | 62.50 | 50.83 | 15:36 |
| Business | 5 | 150 | 74.67 | 64.00 | 15:03 |
| Science | 5 | 150 | 59.33 | 45.33 | 19:39 |
| Health & Medicine | 5 | 150 | 69.33 | 52.00 | 11:10 |
| Humanities & Social Science | 4 | 120 | 71.67 | 47.50 | 4:16 |
| Tech & Engineering | 7 | 210 | 51.90 | 38.10 | 43:37 |
| | **Overall** | **900** | **63.89** | **48.89** | **109:21** |

**카테고리별 관찰**: Tech & Engineering(7과목, 210문항)이 정확도(51.90%)와 시간(43:37, 전체의 40%) 양쪽에서 가장 낮고 가장 오래 걸렸다 — 과목 수가 가장 많고 Architecture_and_Engineering·Mechanical_Engineering·Energy_and_Power·Materials처럼 응답이 긴 과목이 몰려 있다. Business(74.67%)·Humanities & Social Science(71.67%)가 정확도가 가장 높고, Humanities & Social Science는 시간도 가장 짧다(4:16).

### 5.2 과목별 세부 결과

| No. | Subject | Data Num | Acc (v2) | Acc (v1 parser) | 생성 시간 |
|---|---|---|---|---|---|
| 1 | Accounting | 30 | 76.67 | 63.33 | 5:14 |
| 2 | Agriculture | 30 | 50.00 | 43.33 | 0:14 |
| 3 | Architecture_and_Engineering | 30 | 50.00 | 40.00 | 9:51 |
| 4 | Art | 30 | 56.67 | 46.67 | 3:37 |
| 5 | Art_Theory | 30 | 83.33 | 60.00 | 3:17 |
| 6 | Basic_Medical_Science | 30 | 76.67 | 63.33 | 0:26 |
| 7 | Biology | 30 | 50.00 | 43.33 | 3:30 |
| 8 | Chemistry | 30 | 50.00 | 36.67 | 4:42 |
| 9 | Clinical_Medicine | 30 | 70.00 | 50.00 | 3:14 |
| 10 | Computer_Science | 30 | 56.67 | 40.00 | 4:06 |
| 11 | Design | 30 | 80.00 | 70.00 | 0:16 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 30.00 | 16.67 | 0:29 |
| 13 | Economics | 30 | 86.67 | 70.00 | 0:48 |
| 14 | Electronics | 30 | 66.67 | 56.67 | 5:45 |
| 15 | Energy_and_Power | 30 | 53.33 | 30.00 | 8:37 |
| 16 | Finance | 30 | 63.33 | 63.33 | 4:20 |
| 17 | Geography | 30 | 56.67 | 40.00 | 3:49 |
| 18 | History | 30 | 66.67 | 36.67 | 3:14 |
| 19 | Literature | 30 | 83.33 | 50.00 | 0:09 |
| 20 | Manage | 30 | 56.67 | 43.33 | 2:05 |
| 21 | Marketing | 30 | 90.00 | 80.00 | 2:36 |
| 22 | Materials | 30 | 50.00 | 30.00 | 6:52 |
| 23 | Math | 30 | 60.00 | 46.67 | 4:12 |
| 24 | Mechanical_Engineering | 30 | 36.67 | 26.67 | 8:12 |
| 25 | Music | 30 | 30.00 | 26.67 | 8:26 |
| 26 | Pharmacy | 30 | 80.00 | 56.67 | 3:40 |
| 27 | Physics | 30 | 80.00 | 60.00 | 3:26 |
| 28 | Psychology | 30 | 76.67 | 50.00 | 0:41 |
| 29 | Public_Health | 30 | 90.00 | 73.33 | 3:21 |
| 30 | Sociology | 30 | 60.00 | 53.33 | 0:12 |
| | **Overall (macro avg)** | **900** | **63.89** | **48.89** | **109:21** |

계산식: `Overall = mean(30개 과목 accuracy)` (과목당 30문항이라 micro 평균과 같다. 실수 기준 v2 575/900, 원본 440/900). 표는 `scripts/make_report_table.py`로 생성했다. 참고로 객관식(847문항) 정확도는 64.94%, 주관식(53문항)은 47.17%(v2 기준)이다.

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---|
| 공식 (Qwen3-VL Technical Report) | 67.4 |
| 우리 재현 결과 (v2, 주 지표) | 63.89 |
| 우리 재현 결과 (v1 parser) | 48.89 |
| 차이 (Δ, v2 기준) | -3.51 |
| 차이 (Δ, v1 기준) | -18.51 |

## 7. 격차 분석

격차(공식 67.4 대비 Δ -3.51, v2 기준) 주원인은 채점 방식 차이로 판단하되, GPT judge 제거 외에 decoding·seed·이미지 처리·출력 처리 등 다른 프로토콜 차이도 있을 수 있어 전체를 파서 하나로 단정하지 않는다. (1) 파서: v1은 `**C. text**`처럼 글자 뒤에 내용이 붙은 답을 읽지 못해, 최종 실행에서 안 잘린 응답 194건(unparsed 273→79)을 오답 처리했고 135건이 정답이었다(440→575/900, 수기 검수 38건 오추출 0). 프롬프트 실험을 분리하면 상세형 62.89→62.11(p=0.628, 유의한 차이 없음), 풀이형 62.89→59.22(p≈0.011, 유의하게 하락) — v1 기준 +12~15점은 형식 준수 효과였다. (2) 잘림: 8k 12~15%, 16k도 9.1%(82건) 남음. 최종 16k(v2 63.89)는 8k baseline(62.89) 대비 +1.0점(p=0.417)으로 비유의했다(seed42는 +2.7점, p=0.018로 유의했으나 seed 3407에서는 같은 크기로 재현 안 됨). (3) presence_penalty=0은 개선책이 아니었다(-4.0점, p=0.001, 잘림도 급증); 해상도·seed는 유의하거나 일관된 효과를 확인하지 못했다(원인이 아니라 단정하지는 않음). (4) 카테고리별로는 Tech & Engineering(7과목, 210문항)이 51.90%로 가장 낮고 43:37로 가장 오래 걸렸다 — Diagnostics_and_Laboratory_Medicine(30.00%)·Mechanical_Engineering(36.67%)·Music(30.00%)처럼 응답이 길고 정확도가 낮은 과목이 몰려 있다. 안 잘린 unparsed 24건 중 23건이 주관식(v2 미적용, 객관식 64.94% vs 주관식 47.17%)도 남은 격차 후보다. v2는 자체 규칙이며 점수 확인 후 설계돼 공식과 동일 프로토콜이 아니다.

## 8. 기타 특이사항 / 한계 (Optional)

**사전 실험 요약** (모두 900문항, 각 실험은 baseline 대비 변수 1개만 변경. 그룹 내 baseline은 8k, 기존 프롬프트, presence 1.5):

| 그룹 | 실험 | v1 parser | v2 | 잘림% |
|---|---|---|---|---|
| seed 42 | max_new_tokens 4096 | 44.11 | 58.89 | 19.2 |
| seed 42 | **8192 (기준)** | 46.56 | 61.56 | 14.1 |
| seed 42 | max_new_tokens 16384 | 48.89 | 64.22 | 10.2 |
| seed 42 | max_pixels 6,553,600 | 47.00 | 63.00 | 14.2 |
| seed 42 | min_pixels 1,638,400 | 45.67 | 62.11 | 12.9 |
| seed 3407 | **baseline (기준)** | 47.00 | 62.89 | 12.6 |
| seed 3407 | 풀이형 프롬프트 | 59.22 | 59.22 | 14.8 |
| seed 3407 | 상세형 프롬프트 | 62.00 | 62.11 | 15.2 |
| seed 3407 | presence_penalty 0 | 46.56 | 58.89 | 24.0 |
| seed 3407 | **max_new_tokens 16384 (최종 제출)** | 48.89 | **63.89** | 9.1 |

- 표의 4k 실행만 A100 80GB에서 돌렸고 나머지는 40GB다(소요 시간 비교 시 주의).
- 풀이형: `Please reason step by step to solve the problem. End your response with the final answer formatted as **A**.` / 상세형: `You are an expert AI assistant highly specialized in mathematics, science, engineering, and humanities. First, thoroughly analyze the provided images. Second, reason step-by-step logically to break down the problem. Finally, conclude with the correct option formatted strictly as **A**.`
- **설정 선택의 한계**: 최종 설정(16k, 공식 구현을 참고한 기준 프롬프트, presence 1.5)은 validation 점수를 보고 고른 것이다. 이 점수는 파인튜닝 전후 비교의 기준선으로 쓰인다.
- **v2 파서의 한계**: 규칙을 unparsed 응답을 살펴본 뒤 설계했고(점수 확인 이후), 모든 실행에 같은 규칙을 일괄 적용했다. 공식 평가와 동일한 프로토콜이 아니다. fallback은 응답 안에서 `**X**` 형태의 마지막 매치를 최종 답으로 채택하므로, 모델이 설명 중간에 다른 선택지를 굵게 언급하고 정작 마지막 결론에서는 굵게 표시하지 않는 드문 경우 오추출할 가능성이 있다.
- **주관식**: v2가 다루지 않아 최종 실행에서 안 잘린 응답 중 24건이 여전히 unparsed이고, 그중 23건이 주관식이다(주관식은 총 53문항).
- **문항 단위 변동**: seed 42와 3407의 8k baseline은 총점이 비슷해도(v2 61.56 vs 62.89) 문항 단위로는 v2 기준 14.9%(134/900, McNemar gain 73+loss 61)가 서로 다르게 맞고 틀렸다(v1 기준으로는 24.2%, (111+107)/900 — 두 파서의 판정이 달라 수치가 다르다). 다만 두 seed의 8K 결과를 문항별로 짝지은 McNemar 검정에서는 유의한 차이를 확인하지 못했다(v2, p=0.342) — 문항 단위 뒤집힘이 있어도 seed 간 총점 차이 자체는 통계적으로 유의하지 않다는 뜻이다.
- **최종 16k의 개선폭은 유의하지 않았다**: 8k baseline(62.89) 대비 +1.0점(McNemar p=0.417), seed42 16k(64.22) 대비 -0.3점(p=0.851). seed42에서 관찰한 8k→16k 유의한 개선(+2.7점, p=0.018)이 seed 3407 최종 실행에서는 통계적으로 유의한 크기로 재현되지 않았다(p=0.417) — 두 비교 모두 같은 900문항(문항 표본 차이 없음)을 평가했으므로, 이 차이는 **seed에 따라** 토큰 예산 효과의 크기가 흔들릴 수 있음을 보여준다. **16k가 8k보다 항상 우월하다고 결론 내릴 수는 없다.**
- **다중 비교 주의**: 본문과 이 표에서 McNemar p값을 여러 쌍에 걸쳐 인용했다. 사전에 다중비교 보정(Bonferroni 등)을 적용하지 않았으므로, 개별 p값은 참고용이며 "유의/비유의" 판정을 엄밀한 보정 결과로 해석하지 않는다.
- **테스트 미실행**: 최종 실행 노트북에서 `pytest tests/` 단계는 주석 처리되어 실행하지 않았다. 코드 정확성은 이전 세션의 별도 pytest 실행과 v1 파서 재계산-저장값 일치 검증(§4)으로 대체 확인했다.
- **노트북 정리 필요**: 최종 생성에 쓴 노트북에는 이번 실행과 무관한 이전 탐색·디버깅 셀이 뒤쪽에 남아 있다. 제출본에서는 최종 실행에 해당하는 셀만 남기고 정리한다.
- **GPU 요구사항**: 40GB A100에서 검증했다. 더 작은 GPU(예: 24GB)에서의 동작은 확인하지 않았다.
- **시도하지 못한 것**: 프롬프트(풀이형·상세형)와 presence 0 조합, 32k 생성 길이, 공식 GPT judge 재현, 주관식 전용 파서.

## 참고문헌

[1] neur-lab. Assignment: Qwen3-VL-4B MMMU Baseline Evaluation. https://gist.github.com/neur-lab/38deabdfcde9e6dbacf362ab8059eb41

[2] neur-lab. MMMU-val Baseline Evaluation Report — 제출 양식. https://gist.github.com/neur-lab/483852e1f9d8d52f54627e600677c2f9

[3] Qwen. Qwen3-VL-4B-Instruct model card, Generation Hyperparameters / VL. https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

[4] MMMU dataset. https://huggingface.co/datasets/MMMU/MMMU

[5] QwenLM/Qwen3-VL, evaluation/mmmu (build_mmmu_prompt, eval_utils.py). https://github.com/QwenLM/Qwen3-VL/tree/main/evaluation/mmmu

