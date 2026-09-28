# 결과 요약 (score_summary.md)

`reports/mmmu_baseline.md`의 §5~§6을 그대로 옮겼다. 전체 문항별 원본 응답·재채점 근거는 `code/qwen3_vl_mmmu_eval/outputs/final_seed3407_16k/`와 `code/qwen3_vl_mmmu_eval/reports/rescore/`에 있다.

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

**생성 시간 측정 방법(§5.1·§5.2 공통)**: `run_meta.json`엔 총합(`generation_seconds`)만 있고 과목별 필드는 없다. 대신 vLLM이 과목당 30문항을 배치로 처리할 때 찍는 tqdm 진행바(`Processed prompts: 100%|...|[MM:SS<00:00,...]`)가 노트북 stdout에 그대로 남아 있어, 이를 `scripts/extract_subject_timing.py`로 파싱해 두 표에 실었다(`outputs/final_seed3407_16k/subject_timing.json`). 과목별 배치 처리 시간 합계는 109분 21초이며, 전체 생성 단계 기록은 약 113분 47초이다(`generation_seconds` 6826.62초). 약 4분 26초의 차이에는 엔진 초기화, 과목별 입력 준비·결과 저장 등 진행바 측정 범위 밖의 작업이 포함될 수 있다(`generation_time_note` 참고).

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---|
| 공식 (Qwen3-VL Technical Report) | 67.4 |
| 우리 재현 결과 (v2, 주 지표) | 63.89 |
| 우리 재현 결과 (v1 parser) | 48.89 |
| 차이 (Δ, v2 기준) | -3.51 |
| 차이 (Δ, v1 기준) | -18.51 |

67.4는 과제 안내문(assignment_guidance.md)에 제시된 공식 수치이며, GPT judge를 사용한 평가로 추정된다. judge를 제거한 우리 점수와 동일 프로토콜 비교가 아니다.
