# 최종 제출물: Qwen3-VL-4B-Instruct MMMU 성능 개선 (팀 x64)

Qwen3-VL-4B-Instruct의 MMMU 정확도를 파인튜닝으로 높이는 과제의 최종 제출물입니다. 최종 모델은 **ViRL39K로 학습한 GRPO(정답 보상 학습) LoRA 모델**(2라운드 step 170)입니다. 이 문서의 경로는 `final/` 폴더 기준이고, `../`는 저장소 최상위(`MMDL/`)입니다.

## 최종 결과

모든 평가는 A100-SXM4-40GB, seed 3407, 최대 출력 16,384토큰, 세트마다 1회 실행입니다. p는 McNemar 양측 정확 검정입니다.

| 평가 세트 | 문항 | Baseline | GRPO (최종) | 차이 | 새로 맞힘 / 틀림 | p | 잘림 비율 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MMMU validation (모델 선택에 사용) | 900 | 63.89% (575) | 66.22% (596) | +2.33%p | +74 / −53 | 0.076 | 9.1% → 2.4% |
| MMMU test | 10,500 | 61.30% (6,437) | 63.22% (6,638) | +1.91%p | +892 / −691 | <0.001 | 11.7% → 2.4% |
| MMMU-Pro standard (10지선다) | 1,730 | 51.33% (888) | 54.16% (937) | +2.83%p | +171 / −122 | 0.005 | 14.2% → 3.3% |
| MMMU-Pro vision | 1,730 | 48.61% (841) | 50.92% (881) | +2.31%p | +155 / −115 | 0.017 | 13.8% → 3.8% |

- 향상의 대부분은 답을 끝내지 못하고 잘리던 문항이 해소된 데서 나왔습니다. MMMU test에서 baseline만 잘린 1,098문항은 +323, 두 모델 모두 끝까지 답한 9,154문항은 −81입니다.
- 분석 전체: [`results/final_eval/analysis_full.md`](results/final_eval/analysis_full.md)
- 비교표: [`results/final_eval/final_comparison_full.md`](results/final_eval/final_comparison_full.md)
- MMMU 프롬프트·채점은 Assignment 1 baseline과 같습니다. MMMU-Pro는 공식 CoT 프롬프트와 공식 답 추출 방식을 쓰고, 무작위 추측은 끄고, 잘린 응답은 오답으로 처리했습니다. Qwen 공식 보고치와는 설정이 달라 직접 비교할 수 없습니다.

## 최종 방법

1. ViRL39K에서 MMMU·MMMU-Pro와 겹치는 문항을 제외해 5,996문항 풀을 만듭니다(397건 제외).
2. 기본 모델이 문항마다 4번 풀고, 정답과 오답이 섞여 나온 문항만 학습에 씁니다.
3. GRPO로 학습합니다. 보상은 정답 1, 오답 0, **잘린 응답 0**입니다. 그룹 평균보다 나은 풀이를 강화합니다.
   - 1라운드: 기본 모델 풀이 1,160문항 × 4 = 4,640응답, 290 step
   - 2라운드: 1라운드 모델이 새로 푼 680문항 × 8 = 5,440응답, 170 step
   - 2라운드는 step 175에서 생성 엔진(vLLM)과 학습 코드(HF)의 log-prob 차이 검사에 걸려 멈췄습니다. 직전 체크포인트인 step 170을 사후 후보로 검증했습니다(ViRL 검증 296문항 × 4: 894 → 910 / 1,184).

| 항목 | 설정 |
| --- | --- |
| 기본 모델 | `Qwen/Qwen3-VL-4B-Instruct` @ `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| LoRA | r=16, α=32, dropout 0, 언어 모델 self-attention q/k/v/o |
| 손실 | clipped GRPO (ε=0.2), 기준 모델 KL β=0.04, advantage = 보상 − 그룹 평균(표준편차로 나누지 않음, Dr. GRPO), 고정 길이 정규화 6,144 |
| 최적화 | 학습률 2e-5, warmup 10 step, 업데이트당 4그룹, grad clip 1.0 |
| 보상 | 객관식: MMMU 평가와 같은 답 추출 규칙 / 주관식: `open_answer_scoring_v3` + Math-Verify / 잘린 응답: 0 |
| 풀이 생성 | vLLM 0.11.0, 최대 6,144토큰, 평가와 같은 샘플링(temperature 0.7, top-p 0.8, top-k 20, presence 1.5) |
| 하드웨어 | 학습 Colab A100 80GB, 평가 A100-SXM4-40GB |
| 최종 adapter | SHA-256 `55922b8dfc2948d12f6e5190f02be0dd92804aa2ba0b78a19d2a3d533bcfc442` |

## 구조

```text
final/
├── README.md                          이 문서
├── code/
│   ├── finetune/                      Fine-tuning 코드
│   │   ├── data_prep/                 데이터 풀 구성, MMMU·Pro 중복 검사, 주관식 채점, GRPO 보상 (+ 테스트)
│   │   │   └── pool_meta/             풀 집계, 중복 검사 결과, 출처별 라이선스(LICENSES.md)
│   │   ├── tools/grpo_prep.py         보상 검증과 1라운드 문항 구성
│   │   └── train/
│   │       ├── ViRL39K_RFT_4B_SelfDistill_BF16_LoRA_A100.ipynb   4B 4회 풀이 생성 (+ 비교 실험 ViRL RFT)
│   │       ├── ViRL_RoundGRPO_4B_A100_Train_v4.ipynb              최종 GRPO 학습 (실행본)
│   │       └── ViRL_RoundGRPO_v4_Validate_Step170.ipynb           step 170 검증
│   └── final_eval/                    평가 노트북
│       ├── run_virl39k_grpo_mmmu_900_a100_40gb.ipynb              MMMU validation 평가
│       ├── run_final_eval_full_grpo_a100_40gb_mmmu_test.ipynb      MMMU test 평가 (GRPO)
│       ├── run_final_eval_full_baseline_a100_40gb_mmmu_test.ipynb  MMMU test 평가 (baseline)
│       ├── run_final_eval_full_grpo_a100_40gb_mmmu_pro.ipynb       MMMU-Pro 평가 (GRPO)
│       ├── run_final_eval_full_baseline_a100_40gb_mmmu_pro.ipynb   MMMU-Pro 평가 (baseline)
│       └── tools/                     문항 ID 생성, 결과 분석
└── results/
    ├── grpo_training/                 보상 검증(grpo_prep_report.json), round_grpo_v4/ 실행 기록
    │                                  (라운드별 학습 지표 metrics.jsonl, 체크포인트 상태, step 170 선택 근거, 단계별 소요 시간)
    ├── grpo_mmmu_val/                 MMMU validation 결과 요약 (baseline 대비)
    └── final_eval/                    MMMU test·Pro 결과 요약 (scores.json, run_meta.json, 비교표, 분석)
```

최종 평가 모듈은 기존 평가 패키지에 함께 있습니다. 같은 패키지의 `mmmu_eval`과 설정 파일을 그대로 쓰기 때문입니다.

- `../code/qwen3_vl_mmmu_eval/src/mmmu_final/`: MMMU test, MMMU-Pro standard·vision 생성·채점
- `../code/qwen3_vl_mmmu_eval/configs/final_eval_ids_full.json`: 최종 평가 문항 ID (문항 내용 없음)

## 재현 순서

학습 데이터·이미지·생성 풀이·가중치는 저장소에 없습니다(아래 "공개하지 않는 것"). 데이터 준비 스크립트는 비공개 작업 폴더를 환경 변수 `MMDL_PRIV`로 받습니다(예: `export MMDL_PRIV=/path/to/mutimodal_private`). `virl_regrade_v3.py`와 `grpo_prep.py`는 `--priv` 인자로도 지정할 수 있습니다.

노트북은 Colab에서 Google Drive 경로로 입출력합니다. 학습 산출물은 `MyDrive/VIRL39K_RFT_4B/` 아래에 저장됩니다. 평가 결과는 `MyDrive/MMDL/pmc_vqa/eval/mmmu_eval_outputs/` 아래에 저장되며, 저장소에서는 그 요약을 `results/`에 옮겨 두었습니다.

1. **환경**: Python 3.10+, torch 2.8.0, vLLM 0.11.0, transformers 4.57.1, `math-verify`. 학습·생성은 A100(80GB 권장), 평가는 A100-SXM4-40GB입니다.
2. **데이터 풀 (로컬 CPU)**: `TIGER-Lab/ViRL39K` @ `812ec617dea4bc8a4e751663b88e4ebb7de4d00e`를 받은 뒤 `code/finetune/data_prep/`에서 순서대로 실행합니다.
   - `virl_select.py`: 카테고리별 쿼터 추출(seed 3407), 텍스트 전용 출처 제외
   - `virl_leak_scan.py`: MMMU val·test·dev, MMMU-Pro와의 중복 검사(이미지 pHash, CLIP, 텍스트 유사도)
   - `virl_build_pool.py`: Colab용 풀 zip 생성
3. **4B 4회 풀이 생성 (Colab)**: `code/finetune/train/ViRL39K_RFT_4B_SelfDistill_BF16_LoRA_A100.ipynb`의 생성 단계를 실행해 문항당 4개 풀이를 만듭니다. 같은 노트북의 SFT 학습 부분은 비교 실험(ViRL RFT, val 569/900)입니다.
4. **보상 검증·1라운드 문항 구성 (로컬)**: `code/finetune/data_prep/virl_regrade_v3.py` → `code/finetune/tools/grpo_prep.py`를 실행합니다. 보상 테스트는 `code/finetune/data_prep/`에서 `python3 test_grpo_reward.py`입니다.
5. **GRPO 학습 (Colab A100 80GB)**: `code/finetune/train/ViRL_RoundGRPO_4B_A100_Train_v4.ipynb`를 실행합니다(`RUN_TAG=round_grpo_v4`, `MAX_ROUNDS=3`, `LEARNING_RATE=2e-5`). 1라운드 학습 약 1.8시간, 2라운드 생성 약 4시간 + 학습 약 2.7시간(step 170까지)이 걸렸습니다. 학습 지표는 `results/grpo_training/round_grpo_v4/round_0*/train/metrics.jsonl`에 있습니다.
6. **step 170 검증**: `code/finetune/train/ViRL_RoundGRPO_v4_Validate_Step170.ipynb`를 실행합니다.
7. **MMMU validation 평가 (A100 40GB)**: `code/final_eval/run_virl39k_grpo_mmmu_900_a100_40gb.ipynb`를 실행합니다(`POSTHOC_CANDIDATE='round_02_step170'`). adapter 해시를 확인한 뒤 기본 모델과 병합해 평가합니다.
8. **최종 평가 (A100 40GB)**: `code/final_eval/`의 `..._mmmu_test.ipynb`(MMMU test)와 `..._mmmu_pro.ipynb`(MMMU-Pro)를 모델별로 실행합니다. 노트북 첫 셀의 `PART`로 평가 세트를 고릅니다(`'mmmu_test'` 또는 `'pro'`). 다시 실행할 때는 엔진 중단 시 자동 재시도가 들어 있는 `_mmmu_test` 노트북의 코드를 쓰는 것을 권합니다.
   - 문항 ID: `../code/qwen3_vl_mmmu_eval/configs/final_eval_ids_full.json`
   - 데이터셋 고정 revision: MMMU `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`, MMMU-Pro `0d7426df4ccb3d8704a992fe6850abee7c262127`
   - 결과 분석: `python code/final_eval/tools/analyze_final_full.py` (문항별 `scored.jsonl`을 `results/final_eval/<모델>/<세트>/`에 둔 뒤 실행)

## 공개하지 않는 것

- **학습 데이터 원문·이미지·생성 풀이** (`*_PRIVATE*`): ViRL39K는 카드상 MIT지만 원 출처별 라이선스가 섞여 있어(일부 비상업·불명확) 재배포하지 않습니다. 출처별 내역은 [`code/finetune/data_prep/pool_meta/LICENSES.md`](code/finetune/data_prep/pool_meta/LICENSES.md)에 있습니다. 학술 목적으로만 사용했습니다.
- **모델 가중치·LoRA adapter**: 같은 이유로 재배포하지 않습니다. 위의 SHA-256으로 동일한 adapter인지 확인할 수 있습니다.
- **MMMU test·MMMU-Pro 문항별 출력** (`raw_outputs.jsonl`, `scored.jsonl`): 정답이 들어 있어 공개하지 않습니다. 집계(`scores.json`, `run_meta.json`, `analysis_full.*`)만 공개합니다.

## 참고문헌

- Qwen3-VL Technical Report, arXiv:2511.21631 (2025)
- MMMU (CVPR 2024), MMMU-Pro (ACL 2025)
- VL-Rethinker / ViRL39K (NeurIPS 2025), arXiv:2504.08837
- DeepSeekMath (GRPO), arXiv:2402.03300 · Dr. GRPO, arXiv:2503.20783 · DAPO (NeurIPS 2025)
- LoRA (ICLR 2022) · vLLM (SOSP 2023) · Hugging Face TRL, PEFT, Math-Verify
