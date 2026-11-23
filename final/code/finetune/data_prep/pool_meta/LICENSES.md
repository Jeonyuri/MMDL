# ViRL39K 학습 풀의 출처별 라이선스

학습 데이터는 [`TIGER-Lab/ViRL39K`](https://huggingface.co/datasets/TIGER-Lab/ViRL39K) @ `812ec617dea4bc8a4e751663b88e4ebb7de4d00e`에서 만든 5,996문항 풀입니다(VL-Rethinker, NeurIPS 2025). 데이터셋 카드는 MIT이지만, 각 문항의 원 출처 약관까지 정리하지는 않습니다. 아래 표는 풀의 `source` 칸을 원 출처 라이선스로 분류한 결과입니다(2026-11-17 점검).

| 분류 | 출처 (`source`) | 문항 수 | 원 출처 라이선스 |
| --- | --- | ---: | --- |
| 허용적 | MMK12 (+Converted) | 1,845 | Apache-2.0 |
| 허용적 | MMMath | 456 | MIT (THU-KEG/MM_Math) |
| 허용적 | R1OneVision (+Converted) | 206 | Apache-2.0 (LLaVA-OneVision 기반) |
| 비상업 | M3CoT (+Converted) | 520 | Hugging Face 표기는 MIT, 프로젝트 사이트는 CC BY-NC-SA (연구 전용) |
| 비상업 | ScienceQA (+Converted) | 368 | CC BY-NC-SA 4.0 |
| 비상업 | dvqa | 99 | CC BY-NC 4.0 |
| ShareAlike | ai2d (+Converted) | 69 | CC BY-SA |
| 불명확 | Processed | 2,091 | 출처 문서 없음 |
| 불명확 | K12 | 195 | 명시 없음 |
| 불명확 | geoqa_plus (+Converted) | 115 | 라이선스 없음, 허가제 접근 |
| 불명확 | chartqa | 32 | 저장소 GPL-3.0, 차트는 Statista·Pew·OWID·OECD에서 수집 |
| **합계** | | **5,996** | 허용적 41.8% · 비상업 16.5% · ShareAlike 1.2% · 불명확 40.6% |

## 이 저장소의 사용 범위

- 수업 과제(비상업·학술 목적)로만 사용했습니다. 이는 가장 엄격한 출처 조건(비상업·연구 전용)에도 맞습니다.
- 문항 원문·보기·정답·이미지와 모델이 생성한 풀이는 **재배포하지 않습니다.** 이 폴더에는 개수와 해시 같은 집계 정보만 있습니다.
- 이 데이터로 학습한 LoRA adapter도 같은 이유로 공개하지 않습니다. 공개한다면 연구·비상업 용도로 한정하고 위 출처를 함께 밝혀야 합니다.
- 상업적으로 쓰려면 허용적 라이선스 출처만으로 다시 학습해야 합니다.

## 이 폴더의 파일

| 파일 | 내용 |
| --- | --- |
| `pool_manifest.json` | 풀 5,996문항의 카테고리·형식·정답 유형별 개수, 제외 사유별 개수, 데이터셋 revision, 풀 파일 SHA-256 |
| `leak_audit.json` | MMMU(val·test·dev)·MMMU-Pro와의 중복 검사 결과: 후보 6,415 중 397건 격리, 신호(pHash·CLIP·텍스트)·카테고리·출처별 개수와 기준값 |
| `virl_regrade_v3_summary.json` | 기본 4B 모델의 문항당 4회 풀이를 채점한 정답 수별 문항 분포(v1·v3 채점 규칙 비교) |
| `LICENSES.md` | 이 문서 |
