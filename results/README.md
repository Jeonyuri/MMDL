# results/

`reports/mmmu_baseline.md`가 요구하는 "Experimental settings and results"를 담는 폴더다. 아래는 최종 제출 실행(`final_seed3407_16k`)의 핵심 결과만 추린 요약이고, **전체 원본(900문항 응답, 30과목 세부 로그, 9개 사전 실험, 실행 노트북)은 `code/qwen3_vl_mmmu_eval/`에 있다** — 결과를 만들어낸 코드와 같은 곳에 둬야 스크립트 기본 경로가 깨지지 않기 때문이다.

## 파일

| 파일 | 내용 |
|---|---|
| `final_config.yaml` | 최종 실행에 실제 `--config`로 전달된 설정(`code/qwen3_vl_mmmu_eval/configs/final_16k.yaml`과 동일) |
| `score_summary.md` | 30과목/6카테고리 정확도·생성시간 표, 공식 수치(67.4) 대비 비교 (`mmmu_baseline.md` §5~§6 발췌) |
| `rescore_summary.md` | 10개 실행(사전 실험 9 + 최종 1) 전체의 원본/개선 파서 점수, McNemar 검정 결과 |
| `audit_sample.jsonl` | 채점 방식(v2 파서)의 수기 검수 표본 38건(사전실험 30 + 최종실행 8), 오추출 0건 확인 근거 |

## 전체 원본 결과 위치

- 최종 실행 전체: `code/qwen3_vl_mmmu_eval/outputs/final_seed3407_16k/` (900문항 원본 응답 `raw_outputs.jsonl`, 채점 결과 `scored.jsonl`, 실행 메타데이터 `run_meta.json`, 과목별 소요 시간 `subject_timing.json` 등)
- 사전 실험 9개: `code/qwen3_vl_mmmu_eval/outputs/baseline_*/`
- 실행 로그(Colab 노트북, 셀 출력 포함): `code/qwen3_vl_mmmu_eval/notebooks/runs/`
- 재채점 산출물 전체: `code/qwen3_vl_mmmu_eval/reports/rescore/`
