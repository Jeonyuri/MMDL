# configs/

## 실제로 `--config`에 전달된 파일이 아닙니다 — 읽기 전에 꼭 확인하세요

`final_16k.yaml`(제출 실행)을 **제외한 모든 파일**은 실제 실행 노트북(`notebooks/runs/`)에서 `--config configs/mmmu_eval.yaml`로 고정 전달됐고, 실험마다 `mmmu_eval.yaml`의 값을 실행 직전에 직접 고쳐 쓰는 방식으로 진행했습니다. 즉 **아래 이름 붙은 config 파일들은 그 실행 시점에 존재하지 않았습니다.**

그래서 각 실행이 끝난 뒤 그 실행 폴더의 `outputs/<실행>/resolved_config.json`(vLLM에 실제로 적용된 값의 기록)에서 값을 그대로 옮겨 **사후에 재구성**했습니다. 모든 파일을 대응 `resolved_config.json`과 필드 단위(sampling 파라미터, 생성 예산, 해상도, 프롬프트 문구)로 대조해 **완전히 일치함을 확인**했습니다. 즉 내용은 신뢰할 수 있는 재현용 파일이지만, "이 파일이 그 명령에 그대로 쓰였다"는 뜻은 아닙니다.

## 파일명 규칙

`<seed>_<변경 변수>.yaml`. `notebooks/runs/<같은 이름>.ipynb`가 그 실행의 로그다(예: `seed42_4k.yaml` ↔ `notebooks/runs/seed42_4k.ipynb` ↔ `outputs/baseline_seed42_4k/`).

## 파일 목록

| 파일 | 실행 | 실제 실행에 쓰였나 | 대응 outputs |
|---|---|---|---|
| `mmmu_eval.yaml` | 그룹 B 기준선 (seed 3407, 8k) | ✅ `--config`로 직접 전달됨 | `baseline_seed3407_baseline` |
| `final_16k.yaml` | 최종 제출 (seed 3407, 16k) | ✅ `--config`로 직접 전달됨 (유일하게 전용 파일로 실행) | `final_seed3407_16k` |
| `seed42_4k.yaml` | 그룹 A, max_new_tokens 4096 | ❌ 사후 재구성 (실제론 mmmu_eval.yaml 수정) | `baseline_seed42_4k` |
| `seed42_8k.yaml` | 그룹 A 기준선, seed 42 | ❌ 사후 재구성 | `baseline_seed42_8k` |
| `seed42_16k.yaml` | 그룹 A, max_new_tokens 16384 | ❌ 사후 재구성 | `baseline_seed42_16k` |
| `seed42_high_max.yaml` | 그룹 A, max_pixels 상향 | ❌ 사후 재구성 | `baseline_seed42_high_max` |
| `seed42_high_min.yaml` | 그룹 A, min_pixels 상향 | ❌ 사후 재구성 | `baseline_seed42_high_min` |
| `seed3407_prompt.yaml` | 그룹 B, 풀이형 프롬프트 | ❌ 사후 재구성 | `baseline_seed3407_prompt` |
| `seed3407_presence0.yaml` | 그룹 B, presence_penalty 0 | ❌ 사후 재구성 | `baseline_seed3407_presence0` |
| `seed3407_detailed_prompt.yaml` | 그룹 B, 상세형 프롬프트 | ❌ 사후 재구성 | `baseline_seed3407_detailed_prompt` |

풀이형 프롬프트 + presence 0, 상세형 프롬프트 + presence 0 조합은 단일 변수 비교 결과를 본 뒤 **실행하지 않기로 결정**했고, config 파일도 만들지 않았다(대응 `outputs/` 없음).

## 왜 이렇게 됐는지

빠른 반복 실험 단계에서 매번 새 yaml 파일을 만들지 않고 `mmmu_eval.yaml` 하나를 고쳐 쓰는 방식으로 진행했습니다. 이후 재현성 정리 과정에서 "실행 노트북은 항상 `configs/mmmu_eval.yaml`만 가리키는데 `configs/`엔 실험별 파일이 있다"는 불일치가 보여, 각 파일이 실제로 무엇을 근거로 만들어졌는지와 검증 여부를 이 문서에 명시합니다. 값 자체(`resolved_config.json`과의 일치)는 스크립트로 검증했으니, 특정 실험을 다시 돌리고 싶다면 이 파일들을 그대로 `--config`에 넘기면 됩니다 — 다만 "우리가 실제로 그렇게 실행했다"고는 주장하지 않습니다.

`final_16k.yaml`만 예외로, 최종 제출 실행에서 실제로 `--config configs/final_16k.yaml`로 직접 전달됐습니다(`notebooks/runs/seed3407_final_16k.ipynb`에서 확인 가능).
