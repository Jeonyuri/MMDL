# notebooks/

이 폴더의 노트북은 **재현 경로가 아니라 실행 기록**입니다. 모두 Google Colab 전용이며 개인 Drive 절대경로(`/content/drive/MyDrive/...`)와 `google.colab` 임포트가 들어 있어, 이 저장소를 clone한 환경에서는 그대로 실행되지 않습니다.

**실제 재현 경로는 프로젝트 루트 `README.md`와 `reports/mmmu_baseline.md` §1의 CLI 커맨드입니다** (`scripts/run_mmmu_eval.sh --model_path ... --data_root ... --config ...`). 채점자는 이 커맨드를 자기 환경의 경로로 바꿔 실행하면 됩니다.

`runs/*.ipynb`는 아래 10개 실험을 **실제로 실행한 로그**(셀 출력 포함)다. 각 파일은 대응 실행 폴더의 `run_meta.json`(생성 시각·peak VRAM·소요 시간)과 `scores.json`(macro 정확도)을 노트북 셀 출력과 대조해 맞는 실행인지 확인한 뒤 이 이름으로 정리했다.

## `runs/` ↔ `outputs/` 대응

| 노트북 | 대응 `outputs/` 폴더 |
|---|---|
| `seed42_4k.ipynb` | `baseline_seed42_4k` |
| `seed42_8k.ipynb` | `baseline_seed42_8k` |
| `seed42_16k.ipynb` | `baseline_seed42_16k` |
| `seed42_high_max.ipynb` | `baseline_seed42_high_max` |
| `seed42_high_min.ipynb` | `baseline_seed42_high_min` |
| `seed3407_baseline.ipynb` | `baseline_seed3407_baseline` |
| `seed3407_prompt.ipynb` | `baseline_seed3407_prompt` |
| `seed3407_detailed_prompt.ipynb` | `baseline_seed3407_detailed_prompt` |
| `seed3407_presence0.ipynb` | `baseline_seed3407_presence0` |
| `seed3407_final_16k.ipynb` | `final_seed3407_16k` (★ 제출 결과) |

10개 실험 모두 실행 로그가 갖춰졌다.

각 실험의 config는 `configs/`(같은 이름: 예 `seed42_4k.yaml` ↔ `seed42_4k.ipynb`)에 대응 파일이 있다. **다만 최종 실행(`final_16k.yaml`)을 제외하면 실제 노트북은 전부 `--config configs/mmmu_eval.yaml`을 전달했고, 실험마다 이 파일을 직접 고쳐 썼다.** `configs/`의 이름 붙은 파일들은 각 실행의 `resolved_config.json`에서 값을 그대로 옮겨 사후에 재구성한 것이며, 필드 단위로 일치를 확인했다. 자세한 내용은 `configs/README.md` 참고.

## 정리 과정에서 발견한 것

- 일부 실험(`seed42_8k`, `seed3407_final_16k`)은 실행 로그가 두 벌 업로드됐다: 여러 날짜(Sep 22 실패 시도 포함)의 셀이 누적된 "스크래치" 버전과, 해당 실행 날짜의 셀만 남긴 "정리된" 버전. 두 버전 모두 같은 실행(같은 `macro=`, `peak_vram_mib`, `generation minutes`)임을 확인했고, **정리된 버전을 채택**했다(스크래치 버전에는 무관한 실패 로그가 섞여 있어 오해를 살 수 있음).
- `colab_run.ipynb`(재사용 템플릿 초안)는 실제로 한 번도 실행되지 않아 저장소에서 제외했다.
