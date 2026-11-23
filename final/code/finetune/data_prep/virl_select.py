"""ViRL39K -> RFT 후보 풀 (stage 1: 쿼터 추출). 입력은 비공개 폴더의 stage0 parquet(원본은 HF TIGER-Lab/ViRL39K, MIT).
출력: <PRIV>/virl39k/candidates_stage1.parquet (질문·정답 포함 -> 비공개, GitHub 금지)."""
import argparse, pandas as pd, numpy as np
import os
from pathlib import Path

# 비공개 작업 폴더: 환경 변수 MMDL_PRIV로 지정 (기본값: 저장소 상위의 mutimodal_private/)
PRIV = Path(os.environ.get('MMDL_PRIV', Path(__file__).resolve().parents[5] / 'mutimodal_private'))
QUOTA = {  # 누수 격리·채점 실패로 줄어들 것을 감안해 약 6,400건
    'Tables/Diagrams/Charts': 1600, '(GradeSchool) Non-Geo Math': 1600, '(GradeSchool) Science': 950,
    'Spatial Reasoning': 650, '(GradeSchool) Geometric': 650, 'Broader STEM Topics': 10**9, 'Social Science': 10**9}
EXCLUDE_SOURCES = {'DeepScaleR'}   # 텍스트 전용 수학 문제에 이미지 칸만 채워져 있음

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--priv', default=str(PRIV)); ap.add_argument('--seed', type=int, default=3407)
    a = ap.parse_args(); priv = Path(a.priv)
    d = pd.read_parquet(priv / 'virl39k_candidates_stage0.parquet')
    d = d[~d.source.isin(EXCLUDE_SOURCES)]
    imgs = pd.read_parquet(priv / 'virl39k_candidates_images.parquet')
    d = d.merge(imgs, on='qid', how='left')
    parts = []
    for cat, n in QUOTA.items():
        sub = d[d.category == cat]
        parts.append(sub.sample(n=min(n, len(sub)), random_state=a.seed))
    out = pd.concat(parts, ignore_index=True)
    out['image_member'] = out.image.map(lambda x: x[0])
    out = out.drop(columns=['image'])
    (priv / 'virl39k').mkdir(exist_ok=True)
    out.to_parquet(priv / 'virl39k' / 'candidates_stage1.parquet')
    print(len(out)); print(out.category.value_counts().to_string()); print(out.ans_kind.value_counts().to_dict()); print((out.n_opts >= 2).mean().round(3))

if __name__ == '__main__':
    main()
