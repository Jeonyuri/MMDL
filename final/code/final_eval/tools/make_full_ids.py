"""전체 최종 평가용 문항 ID 파일 생성 (2026-11-21): MMMU test 10,500 전부 + MMMU-Pro 1,730 전부.
문항 내용은 넣지 않고 ID만 데이터셋 순서대로 적는다(표본 추출 없음).
입력: 로컬 MMMU 캐시(고정 revision의 test arrow), MMMU-Pro 메타데이터(pro_meta.py 출력, 비공개)
출력: MMDL/code/qwen3_vl_mmmu_eval/configs/final_eval_ids_full.json
실행: python make_full_ids.py [--pro_meta <path>]"""
import argparse, collections, json, os
from pathlib import Path
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[4]   # 저장소 최상위 (MMDL/)
REPO = ROOT.parent
PROJ = ROOT / 'code/qwen3_vl_mmmu_eval'
# 로컬 MMMU 캐시(HF datasets, 고정 revision). 환경 변수 MMMU_CACHE로 지정
CACHE = Path(os.environ.get('MMMU_CACHE', REPO / 'MMDL/pmc_vqa/eval/mmmu_dataset_cache/MMMU___mmmu'))
MMMU_REV = '98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68'
PRO_REV = '0d7426df4ccb3d8704a992fe6850abee7c262127'
import sys
sys.path.insert(0, str(PROJ / 'src'))
from mmmu_eval.data import SUBJECTS


def arrow_ids(path):
    with pa.memory_map(str(path)) as src:
        try:
            t = pa.ipc.open_stream(src).read_all()
        except pa.ArrowInvalid:
            t = pa.ipc.open_file(src).read_all()
    return [str(x) for x in t.column('id').to_pylist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pro_meta', default=str(Path(os.environ.get('MMDL_PRIV', REPO / 'mutimodal_private')) / 'final_eval/pro_meta_PRIVATE.json'))
    a = ap.parse_args()
    test = {}
    for s in SUBJECTS:
        files = sorted((CACHE / s / '0.0.0' / MMMU_REV).glob('mmmu-test*.arrow'))
        assert files, f'MMMU test arrow 없음: {s}'
        ids = [i for f in files for i in arrow_ids(f)]
        assert len(ids) == len(set(ids)) and all(i.startswith(f'test_{s}_') for i in ids), s
        test[s] = ids
    n_test = sum(map(len, test.values()))
    assert n_test == 10500, n_test

    meta = json.load(open(a.pro_meta))
    assert meta['revision'] == PRO_REV
    std, vis = meta['standard'], meta['vision']
    assert len(std) == len(vis) == 1730 and {r['id'] for r in std} == {r['id'] for r in vis}
    vs = {r['id']: r['subject'] for r in vis}
    pro = collections.OrderedDict((s, []) for s in SUBJECTS)
    for r in std:
        assert r['subject'] in pro and vs[r['id']] == r['subject'], r['id']
        pro[r['subject']].append(str(r['id']))
    assert all(pro.values())

    spec = {
        'note': '전체 최종 평가 문항 ID (2026-11-21, 표본 없음 — 각 세트의 고정 revision test 분할 전부, 데이터셋 순서). 문항 내용은 없음.',
        'mmmu_test': {'dataset': 'MMMU/MMMU', 'revision': MMMU_REV, 'split': 'test', 'mode': 'full', 'per_subject': None,
                      'counts': {s: len(v) for s, v in test.items()}, 'total': n_test, 'ids': test},
        'mmmu_pro': {'dataset': 'MMMU/MMMU_Pro', 'revision': PRO_REV, 'split': 'test', 'mode': 'full', 'per_subject': None,
                     'configs': {'pro_standard': 'standard (10 options)', 'pro_vision': 'vision'},
                     'rule': 'standard (10 options)의 1,730문항 전부, vision도 같은 ID',
                     'counts': {s: len(v) for s, v in pro.items()}, 'total': 1730, 'ids': dict(pro),
                     'stats': {'validation_derived': sum(i.startswith('validation_') for v in pro.values() for i in v)}},
    }
    out = PROJ / 'configs/final_eval_ids_full.json'
    if out.exists():
        old = json.loads(out.read_text())
        assert old == spec, f'{out}이 이미 있고 내용이 다릅니다. 고정본이므로 덮어쓰지 않습니다.'
        print('이미 같은 내용:', out); return
    out.write_text(json.dumps(spec, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('wrote', out, '| MMMU test', n_test, 'min/max per subject', min(spec['mmmu_test']['counts'].values()),
          max(spec['mmmu_test']['counts'].values()), '| Pro 1730 per subject', min(spec['mmmu_pro']['counts'].values()),
          max(spec['mmmu_pro']['counts'].values()), '| validation-derived', spec['mmmu_pro']['stats']['validation_derived'])


if __name__ == '__main__':
    main()
