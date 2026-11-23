"""ViRL39K 풀(5,996)의 4B 4회 생성물을 v1(학습 노트북과 같은 규칙)과 v3 판정으로 다시 채점해 상태 변화를 비교한다.
입력(비공개): <PRIV>/virl39k/pool_for_colab/candidates.jsonl, <PRIV>/VIRL39K_RFT_4B/virl_generations_PRIVATE.jsonl
출력(비공개): <PRIV>/VIRL39K_RFT_4B/virl_status_v3_PRIVATE.jsonl (문항별 상태), virl_regrade_v3_changes_PRIVATE.jsonl (판정이 바뀐 샘플)
출력(집계, 공개 가능): <PRIV>/VIRL39K_RFT_4B/virl_regrade_v3_summary.json
v1 상태 분포가 학습 노트북 결과(virl_selection_report.json의 status_counts)와 같아야 통과한다.
실행: python3 virl_regrade_v3.py  (math_verify 필요)"""
import argparse, collections, json, os, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[3] / 'code/qwen3_vl_mmmu_eval/src'))
import open_answer_scoring as v1
import open_answer_scoring_v3 as v3
from mmmu_eval.parse_v2 import extract_answer_v2
from mmmu_eval.parse import DEFAULT_REGEX

RFT_SAMPLES, MIN_T, MAX_T = 4, 8, 4096
HARD_RULES = {'open': (1, 3), 'mc': (2, 3)}
CJK = re.compile(r'[぀-ヿ㐀-䶿一-鿿가-힯]')
EXPECTED_V1 = {'easy_4of4': 3211, 'hard': 1122, 'correct_0_of_4': 822, 'easy_4of4_no_valid_target': 499,
               'correct_1_of_4': 178, 'hard_no_valid_target': 164}

def status_of(kind, oks, toks, cjk):
    k = sum(oks); lo, hi = HARD_RULES[kind]
    st = 'hard' if lo <= k <= hi else 'easy_4of4' if k == RFT_SAMPLES else f'correct_{k}_of_{RFT_SAMPLES}'
    valid = any(o and not c and MIN_T <= t <= MAX_T for o, t, c in zip(oks, toks, cjk))
    if st in ('hard', 'easy_4of4') and not valid: st += '_no_valid_target'
    return st

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--priv', default=os.environ.get('MMDL_PRIV', str(HERE.parents[4] / 'mutimodal_private')))
    a = ap.parse_args(); priv = Path(a.priv); out_dir = priv / 'VIRL39K_RFT_4B'
    pool = [json.loads(l) for l in open(priv / 'virl39k/pool_for_colab/candidates.jsonl', encoding='utf-8') if l.strip()]
    gens = {}
    for l in open(out_dir / 'virl_generations_PRIVATE.jsonl', encoding='utf-8'):
        try: r = json.loads(l)
        except json.JSONDecodeError: continue
        if len(r.get('samples', [])) == RFT_SAMPLES: gens[r['uid']] = r
    rows, changes = [], []
    for q in pool:
        uid = q['uid'] if 'uid' in q else 'virl:' + q['qid']
        rec = gens.get(uid)
        if rec is None: continue
        o1, o3, toks, cjk, why = [], [], [], [], []
        for s in rec['samples']:
            text, trunc = s['text'], s['finish_reason'] != 'stop'
            toks.append(int(s['output_tokens'])); cjk.append(bool(CJK.search(text)))
            if q['kind'] == 'mc':
                p = extract_answer_v2(text, dict(zip('ABCDEFGHI', q['choices'])), DEFAULT_REGEX, 'multiple-choice', trunc)
                ok = (not trunc) and p['method'] not in ('unparsed', 'refused') and p['extracted'] == q['gold']
                o1.append(ok); o3.append(ok); why.append(None); continue
            if trunc:
                o1.append(False); o3.append(False); why.append(None); continue
            cands = v1.open_candidates(text)
            k1 = any(v1.match_open(c, q['gold']) for c in cands)
            res = [v3.match_open_why(c, q['gold']) for c in cands]
            k3 = any(ok for ok, _ in res)
            w = next((r for ok, r in res if ok), None) if k3 else ('int_fix' if any(r == 'int_fix' for _, r in res) else None)
            o1.append(k1); o3.append(k3); why.append(w)
            if k1 != k3:
                changes.append({'uid': uid, 'category': q['category'], 'source': q['source'], 'gold': q['gold'],
                                'cands': cands[:3], 'v1': k1, 'v3': k3, 'why': w})
        s1, s3 = status_of(q['kind'], o1, toks, cjk), status_of(q['kind'], o3, toks, cjk)
        rows.append({'uid': uid, 'kind': q['kind'], 'category': q['category'], 'source': q['source'].replace('Converted', ''),
                     'n_ok_v1': sum(o1), 'n_ok_v3': sum(o3), 'status_v1': s1, 'status_v3': s3, 'n_trunc': sum(s['finish_reason'] != 'stop' for s in rec['samples']),
                     'pass32': q.get('pass32'), 'why_v3': [w for w in why if w]})
    c1 = collections.Counter(r['status_v1'] for r in rows)
    assert dict(c1) == EXPECTED_V1, f'v1 재현 실패: {dict(c1)}'
    c3 = collections.Counter(r['status_v3'] for r in rows)
    trans = collections.Counter((r['status_v1'], r['status_v3']) for r in rows if r['status_v1'] != r['status_v3'])
    summary = {'n': len(rows), 'status_v1': dict(c1), 'status_v3': dict(c3),
               'transitions': {f'{a} -> {b}': n for (a, b), n in trans.most_common()},
               'changed_samples': len(changes), 'changed_by_rule': dict(collections.Counter(c['why'] for c in changes)),
               'note': 'v1 = 학습 노트북 규칙(재현 확인), v3 = open_answer_scoring_v3.py. 객관식 판정은 동일.'}
    with open(out_dir / 'virl_status_v3_PRIVATE.jsonl', 'w', encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(out_dir / 'virl_regrade_v3_changes_PRIVATE.jsonl', 'w', encoding='utf-8') as f:
        for c in changes: f.write(json.dumps(c, ensure_ascii=False) + '\n')
    (out_dir / 'virl_regrade_v3_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
