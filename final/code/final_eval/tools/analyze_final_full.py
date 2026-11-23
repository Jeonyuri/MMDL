"""최종 전체 평가 분석(CPU, 2026-11-23): 과목·분야별 변화 + 잘림 기여도.
입력: final/results/final_eval/{baseline,grpo_v4_step170}/<set>/scored.jsonl (비공개, 로컬)
출력: 같은 폴더의 analysis_full.{json,md} (집계만 — 문항 원문 없음)
실행: python final/code/final_eval/tools/analyze_final_full.py"""
import json
from math import comb
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3] / 'results/final_eval'
SETS = {'mmmu_test': 'MMMU test', 'pro_standard': 'MMMU-Pro standard', 'pro_vision': 'MMMU-Pro vision'}
DISC = {'Art & Design': ['Art', 'Art_Theory', 'Design', 'Music'],
        'Business': ['Accounting', 'Economics', 'Finance', 'Manage', 'Marketing'],
        'Science': ['Biology', 'Chemistry', 'Geography', 'Math', 'Physics'],
        'Health & Medicine': ['Basic_Medical_Science', 'Clinical_Medicine', 'Diagnostics_and_Laboratory_Medicine', 'Pharmacy', 'Public_Health'],
        'Humanities & Social Sci.': ['History', 'Literature', 'Psychology', 'Sociology'],
        'Tech & Engineering': ['Agriculture', 'Architecture_and_Engineering', 'Computer_Science', 'Electronics', 'Energy_and_Power', 'Materials', 'Mechanical_Engineering']}
S2D = {s: d for d, ss in DISC.items() for s in ss}

def p_mcnemar(g, l):
    n, k = g + l, min(g, l)
    return min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0

def stats(pairs):
    a = sum(x[0]['correct'] for x in pairs); b = sum(x[1]['correct'] for x in pairs)
    g = sum(1 for x in pairs if not x[0]['correct'] and x[1]['correct']); l = sum(1 for x in pairs if x[0]['correct'] and not x[1]['correct'])
    return {'n': len(pairs), 'base': a, 'grpo': b, 'net': b - a, 'gain': g, 'loss': l, 'p': p_mcnemar(g, l),
            'base_acc': a / len(pairs) if pairs else 0, 'grpo_acc': b / len(pairs) if pairs else 0,
            'base_trunc': sum(x[0]['truncated'] for x in pairs), 'grpo_trunc': sum(x[1]['truncated'] for x in pairs)}

def load(tag, s):
    return {r['id']: r for r in map(json.loads, open(ROOT / tag / s / 'scored.jsonl', encoding='utf-8'))}

report = {}
for s in SETS:
    A, B = load('baseline', s), load('grpo_v4_step170', s); assert set(A) == set(B)
    P = [(A[i], B[i]) for i in A]
    rep = {'all': stats(P)}
    rep['subject'] = {sub: stats([x for x in P if x[0]['subject'] == sub]) for sub in sorted({x[0]['subject'] for x in P})}
    rep['discipline'] = {d: stats([x for x in P if S2D[x[0]['subject']] == d]) for d in DISC}
    groups = {'base_trunc_only': lambda a, b: a['truncated'] and not b['truncated'],
              'both_trunc': lambda a, b: a['truncated'] and b['truncated'],
              'grpo_trunc_only': lambda a, b: not a['truncated'] and b['truncated'],
              'neither_trunc': lambda a, b: not a['truncated'] and not b['truncated']}
    rep['truncation'] = {k: stats([x for x in P if f(*x)]) for k, f in groups.items()}
    # 길이 구간(baseline 출력 토큰 기준)
    bins = [(0, 1000), (1000, 5000), (5000, 16384), (16384, 10 ** 9)]
    rep['base_len_bin'] = {f'{lo}-{hi if hi < 10**9 else "trunc"}': stats([x for x in P if lo <= x[0]['output_tokens'] < hi]) for lo, hi in bins}
    if s != 'mmmu_test':
        rep['origin'] = {o: stats([x for x in P if x[0]['origin'] == o]) for o in ('test', 'validation')}
    else:
        rep['question_type'] = {q: stats([x for x in P if x[0]['question_type'] == q]) for q in ('multiple-choice', 'open')}
    report[s] = rep
(ROOT / 'analysis_full.json').write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding='utf-8')

def row(name, e):
    return (f"| {name} | {e['n']} | {e['base']} ({100*e['base_acc']:.1f}%) | {e['grpo']} ({100*e['grpo_acc']:.1f}%) | {e['net']:+d} | "
            f"+{e['gain']}/−{e['loss']} | {e['p']:.3f} | {e['base_trunc']}→{e['grpo_trunc']} |")
H = '| 구분 | n | baseline | GRPO | 순변화 | +/− | p | 잘림 수 |\n|---|---|---|---|---|---|---|---|'
md = ['# 최종 전체 평가 분석 (baseline vs GRPO v4 step 170, A100 40GB, seed 3407)',
      '', 'p = McNemar 정확 검정(양측, 다중 비교 보정 없음 — 과목·분야 행은 기술 통계로만 읽을 것).', '']
for s, name in SETS.items():
    r = report[s]
    md += [f'## {name}', '', '### 잘림 기여도', H, row('전체', r['all'])]
    md += [row({'base_trunc_only': 'baseline만 잘림', 'both_trunc': '둘 다 잘림', 'grpo_trunc_only': 'GRPO만 잘림',
                'neither_trunc': '둘 다 안 잘림'}[k], e) for k, e in r['truncation'].items()]
    md += ['', '### baseline 출력 길이 구간', H] + [row(k, e) for k, e in r['base_len_bin'].items()]
    extra = r.get('origin') or r.get('question_type')
    md += ['', '### 출처/문항 유형', H] + [row(k, e) for k, e in extra.items()]
    md += ['', '### 분야(6)', H] + [row(d, e) for d, e in r['discipline'].items()]
    md += ['', '### 과목(30, 순변화 순)', H] + [row(k, e) for k, e in sorted(r['subject'].items(), key=lambda kv: -kv[1]['net'])] + ['']
(ROOT / 'analysis_full.md').write_text('\n'.join(md), encoding='utf-8')
print('\n'.join(md))
