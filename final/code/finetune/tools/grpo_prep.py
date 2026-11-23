"""GRPO 0유닛 준비 (2026-11-19): 보상 검증 + v3 기준 문항 목록 재구성 + 1라운드(기존 4B 생성물) 학습 샘플 집계.

입력(비공개, 로컬):
  <PRIV>/virl39k/pool_for_colab/{candidates.jsonl, images/}
  <PRIV>/VIRL39K_RFT_4B/{virl_generations_PRIVATE.jsonl, virl_status_v3_PRIVATE.jsonl, virl_selected_PRIVATE.jsonl}
  MMDL/code/qwen3_vl_mmmu_eval/outputs/final_seed3407_16k/scored.jsonl (baseline MMMU 응답, 객관식 보상=v2 확인용)
  MMDL/pmc_vqa/eval/mmmu_eval_outputs/*_v2/per_question_vs_baseline.csv (baseline v2 정오)
  MMDL/virl39k/train/ViRL39K_GRPO_4B_BF16_LoRA_A100.ipynb (기존 GRPO 노트북 보상과 비교, 선택)
출력:
  <PRIV>/VIRL39K_RFT_4B/grpo_items_PRIVATE.jsonl        문항별 상태·보상·태그 (비공개)
  <PRIV>/VIRL39K_RFT_4B/grpo_reward_disagree_PRIVATE.jsonl  기존 노트북 보상과 다른 샘플 (검토용, 비공개)
  final/results/grpo_training/grpo_prep_report.json               집계 (공개 가능, 원문 없음)
실행: python grpo_prep.py   (math_verify·numpy·pandas 필요)

문항 규칙(평가 전 고정): train 분할(노트북과 같은 seed·카테고리별 5% 검증 분할) 중
  주관식 1~3/4, 객관식 2~3/4 정답(새 보상 기준) = 1라운드 활성 문항. 객관식 1/4는 찍어서 맞힐 확률이 높아 제외(기존 hard 규칙과 동일).
  검증 문항과 이미지 파일이 같거나 정규화한 질문이 같은 학습 문항은 제외."""
import argparse, ast, collections as C, hashlib, json, os, re, statistics as S, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]   # 저장소 최상위 (MMDL/)
REPO = ROOT.parent         # 비공개 폴더 기본 위치의 기준
sys.path.insert(0, str(HERE.parent / 'data_prep'))
import numpy as np
import pandas as pd
from grpo_reward import score, gold_self_check, REWARD_VERSION

SEED, VAL_FRACTION = 3407, 0.05
ACTIVE = {'open': (1, 3), 'mc': (2, 3)}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]


def notebook_split(cands):
    """GRPO 노트북 셀 9와 같은 방식(np.RandomState(SEED), category별 5%)."""
    df = pd.DataFrame(cands)
    rng = np.random.RandomState(SEED)
    df['split'] = 'train'
    for _, idx in df.groupby('category').groups.items():
        idx = list(idx)
        v = rng.choice(idx, size=max(1, int(len(idx) * VAL_FRACTION)), replace=False)
        df.loc[v, 'split'] = 'val'
    return dict(zip(df.uid, df.split))


def old_notebook_reward(nb_path):
    """기존 GRPO 노트북에 내장된 보상 모듈을 그대로 불러온다(비교용)."""
    nb = json.load(open(nb_path, encoding='utf-8'))
    for c in nb['cells']:
        src = ''.join(c['source'])
        if src.startswith('SUPPORT_SOURCE = '):
            code = ast.literal_eval(src.splitlines()[0][len('SUPPORT_SOURCE = '):])
            mod = type(sys)('grpo_support_old')
            exec(compile(code, 'grpo_support_old', 'exec'), mod.__dict__)
            return mod
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--priv', default=os.environ.get('MMDL_PRIV', str(REPO / 'mutimodal_private')))
    a = ap.parse_args()
    priv = Path(a.priv); pool_dir = priv / 'virl39k/pool_for_colab'; rft = priv / 'VIRL39K_RFT_4B'
    cands = load_jsonl(pool_dir / 'candidates.jsonl')
    Q = {q['uid']: q for q in cands}
    gens = {g['uid']: g for g in load_jsonl(rft / 'virl_generations_PRIVATE.jsonl')}
    st3 = {s['uid']: s for s in load_jsonl(rft / 'virl_status_v3_PRIVATE.jsonl')}
    sel = {s['uid']: s for s in load_jsonl(rft / 'virl_selected_PRIVATE.jsonl')}
    split = notebook_split(cands)
    assert all(split[u] == s['split'] for u, s in sel.items()), '선별 파일의 split과 노트북 split이 다릅니다'
    old = old_notebook_reward(ROOT / 'final/code/finetune/train/ViRL39K_GRPO_4B_BF16_LoRA_A100.ipynb')  # 저장소에 없음(초기 노트북, 선택 비교)

    # ---------- 1. 보상 검증 ----------
    items, disagree = [], []
    agree_v3 = C.Counter(); old_cmp = C.Counter(); methods = C.Counter()
    for uid, q in Q.items():
        g = gens[uid]
        rs, toks, tr, meth, mk = [], [], [], [], []
        for s in g['samples']:
            sc = score(s['text'], s['finish_reason'], q['kind'], q['gold'], q['choices'])
            rs.append(int(sc['reward'])); toks.append(int(s['output_tokens'])); tr.append(sc['truncated']); meth.append(sc['method']); mk.append(sc['mask'])
            methods[(q['kind'], sc['method'] + ('|masked' if sc['mask'] else ''), int(sc['reward']))] += 1
            if old is not None:
                o = int(old.score_text(s['text'], q['gold'], q['kind'], q['choices'],
                                       terminated=s['finish_reason'] == 'stop', symbolic=True)['reward'])
                old_cmp[(q['kind'], o, rs[-1])] += 1
                if o != rs[-1] and len(disagree) < 4000:
                    disagree.append({'uid': uid, 'kind': q['kind'], 'gold': q['gold'], 'old': o, 'new': rs[-1],
                                     'method': sc['method'], 'tail': s['text'][-300:]})
        k = sum(rs); agree_v3[(q['kind'], k == st3[uid]['n_ok_v3'])] += 1
        lo, hi = ACTIVE[q['kind']]
        um = [r for r, m in zip(rs, mk) if not m]          # 마스크 안 된 샘플만으로 보상이 엇갈려야 학습 가능
        trainable = len(um) >= 2 and 0 < sum(um) < len(um)
        items.append({'uid': uid, 'kind': q['kind'], 'category': q['category'], 'split': split[uid],
                      'image': q['image'], 'question_norm': ' '.join(str(q['question']).casefold().split()),
                      'k': k, 'rewards': rs, 'mask': mk, 'tokens': toks, 'truncated': tr, 'input_tokens': g.get('input_tokens'),
                      'gold_ok': gold_self_check(q['kind'], q['gold'], q['choices']), 'trainable': trainable,
                      'status_v3': st3[uid]['status_v3'], 'n_ok_v3': st3[uid]['n_ok_v3'],
                      'active': lo <= k <= hi,
                      'in_old_notebook_set': uid in sel and sel[uid]['split'] == 'train' and 0 < sel[uid]['n_correct'] < 4})

    # MMMU baseline 객관식: 잘리지 않은 응답에서 보상 == v2 정오인지
    base = {json.loads(l)['id']: json.loads(l) for l in open(ROOT / 'code/qwen3_vl_mmmu_eval/outputs/final_seed3407_16k/scored.jsonl')}
    import csv
    v2 = {r['id']: int(r['baseline_correct']) for r in csv.DictReader(open(
        Path(os.environ.get('MMDL_BASELINE_V2_CSV', REPO / 'MMDL/pmc_vqa/eval/mmmu_eval_outputs/virl39k_shortest_100pct_seed3407_16k_a100_40gb_v2/per_question_vs_baseline.csv'))))}  # baseline 문항별 정오: 저장소에 없음(로컬 작업 폴더), MMDL_BASELINE_V2_CSV로 지정
    mmmu = C.Counter()
    for i, r in base.items():
        if r['question_type'] != 'multiple-choice':
            continue
        ch = r['scoring_choices']; ch = ch if isinstance(ch, dict) else ast.literal_eval(ch)
        trunc = str(r['truncated']) == 'True'
        rw = int(score(r['response'], 'length' if trunc else 'stop', 'mc', r['answer'], [ch[x] for x in sorted(ch)])['reward'])
        mmmu[('trunc' if trunc else 'ok', v2[i], rw)] += 1
    mmmu_nt = {k: v for k, v in mmmu.items() if k[0] == 'ok'}
    assert all(k[1] == k[2] for k in mmmu_nt), f'잘리지 않은 MMMU 객관식에서 보상과 v2가 다릅니다: {mmmu_nt}'

    # ---------- 2. 문항 목록 재구성 ----------
    val = [it for it in items if it['split'] == 'val']
    val_img = {sha(pool_dir / it['image']) for it in val}; val_q = {it['question_norm'] for it in val}
    train_active = []
    overlap = 0; gold_fail = 0; not_trainable = 0
    for it in items:
        if it['split'] != 'train' or not it['active']:
            continue
        if not it['gold_ok']:
            gold_fail += 1; it['excluded'] = 'gold_self_check'; continue
        if not it['trainable']:
            not_trainable += 1; it['excluded'] = 'masked_no_contrast'; continue
        if sha(pool_dir / it['image']) in val_img or it['question_norm'] in val_q:
            overlap += 1; it['excluded'] = 'val_overlap'; continue
        it['round1'] = True; train_active.append(it)
    long_ok = lambda it: any(r and t > 4096 for r, t in zip(it['rewards'], it['tokens']))
    for it in train_active:
        it['tags'] = [t for t, c in (('long_correct', long_ok(it)), ('was_no_valid_target', it['status_v3'].endswith('no_valid_target')),
                                     ('has_truncated', any(it['truncated']))) if c]
    old_set = {it['uid'] for it in items if it['in_old_notebook_set']}
    new_set = {it['uid'] for it in train_active}

    # 1라운드 샘플(advantage = r - 그룹 평균, Dr.GRPO: 표준편차로 나누지 않음)
    samp = C.Counter(); comp_tokens = 0; prompt_tokens = 0
    for it in train_active:
        um = [r for r, mm in zip(it['rewards'], it['mask']) if not mm]; m = sum(um) / len(um)
        for r, t, tr, mm in zip(it['rewards'], it['tokens'], it['truncated'], it['mask']):
            if mm:
                samp['masked'] += 1; continue
            samp['pos' if r - m > 0 else 'neg'] += 1
            if tr: samp['neg_truncated'] += 1
            comp_tokens += t; prompt_tokens += it['input_tokens'] or 0

    def table(rows, key):
        return dict(sorted(C.Counter(key(r) for r in rows).items()))

    report = {
        'reward_version': REWARD_VERSION, 'date': '2026-11-19',
        'reward_checks': {
            'reproduces_v3_item_counts': {f'{k}_{ok}': v for (k, ok), v in agree_v3.items()},
            'mmmu_baseline_mc_reward_vs_v2': {f'{t}|v2={a}|reward={b}': v for (t, a, b), v in sorted(mmmu.items())},
            'old_notebook_vs_new_samples': {f'{k}|old={o}|new={n}': v for (k, o, n), v in sorted(old_cmp.items())},
            'methods': {f'{k}|{m}|r={r}': v for (k, m, r), v in sorted(methods.items())},
        },
        'items': {
            'pool': len(items), 'val': len(val), 'train': sum(it['split'] == 'train' for it in items),
            'round1_active_train': len(train_active), 'excluded_val_overlap': overlap,
            'excluded_gold_self_check': gold_fail, 'excluded_masked_no_contrast': not_trainable,
            'gold_self_check_fail_all_pool': sum(not it['gold_ok'] for it in items),
            'old_notebook_set': len(old_set), 'kept_from_old': len(old_set & new_set),
            'dropped_from_old': len(old_set - new_set), 'added_vs_old': len(new_set - old_set),
            'dropped_from_old_by_k': table([it for it in items if it['uid'] in old_set - new_set], lambda r: f"{r['kind']}_k{r['k']}"),
            'added_by_status_v3': table([it for it in train_active if it['uid'] in new_set - old_set], lambda r: r['status_v3']),
            'by_kind': table(train_active, lambda r: r['kind']),
            'by_category': table(train_active, lambda r: r['category']),
            'old_by_category': table([it for it in items if it['uid'] in old_set], lambda r: r['category']),
            'by_k': table(train_active, lambda r: f"{r['kind']}_k{r['k']}"),
            'tags': dict(C.Counter(t for it in train_active for t in it['tags'])),
            'not_active_train_by_k': table([it for it in items if it['split'] == 'train' and not it['active']], lambda r: f"{r['kind']}_k{r['k']}"),
        },
        'round1_samples': {
            'samples': len(train_active) * 4, 'masked_mc_conflict': samp['masked'], 'positive_adv': samp['pos'], 'negative_adv': samp['neg'],
            'negative_truncated': samp['neg_truncated'],
            'completion_tokens': comp_tokens, 'prompt_tokens_incl_image': prompt_tokens,
            'completion_tokens_median': S.median(t for it in train_active for t in it['tokens']),
        },
    }
    out = ROOT / 'final/results/grpo_training'; out.mkdir(parents=True, exist_ok=True)
    (out / 'grpo_prep_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with open(rft / 'grpo_items_PRIVATE.jsonl', 'w', encoding='utf-8') as f:
        for it in items:
            it = {k: v for k, v in it.items() if k not in ('question_norm',)}
            f.write(json.dumps(it, ensure_ascii=False) + '\n')
    with open(rft / 'grpo_reward_disagree_PRIVATE.jsonl', 'w', encoding='utf-8') as f:
        for d in disagree:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
