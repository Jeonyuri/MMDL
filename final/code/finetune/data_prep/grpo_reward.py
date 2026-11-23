"""GRPO 보상 v1 (2026-11-19). 평가·데이터 선별과 같은 채점기를 그대로 쓰고, RL용 안전장치만 더한다.

- 객관식: MMMU 평가와 같은 v2 파서(`mmmu_eval.parse_v2.extract_answer_v2`, `DEFAULT_REGEX`).
  잘리지 않은 응답에서는 MMMU v2 정답 판정과 정확히 같다.
- 주관식: `open_answer_scoring_v3`(후보 추출은 v1과 같고 판정만 v3, 사람이 규칙별 표본을 검토함).
  v3는 "후보 중 하나라도 맞으면 정답"이라, 답을 나열하는 꼼수(`**3** or **5**`)에 보상이 갈 수 있다.
  -> 최종 답에 'or'로 이어진 대안이 있으면 모든 대안 조각이 정답이어야 보상(답 나열 방지).
     4B 생성물 10,889개 정답 중 이 규칙에 걸린 것은 4개(모두 `1/2 or 50%` 같은 같은 값의 다른 표기).
- 객관식 파서 충돌 마스크: v2가 고른 글자와 명시적 최종 선언(`parse_v2.fallback_letter`, 마지막 "Final Answer: X"·\\boxed{X} 등)이
  서로 다르면 보상을 믿을 수 없으므로 mask=True(학습에서 advantage 0으로 제외). 4B ViRL 객관식 9,218개 중 236(2.6%), 그중 183은
  선언이 정답인데 v2가 다른 글자를 고른 경우(v2 파서의 거짓 오답). MMMU baseline 객관식은 771개 중 15(1.9%).
- 정답 자체를 이 채점기가 못 읽는 문항(예: 정답이 `>` 기호 하나)은 문항 목록에서 뺀다(`gold_self_check`).
- 잘린 응답(finish_reason != 'stop')은 항상 0. (MMMU 평가는 잘린 응답도 파싱되면 정답으로 셀 수 있음 — 의도된 차이)

math_verify가 없으면 v1의 기호 비교가 조용히 오답 처리되므로 import 시 확인한다."""
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parents[3] / 'code' / 'qwen3_vl_mmmu_eval' / 'src'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import math_verify  # noqa: F401  (v1/v3 기호 비교에 필요)
from open_answer_scoring_v3 import open_candidates, match_open
from mmmu_eval.parse import DEFAULT_REGEX
from mmmu_eval.parse_v2 import extract_answer_v2, fallback_letter

REWARD_VERSION = 'grpo-reward-v1'
LETTERS = 'ABCDEFGHI'
# 답처럼 보이는 두 조각 사이의 'or' (예: `3** or **5`, `1/2 or 50%`). 문장 속 'or'(예: "more or less")는 제외.
_OR = re.compile(r'(?:\d|\*\*|\}|\$|%)\s*(?:\*\*)?\s*\b(?i:or)\b\s*(?:\*\*)?\s*(?:\$|\\|\d|\*\*|-|\()')


def score(text, finish_reason, kind, gold, choices):
    """반환: {'reward': 0.0|1.0, 'method': str, 'truncated': bool, 'mask': bool}  (mask=True면 학습에서 제외)"""
    truncated = finish_reason != 'stop'
    if truncated:
        return {'reward': 0.0, 'method': 'truncated', 'truncated': True, 'mask': False}
    if kind == 'mc':
        ch = dict(zip(LETTERS, choices))
        p = extract_answer_v2(text, ch, DEFAULT_REGEX, 'multiple-choice', False)
        ok = p['method'] not in ('unparsed', 'refused') and p['extracted'] == gold
        decl = fallback_letter(text, ch)
        conflict = decl is not None and p['extracted'] is not None and decl != p['extracted']
        return {'reward': float(ok), 'method': 'mc_' + p['method'], 'truncated': False, 'mask': conflict}
    if kind != 'open':
        raise ValueError(f'unknown kind: {kind}')
    cands = open_candidates(text)
    if not cands:
        return {'reward': 0.0, 'method': 'open_unparsed', 'truncated': False, 'mask': False}
    oks = [match_open(c, gold) for c in cands]
    if _OR.search(cands[0]) and len(cands) > 1:
        ok = all(oks[1:])                       # 'or'로 나열된 대안 조각이 전부 정답이어야 함
        return {'reward': float(ok), 'method': 'open_or_guard', 'truncated': False, 'mask': False}
    return {'reward': float(any(oks)), 'method': 'open_v3', 'truncated': False, 'mask': False}


def gold_self_check(kind, gold, choices):
    """정답을 그대로 적은 응답이 보상 1을 받는지. 실패하면 그 문항은 학습에서 뺀다."""
    return score('Answer: ' + str(gold), 'stop', kind, gold, choices)['reward'] == 1.0
