"""주관식 정답 판정 v3 (2026-11-18, teacher 실험용). 기존 `open_answer_scoring.py`(제출 모델 학습 데이터에 쓴 규칙)는 그대로 두고 그 위에 규칙만 고친다.

기존 판정에서 확인된 문제 (0/4 문항 사전 분석, CLAUDE.md §10):
  - 거짓 오답: 숫자 뒤에 단위·설명이 붙은 답(`$21.29 billion` vs 21.29), `Yes, ...`처럼 설명이 붙은 Yes/No,
    백분율 표기(`7.66%` vs 0.0766), 부등식↔구간(`(1, +\\infty)` vs `x > 1`), 정답이 식으로 적힌 경우(`4 - 3 = 1`).
  - 거짓 정답: 상대 허용오차 0.1%를 정수에도 적용해 연도가 1~2년 틀려도 정답 처리(2016 vs 2015).

v3 규칙 = 기존 판정 + 아래 5개 추가 인정 − 정수 허용오차 버그 제거. 객관식은 바꾸지 않는다(v2 파서 그대로).
로컬 테스트: test_open_answer_scoring_v3.py. 노트북에서는 기존 셀 뒤에 이 파일 내용을 붙여 `match_open`을 덮어쓴다."""
import re
from open_answer_scoring import _norm, _strip_math, _strip_units, _to_float, match_open as match_open_v1, open_candidates

_NUM = r'-?\d[\d,]*(?:\.\d+)?'
_INT_RE = re.compile(r'-?\d[\d,]*')
_SCALE_OR_WORDS = r'[A-Za-z][A-Za-z\s\-/\.\']*'

def _num(s):
    try: return float(str(s).replace(',', ''))
    except ValueError: return None

def _same_number(a, b, a_txt, b_txt):
    """둘 다 정수 표기면 정확히 같아야 하고, 아니면 기존과 같은 상대 0.1% 허용."""
    if a is None or b is None: return False
    if _INT_RE.fullmatch(a_txt.replace(' ', '')) and _INT_RE.fullmatch(b_txt.replace(' ', '')):
        return a == b
    return abs(a - b) <= max(1e-6, 1e-3 * abs(b))

def _int_tolerance_bug(pred, gold):
    """기존 판정이 정수끼리 허용오차로 맞다고 한 경우(예: 2016 vs 2015)."""
    pu, gu = _strip_units(pred), _strip_units(gold)
    if _INT_RE.fullmatch(pu) and _INT_RE.fullmatch(gu):
        return _num(pu) != _num(gu)
    return False

def _gold_number(gold):
    g = _strip_math(gold).strip().rstrip('.')
    return g if re.fullmatch(_NUM, g) else None

def _rule_number_words(pred, gold):
    """정답이 숫자 하나이고, 답이 '그 숫자 + 단위·규모·설명 단어'(다른 숫자 없음)인 경우."""
    g = _gold_number(gold)
    if g is None: return False
    p = _strip_math(pred).strip().strip('*').strip().rstrip('.')
    m = re.fullmatch(r'\$?\s*(' + _NUM + r')\s*(' + _SCALE_OR_WORDS + r')?', p)
    if not m or not m.group(2): return False          # 단어가 없으면 기존 판정 몫
    if re.search(r'\d', m.group(2)): return False
    return _same_number(_num(m.group(1)), _num(g), m.group(1), g)

def _rule_yes_no(pred, gold):
    g = _norm(gold)
    if g not in ('yes', 'no'): return False
    p = _strip_math(pred).strip().lstrip('*').strip().casefold()
    m = re.match(r'(yes|no)\b', p)
    return bool(m) and m.group(1) == g and len(p) <= 200

def _rule_percent(pred, gold):
    """백분율 ↔ 소수: 한쪽에 %가 명시되어 있고 나머지가 그 값/100인 경우만."""
    p, g = _strip_math(pred).strip().strip('*').rstrip('.'), _strip_math(gold).strip().rstrip('.')
    mp = re.fullmatch(r'(' + _NUM + r')\s*%\s*(?:' + _SCALE_OR_WORDS + r')?', p)
    mg = re.fullmatch(r'(' + _NUM + r')\s*%', g)
    if mp and re.fullmatch(_NUM, g):
        return _same_number(_num(mp.group(1)) / 100, _num(g), 'x.0', g)
    if mg and re.fullmatch(_NUM, p):
        return _same_number(_num(p) * 100, _num(mg.group(1)), 'x.0', mg.group(1))
    return False

_INF = r'(?:\+?\s*\\?infty|\+?\s*∞|\+?\s*inf(?:inity)?)'
_NINF = r'(?:-\s*\\?infty|-\s*∞|-\s*inf(?:inity)?)'

def _as_interval(s):
    """'x > a' 꼴 부등식이나 '(a, +∞)' 꼴 구간을 (lo, lo_closed, hi, hi_closed)로. 못 읽으면 None."""
    t = _strip_math(s).replace('\\geq', '≥').replace('\\ge', '≥').replace('\\leq', '≤').replace('\\le', '≤')
    t = t.replace('>=', '≥').replace('<=', '≤').strip().rstrip('.')
    m = re.fullmatch(r'[a-z]\s*([<>≤≥])\s*(' + _NUM + r')', t)
    if m:
        op, a = m.group(1), _num(m.group(2))
        return {'>': (a, False, None, False), '≥': (a, True, None, False),
                '<': (None, False, a, False), '≤': (None, False, a, True)}[op]
    m = re.fullmatch(r'([\(\[])\s*(' + _NUM + r')\s*,\s*' + _INF + r'\s*\)', t)
    if m: return (_num(m.group(2)), m.group(1) == '[', None, False)
    m = re.fullmatch(r'\(\s*' + _NINF + r'\s*,\s*(' + _NUM + r')\s*([\)\]])', t)
    if m: return (None, False, _num(m.group(1)), m.group(2) == ']')
    return None

def _rule_interval(pred, gold):
    a, b = _as_interval(pred), _as_interval(gold)
    return a is not None and a == b

def _rule_gold_equation(pred, gold):
    """정답이 '4 - 3 = 1'처럼 산술식 = 값으로 적힌 경우 마지막 값과 비교(좌변에 문자가 없을 때만)."""
    g = _strip_math(gold)
    if g.count('=') != 1: return False
    lhs, rhs = g.split('=')
    if re.search(r'[A-Za-z\\]', lhs) or not re.search(r'\d', lhs): return False
    return match_open(pred, rhs.strip())

RULES = (('number_words', _rule_number_words), ('yes_no', _rule_yes_no), ('percent', _rule_percent),
         ('interval', _rule_interval), ('gold_equation', _rule_gold_equation))

def match_open_why(pred, gold):
    """(정답 여부, 근거). 근거: 'v1' | 추가 규칙 이름 | 'int_fix'(기존엔 정답이었으나 정수 버그로 취소) | None."""
    if pred is None: return False, None
    if match_open_v1(pred, gold):
        if _int_tolerance_bug(pred, gold): return False, 'int_fix'
        return True, 'v1'
    for name, rule in RULES:
        try:
            if rule(pred, gold): return True, name
        except (ValueError, ZeroDivisionError, TypeError):
            continue
    return False, None

def match_open(pred, gold):
    return match_open_why(pred, gold)[0]

def match_response(response, gold):
    """기존과 같은 후보 추출, 판정만 v3. (후보 없음 -> (False, 'open_unparsed'))"""
    cands = open_candidates(response)
    if not cands: return False, 'open_unparsed'
    return any(match_open(c, gold) for c in cands), 'open_checked'
