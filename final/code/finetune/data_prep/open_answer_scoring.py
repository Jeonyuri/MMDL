"""RFT 정답 판정(ViRL39K 풀). 객관식은 평가와 같은 v2 파서(노트북에서 주입), 주관식은 아래 규칙.
이 파일의 내용은 Colab 노트북 셀에 그대로 복사된다(노트북 독립 실행 가능하도록). 로컬 테스트: test_open_answer_scoring.py."""
import re, math, unicodedata

_UNIT = r'(?:square\s+)?(?:cm|mm|km|m|kg|g|mg|s|sec|seconds?|min|minutes?|hours?|hrs?|h|days?|years?|units?|degrees?|meters?|metres?|centimeters?|cm\^?2|cm\^?3|m\^?2|m\^?3|n|j|w|v|a|hz|pa|mol|l|ml|yuan|dollars?|cents?|pieces?|people|persons?|times?)'

def _strip_math(s):
    s = str(s)
    s = re.sub(r'\\text\s*\{([^{}]*)\}', r'\1', s); s = re.sub(r'\\mathrm\s*\{([^{}]*)\}', r'\1', s)
    s = s.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac').replace('\\left', '').replace('\\right', '')
    s = s.replace('\\,', '').replace('\\!', '').replace('\\ ', ' ').replace('\\%', '%').replace('\\$', '')
    s = s.replace('^{\\circ}', '°').replace('^\\circ', '°').replace('\\circ', '°').replace('$', '')
    s = s.replace('π', '\\pi').replace('√', '\\sqrt')
    s = unicodedata.normalize('NFKC', s)
    return s.strip()

def _norm(s):
    s = _strip_math(s).casefold().strip(' .,;:!?*"\'')
    s = re.sub(r'^[a-z]\s*=\s*', '', s)                      # "x = 5" -> "5"
    if s.startswith('{') and s.endswith('}') and '{' not in s[1:-1]: s = s[1:-1]
    if s.startswith('(') and s.endswith(')') and ',' not in s:          # (4n+2) -> 4n+2 (좌표·구간은 쉼표가 있어 제외)
        depth = 0
        for k, ch in enumerate(s):
            depth += (ch == '(') - (ch == ')')
            if depth == 0 and k < len(s) - 1: break
        else: s = s[1:-1]
    return re.sub(r'\s+', '', s)

def _strip_units(s):
    s = _norm(s)
    s = re.sub(r'(?<=[\d\}])\s*' + _UNIT + r'$', '', s)
    s = s.replace('°', '').replace('degrees', '').replace('degree', '').replace('%', '')
    s = s.replace('(', '(').strip()
    return s

def _to_float(s):
    t = _strip_units(s).replace(',', '') if re.fullmatch(r'[\d,]+(\.\d+)?', _strip_units(s)) else _strip_units(s)
    t = re.sub(r'\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}', r'(\1)/(\2)', t)
    t = re.sub(r'\\sqrt\s*\{([^{}]+)\}', r'sqrt(\1)', t); t = re.sub(r'\\sqrt\s*(\d+)', r'sqrt(\1)', t)
    t = t.replace('\\pi', '*pi').replace('\\cdot', '*').replace('\\times', '*').replace('^', '**')
    t = re.sub(r'(\d)(\(|sqrt|pi)', r'\1*\2', t); t = re.sub(r'\)(\d|\(|sqrt|pi)', r')*\1', t)
    if not t or not re.fullmatch(r'[\d\.\+\-\*/\(\)sqrtpi]+', t) or t.count('(') != t.count(')'): return None
    try:
        v = eval(t, {'__builtins__': {}}, {'sqrt': math.sqrt, 'pi': math.pi})   # 문자 집합을 위 정규식으로 제한
        return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None
    except Exception:
        return None

def _boxed(text):
    out = []
    for m in re.finditer(r'\\boxed\s*\{', text):
        i, depth = m.end(), 1
        while i < len(text) and depth:
            depth += (text[i] == '{') - (text[i] == '}'); i += 1
        if depth == 0: out.append(text[m.end():i - 1])
    return out[-1].strip() if out else None

def _clean_cand(t):
    t = str(t).strip()
    t = re.sub(r'^[>\s]+', '', t); t = t.replace('✅', '').replace('\\[', '').replace('\\]', '').strip()
    t = re.sub(r'^\*+|\*+$', '', t.strip()).strip()
    return t

def extract_open(response):
    """명시적 표지가 있는 최종 답 후보: \\boxed 마지막 > 'Answer:' 마지막 줄 > 'the answer is'. 없으면 None."""
    b = _boxed(response)
    if b: return b
    ms = list(re.finditer(r'(?:final\s+answer|answer)\s*(?:is\s*)?[:：]\s*\**\s*(.+)', response, flags=re.I))
    if ms:
        t = ms[-1].group(1).strip().strip('*').strip()
        t = re.sub(r'\s*\*+\s*$', '', t)
        return t.split('\n')[0].strip() if t else None
    ms = list(re.finditer(r'the\s+(?:final\s+)?answer\s+is\s*[:：]?\s*\**\s*([^\n]+?)\s*\**\s*(?:\.\s|\.$|$)', response, flags=re.I | re.M))
    if ms: return ms[-1].group(1).strip().strip('*').strip()
    return None

def open_candidates(response):
    """채점에 쓸 후보 목록. 표지가 있으면 그 답(과 그 안의 굵은 조각들)만, 없으면 끝부분의 굵은 글씨 / 마지막 짧은 줄을 후보로 쓴다."""
    ex = extract_open(response)
    if ex:
        c = [_clean_cand(ex)]
        parts = [p for p in re.split(r'\*\*', ex) if p.strip() and p.strip().lower() not in ('or', 'and', ',', '=')]
        c += [_clean_cand(p) for p in parts if len(parts) > 1]
        return [x for x in dict.fromkeys(c) if x]
    tail = response[-400:]
    bold = re.findall(r'\*\*([^*\n]{1,80})\*\*', tail)
    cand = [_clean_cand(bold[-1])] if bold else []
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    if lines:
        last = _clean_cand(lines[-1]).strip('. ')
        if 0 < len(last) <= 40 and re.search(r'[\d\w]', last): cand.append(last)
    return [x for x in dict.fromkeys(cand) if x]

def match_response(response, gold):
    """응답에서 후보 답들을 뽑아 하나라도 gold와 맞으면 정답. (후보 없음 -> (False, 'open_unparsed'))"""
    cands = open_candidates(response)
    if not cands: return False, 'open_unparsed'
    return any(match_open(c, gold) for c in cands), 'open_checked'

def _verify_symbolic(pred, gold):
    try:
        from math_verify import parse, verify
        return bool(verify(parse('$' + _strip_math(gold) + '$'), parse('$' + _strip_math(pred) + '$')))
    except Exception:
        return False

def match_open(pred, gold):
    if pred is None: return False
    pn, gn = _norm(pred), _norm(gold)
    if not pn or not gn: return False
    if pn == gn: return True
    pu, gu = _strip_units(pred), _strip_units(gold)
    if pu == gu: return True
    pf, gf = _to_float(pred), _to_float(gold)
    if pf is not None and gf is not None:
        return abs(pf - gf) <= max(1e-6, 1e-3 * abs(gf))
    if re.fullmatch(r'[\(\[].*[\)\]]', gu) or re.fullmatch(r'[\(\[].*[\)\]]', pu):      # 좌표·구간: 공백만 무시
        return re.sub(r'\s+', '', pu) == re.sub(r'\s+', '', gu)
    return _verify_symbolic(pred, gold)
