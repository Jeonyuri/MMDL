"""Conservative fallback on top of the Qwen rule parser (parse.py).

parse.extract_answer runs first and its result is kept whenever it is not
"unparsed". Only when it fails, the response finished naturally (not truncated)
and the question is multiple-choice do the explicit-declaration patterns below
run; the last valid match wins. Truncated responses and open questions are left
unparsed so the fallback cannot turn mid-reasoning mentions into answers.
"""
import re

from .parse import extract_answer

_L = r"([A-I])"
PATTERNS = (
    # **C. Philomena**, **B) decrease**, **A: text
    re.compile(r"\*\*\s*\(?" + _L + r"\s*[\.\):]"),
    # **Option A**, **Answer: B**, **(C)**  (no adjacent optional \s*: avoids quadratic backtracking)
    re.compile(r"\*\*\s*(?:(?i:option|choice|answer)\s*:?\s*)?\(?" + _L + r"\)?\s*(?:[\.\):]\s*)?\*\*"),
    # final answer is B / Correct Answer: C / answer: (D)
    re.compile(r"(?i:\b(?:final\s+answer|correct\s+answer|answer)\b)(?:\s+(?i:is)\b)?[\s:\*\(\[]*" + _L + r"(?![A-Za-z0-9])"),
    # \boxed{B}, \boxed{(B)}, \boxed{\text{B}}
    re.compile(r"\\boxed\{\s*(?:\\text\{\s*)?\(?" + _L + r"\)?[\.\}\s]"),
)


def fallback_letter(response, choices):
    hits = []
    for pattern in PATTERNS:
        for match in pattern.finditer(response):
            if match.group(1) in choices:
                hits.append((match.end(), match.group(1)))
    return max(hits)[1] if hits else None


def extract_answer_v2(response, choices, regex, question_type, truncated):
    parsed = extract_answer(response, choices, regex)
    if parsed["method"] != "unparsed" or truncated or question_type != "multiple-choice":
        return parsed
    letter = fallback_letter(response, choices)
    if letter:
        return {"extracted": letter, "method": "fallback"}
    return parsed
