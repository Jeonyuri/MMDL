"""QwenLM/Qwen3-VL evaluation/mmmu/eval_utils.py rule semantics.

Only the judge is replaced. Retains case-sensitive refusals, the article-A
guard, distinct-choice counting, and the standalone-Z behavior upstream.
"""
import re

DEFAULT_REGEX = r"(?i:\banswer\s*(?:is\s*:?|:)\s*)\(?([A-I])\)?(?!\w)|\*\*([A-I])\*\*"
REFUSALS = (
    "Sorry, I can't help with images of people yet.",
    "I can't process this file.",
    "I'm sorry, but without the image provided",
    "Cannot determine the answer",
)


def can_infer_option(answer, choices):
    if "Failed to obtain answer via API" in answer:
        return False
    if any(err in answer for err in REFUSALS):
        return "Z"
    answer_mod = answer
    for char in ".()[],:;!*#{}":
        answer_mod = answer_mod.replace(char, " ")
    splits = [x.strip() for x in answer_mod.split()]
    candidates = [ch for ch in choices if ch in splits]
    if len(candidates) == 1:
        if "A" in splits and len(splits) > 3:
            return False
        return candidates[0]
    if not candidates and "Z" in splits:
        return "Z"
    return False


def can_infer_text(answer, choices):
    answer = answer.lower()
    assert isinstance(choices, dict)
    for key in choices:
        assert key in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        choices[key] = str(choices[key]).lower()
    candidates = [key for key, value in choices.items() if value in answer]
    return candidates[0] if len(candidates) == 1 else False


def extract_answer(answer, choices, regex=DEFAULT_REGEX):
    answer = str(answer)
    option = can_infer_option(answer, choices)
    if option:
        return {"extracted": option, "method": "refused" if option == "Z" else "option"}
    text = can_infer_text(answer, dict(choices))
    if text:
        return {"extracted": text, "method": "text"}
    matches = [next(group for group in match.groups() if group is not None)
               for match in re.finditer(regex, answer)]
    valid = [value for value in matches if value in choices]
    if valid:
        return {"extracted": valid[-1], "method": "regex"}
    return {"extracted": None, "method": "unparsed"}
