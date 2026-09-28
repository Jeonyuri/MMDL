import pytest
from mmmu_eval.parse import can_infer_option, extract_answer

CHOICES = {"A": "red", "B": "blue", "C": "green", "D": "yellow"}


@pytest.mark.parametrize("response,expected,method", [
    ("B", "B", "option"), ("(C).", "C", "option"),
    ("The answer is B.", "B", "option"),
    ("First compute the ratio. Therefore the answer is D.", "D", "option"),
    ("B and C are plausible. Answer: C", "C", "regex"),
    ("A and B fail. answer is (D)", "D", "regex"),
    ("A useful observation follows", None, "unparsed"),
    ("A useful observation. Answer: B", "B", "regex"),
    ("It is BLUE", "B", "text"), ("red or blue", None, "unparsed"),
    ("Cannot determine the answer. B", "Z", "refused"),
    ("I can't process this file.", "Z", "refused"),
    ("", None, "unparsed"), ("b", None, "unparsed"),
    ("A B answer: C then Answer: D", "D", "regex"),
    ("B C ultimately **D**", "D", "regex"),
    ("B B B", "B", "option"), ("Z", "Z", "refused"),
    ("No idea", None, "unparsed"), ("B C Answer: I", None, "unparsed"),
    ("B C Answer: Blue", "B", "text"),
])
def test_extract(response, expected, method):
    assert extract_answer(response, CHOICES) == {"extracted": expected, "method": method}


def test_no_choice_mutation():
    choices = {"A": "RED", "B": "BLUE"}
    extract_answer("blue", choices)
    assert choices == {"A": "RED", "B": "BLUE"}


def test_api_failure_upstream():
    assert can_infer_option("Failed to obtain answer via API B", CHOICES) is False
