"""MMMU-Pro official prompts and answer parser, vendored.

Source: https://github.com/MMMU-Benchmark/MMMU (Apache-2.0), commit
b26bdefc7e5e007962fde8f93b2bb53c99c8d13d, files mmmu-pro/prompts.yaml and
mmmu-pro/evaluate.py (functions parse_multi_choice_response, get_multi_choice_info).

Only change: the final "no candidate -> random.choice(all_choices)" step takes an
explicit `rng`. rng=None returns None (no guess) for our primary metric;
rng=random.Random(42) reproduces the official behaviour (evaluate.py calls
random.seed(42) once and guesses in file order).
"""
import numpy as np

COT_PROMPTS = {
    "vision": "Write out the multiple-choice question in the image and then solve it. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering.",
    "standard": "Answer the preceding multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering.",
}


def get_multi_choice_info(options):
    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))
    return index2ans, all_choices


def parse_options(options):
    """infer/infer_*.py parse_options: 'A. opt' lines joined by newlines."""
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    return "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])


def parse_multi_choice_response(response, all_choices, index2ans, rng=None):
    last_answer_pos = response.rfind("Answer:")
    if last_answer_pos != -1:
        answer_str = response[last_answer_pos + len("Answer:"):].strip()
        matching_options = [option for option in all_choices if option in answer_str]
        if len(matching_options) == 1:
            return matching_options[0]

    if isinstance(response, str):
        for char in [",", ".", "!", "?", ";", ":", "'"]:
            response = response.strip(char)
        response = " " + response + " "
    else:
        print(response)
        response = ""

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if len(candidates) == 0:
        pred_index = None if rng is None else rng.choice(all_choices)   # (changed: was random.choice(all_choices))
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        pred_index = candidates[0]

    return pred_index
