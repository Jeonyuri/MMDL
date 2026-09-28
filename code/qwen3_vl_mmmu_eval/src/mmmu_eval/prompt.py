"""Prompt bytes follow the task's Qwen build_mmmu_prompt specification."""

DEFAULT_PROMPT = {
    "role": "user", "system": None, "content_order": "images_first",
    "question_template": "Question: {question}\n", "options_header": "Options:\n",
    "option_template": "{letter}. {opt}\n",
    "instruction": "Please select the correct answer from the options above. \n",
    "rstrip": True,
}


def render_text(question, options, spec=None):
    spec = DEFAULT_PROMPT if spec is None else spec
    if len(options) > 9:
        raise ValueError("At most nine options are supported")
    prompt = spec["question_template"].format(question=question)
    if options:
        prompt += spec["options_header"]
        for letter, opt in zip("ABCDEFGHI", options):
            prompt += spec["option_template"].format(letter=letter, opt=opt)
        prompt += spec["instruction"]
    return prompt.rstrip() if spec["rstrip"] else prompt


def build_messages(row, min_pixels, max_pixels, spec=None):
    content = [{"type": "image", "image": image, "min_pixels": min_pixels,
                "max_pixels": max_pixels} for image in row["images"]]
    content.append({"type": "text", "text": render_text(row["question"], row["options"], spec)})
    return [{"role": "user", "content": content}]
