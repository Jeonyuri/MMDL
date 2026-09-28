from mmmu_eval.prompt import build_messages, render_text


def test_multiple_choice_exact():
    assert render_text("What is <image 1>?", ["cat", "dog"]) == (
        "Question: What is <image 1>?\nOptions:\nA. cat\nB. dog\n"
        "Please select the correct answer from the options above.")


def test_open_exact():
    assert render_text("Read <image 2>.", []) == "Question: Read <image 2>."


def test_image_order_and_pixels():
    images = [object(), object()]
    messages = build_messages({"question": "Q", "options": [], "images": images}, 1003520, 4014080)
    assert len(messages) == 1 and messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert [x["image"] for x in content[:-1]] == images
    assert all(x["min_pixels"] == 1003520 and x["max_pixels"] == 4014080 for x in content[:-1])
    assert content[-1] == {"type": "text", "text": "Question: Q"}


def test_nine_options():
    assert "I. 8\n" in render_text("Q", list(map(str, range(9))))
