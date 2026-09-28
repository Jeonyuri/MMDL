import pytest
from PIL import Image

from mmmu_eval.data import SUBJECTS, load_data, normalize_row, parse_options


def test_options_literal_and_images():
    row = {"id": "x", "question": "<image 1> + <image 3>", "options": "['one', 'two']",
           "answer": "B", "question_type": "multiple-choice", "image_1": Image.new("L", (2, 2)),
           "image_2": None, "image_3": Image.new("RGBA", (3, 3))}
    result = normalize_row(row, "Art")
    assert result["question"] == row["question"]
    assert [image.size for image in result["images"]] == [(2, 2), (3, 3)]
    assert all(image.mode == "RGB" for image in result["images"])


@pytest.mark.parametrize("value", ["__import__('os').getcwd()", "{}", "[1]", str(list("ABCDEFGHIJ"))])
def test_invalid_options(value):
    with pytest.raises((ValueError, SyntaxError)):
        parse_options(value)


class FakeDataset:
    def __init__(self, subject, n=30):
        self.rows = [{"id": f"{subject}_{i}", "question_type": "open", "question": "Q",
                      "options": "[]", "answer": "42"} for i in range(n)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        return [row[key] for row in self.rows] if isinstance(key, str) else self.rows[key]

    def __iter__(self):
        return iter(self.rows)


def test_load_all_configs_and_counts():
    calls = []
    def loader(path, subject, **kwargs):
        calls.append((path, subject, kwargs))
        return FakeDataset(subject)
    config = {"subjects": SUBJECTS, "dataset_path": "MMMU/MMMU", "split": "validation", "dataset_revision": "pinned"}
    datasets, stats = load_data(config, "cache", loader)
    assert list(datasets) == SUBJECTS and stats["total"] == 900
    assert stats["question_type"] == {"open": 900}
    assert all(call[2] == {"split": "validation", "revision": "pinned", "cache_dir": "cache"} for call in calls)


def test_count_failure():
    config = {"subjects": SUBJECTS, "dataset_path": "MMMU/MMMU", "split": "validation", "dataset_revision": "pinned"}
    with pytest.raises(ValueError, match="expected 30"):
        load_data(config, "cache", lambda *args, **kwargs: FakeDataset("Art", 29))
