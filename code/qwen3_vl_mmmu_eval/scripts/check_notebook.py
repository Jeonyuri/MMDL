"""Offline JSON and Python syntax check for the Colab notebook (no execution)."""
import json
from pathlib import Path


def main():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "colab_run.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell["source"]), f"{path}:cell{index}", "exec")
    print("Notebook JSON and Python cell syntax: OK (not executed)")


if __name__ == "__main__":
    main()
