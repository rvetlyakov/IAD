import json
import unittest
from pathlib import Path


class NotebookStructureTests(unittest.TestCase):
    def test_code_cells_are_valid_python(self):
        notebook = Path(__file__).resolve().parents[1] / "lab1" / "testing.ipynb"
        nb = json.loads(notebook.read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                with self.subTest(cell=i):
                    compile("".join(cell["source"]), f"cell-{i}", "exec")


if __name__ == "__main__":
    unittest.main()
