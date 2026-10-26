"""Regression tests for real-dimension presentation curves."""
import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab1"))

from scale_curve_research import GRIDS, build_summary, collect_curves, plot_curves

class CurveResearchTests(unittest.TestCase):
    def test_grids_have_measurements_inside_each_class(self):
        self.assertEqual(len(GRIDS["medium"]), 11)
        self.assertEqual(GRIDS["medium"][0], 100)
        self.assertEqual(GRIDS["medium"][-1], 500)
        self.assertEqual(GRIDS["large"][0], 1000)
        self.assertEqual(GRIDS["large"][-1], 2000)
        self.assertEqual(len(set(GRIDS["large"])), 11)

    def test_smoke_has_actual_paired_runs_not_fictitious_interpolation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            saved_params = Path("results/scale_benchmark_full/best_parameters.json")
            rows = collect_curves(saved_params, root, mode="smoke", repeats=1, workers=2, seed=23)
            self.assertEqual(len(rows), 12)  # 3 classes x 2 sizes x 2 methods
            self.assertTrue(all(r["size"] < 100 for r in rows))
            for a, b in zip(rows[0::2],rows[1::2]):
                self.assertEqual(a["matrix_seed"], b["matrix_seed"])
                self.assertEqual(a["total_population"], b["total_population"])
                self.assertEqual(a["exact_cost"], b["exact_cost"])
            self.assertTrue((root/"raw.csv").exists())
            self.assertEqual(len(build_summary(rows)), 12)
            plot_curves(build_summary(rows), root/"plots", mode="smoke")
            self.assertEqual(len(list((root/"plots").glob("*.png"))), 9)
            self.assertGreater(min(p.stat().st_size for p in (root/"plots").glob("*.png")), 20000)
            report=json.loads((root/"metadata.json").read_text())
            self.assertEqual(report["mode"],"smoke")
            self.assertEqual(report["size_grids"]["medium"],[7,8])

if __name__ == "__main__":
    unittest.main()
