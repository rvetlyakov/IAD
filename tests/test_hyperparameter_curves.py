"""All-class, per-hyperparameter plots with measured (not invented) points."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab1"))
from hyperparameter_curves import (
    HYPERPARAMETERS,
    REAL_SIZES,
    axis_position,
    run_experiment,
    run_plateau_variant,
)
from scale_benchmark import generate_matrix


class HyperparameterCurvesTests(unittest.TestCase):
    def test_every_chart_contains_all_requested_size_classes(self):
        self.assertEqual(REAL_SIZES["medium"], (100, 200, 300, 400, 500))
        self.assertEqual(REAL_SIZES["large"], (1000, 1250, 1500, 1750, 2000))
        self.assertEqual(REAL_SIZES["small"], (10, 20, 30, 40, 50))
        self.assertEqual(len(HYPERPARAMETERS), 10)
        self.assertIn("num_islands", HYPERPARAMETERS)
        self.assertIn("plateau_diff", HYPERPARAMETERS)

    def test_axis_uses_equal_space_for_three_disparate_size_ranges(self):
        x1=axis_position("small",10)
        x2=axis_position("medium",100)
        x3=axis_position("large",1000)
        self.assertLess(x1,x2)
        self.assertLess(x2,x3)
        self.assertGreater(x2-x1,1)
        self.assertGreater(x3-x2,1)

    def test_smoke_honestly_labels_tiny_matrix_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=run_experiment(
                mode="smoke", parameters=("mutation_probability","num_islands","plateau_diff"),
                output_dir=Path(tmp), repeats=1, workers=2, seed=701
            )
            self.assertEqual(report["mode"],"smoke")
            self.assertEqual(report["parameters"],["mutation_probability","num_islands","plateau_diff"])
            self.assertTrue((Path(tmp)/"raw.csv").exists())
            self.assertTrue((Path(tmp)/"metadata.json").exists())
            self.assertEqual(len(list((Path(tmp)/"plots").glob("*.png"))),3)
            self.assertTrue(all(row["size"]<100 for row in report["rows"]))
            self.assertEqual({row["class"] for row in report["rows"]},set(REAL_SIZES))
            self.assertGreater(len(report["rows"]),10)

    def test_plateau_is_not_silent_noop(self):
        matrix=generate_matrix(7,42)
        settings={"total_population": 12,"mutation_probability": .7,"mutation_percent": 30,
                      "elitist_percent": .2,"max_generations": 20,
                      "plateau_limit": 2,"plateau_diff": 2.0}
        early=run_plateau_variant(matrix,settings,seed=101)
        self.assertLess(early["generations"],20)
        settings["plateau_diff"]=0.
        strict=run_plateau_variant(matrix,settings,seed=101)
        self.assertLessEqual(strict["generations"],20)
        self.assertGreaterEqual(strict["cost"],7)

if __name__=="__main__":
    unittest.main()
