import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab1"))
from experiments import (
    BASELINE,
    generate_cases,
    parameter_settings,
    run_study,
    save_study,
    summarize,
)


class ExperimentsTests(unittest.TestCase):
    def test_cases_reproducible(self):
        a = generate_cases(2, seed=100, classes={"small": (3,)})
        b = generate_cases(2, seed=100, classes={"small": (3,)})
        self.assertEqual(a, b)

    def test_single_parameter_changes_only(self):
        for parameter, value in (("population_size", 50),
                                 ("mutation_percent", 60),
                                 ("elitist_percent", 0.1)):
            changed = parameter_settings(parameter, value)
            difference = {k for k in BASELINE if BASELINE[k] != changed[k]}
            self.assertEqual(difference, {parameter})

    def test_paired_experiment_same_matrix_and_seed(self):
        records = run_study(matrix_count=1, repeats=1, seed=22,
                            classes={"small": (3,)},
                            sweeps={"population_size": (10, 20)})
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["seed"], records[1]["seed"])
        self.assertEqual(records[0]["matrix_id"], records[1]["matrix_id"])
        self.assertTrue(all(x["relative_gap"] >= 0 for x in records))
        summaries = summarize(records)
        self.assertEqual(len([x for x in summaries if x["size"] == "all"]), 2)

    def test_smoke_creates_csv_and_plots(self):
        with tempfile.TemporaryDirectory() as td:
            raw, summary = save_study(
                td, matrix_count=1, repeats=1, seed=2,
                classes={"small": (3,), "large": (8,)},
                sweeps={"population_size": (10, 20)}
            )
            self.assertEqual(len(raw), 4)
            self.assertTrue((Path(td) / "raw_results.csv").exists())
            self.assertTrue((Path(td) / "summary.csv").exists())
            self.assertTrue((Path(td) / "metadata.json").exists())
            self.assertTrue((Path(td) / "plots" /
                             "population_size_mean_gap.png").exists())
            with (Path(td) / "summary.csv").open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), len(summary))


if __name__ == "__main__":
    unittest.main()
