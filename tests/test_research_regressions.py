"""Regressions discovered while auditing the GA and experimental protocol.

These tests intentionally reproduce faults of the 2026-10-27 main branch
before their fixes. Run with:
python -m unittest discover -s tests -p test_research_regressions.py -v
"""
import csv
import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab1"))
from lab1 import Employee, Employees
from lab1_test import run_ga
from hyperparameter_curves import HYPERPARAMETERS, run_experiment, _settings_for

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "results" / "scale_benchmark_full" / "best_parameters.json"


class GAResearchRegressions(unittest.TestCase):
    def test_employee_fitness_uses_supplied_chromosome(self):
        matrix = [[1, 80], [80, 1]]
        original = Employee([0, 1], 2, matrix, 30)
        self.assertAlmostEqual(original.fitness([1, 0]), 1 / 160)
        # Querying another vector must not silently use self.vector or corrupt it.
        self.assertEqual(original.vector, [0, 1])

    def test_zero_cost_assignments_do_not_crash_rss(self):
        matrix = [[0, 100], [100, 0]]
        random.seed(77)
        ga = Employees(20, .8, matrix, 45, .2)
        ga.mutate()
        ga.select()
        self.assertEqual(len(ga.population), 20)
        self.assertEqual(ga.get_best()[1], 0)
        self.assertTrue(all(x.fit >= 0 for x in ga.population))

    def test_run_ga_plateau_counts_changes_including_initial_population(self):
        matrix = [[1, 40, 90], [90, 1, 40], [40, 90, 1]]
        random.seed(100)
        found, generations, error = run_ga(
            population_size=15, mutation_probability=0,
            mutation_percent=45, tournament_size=2, max_iterations=25,
            plateau_limit=2, plateau_diff=1000.0, elitist_percent=.2,
            matrix=matrix,
        )
        self.assertEqual(generations, 2)  # two changes, three mean-fitness snapshots

    def test_run_ga_remembers_good_mutant_even_when_not_selected(self):
        # Force initial permutation to be poor, mutation to create optimum
        # and RSS to discard it. The best visited solution must still be reported.
        matrix = [[100, 1], [1, 100]]
        def optimum_mutant(self):
            return Employee([1, 0], self.length, self.matrix, self.mutation_percent)
        def discard_all(self):
            self.population = [Employee([0, 1], 2, matrix, 45)
                               for _ in range(self.n)]
        with patch("random.sample", return_value=[0, 1]):
            with patch.object(Employee, "mutate", optimum_mutant):
                with patch.object(Employees, "select", discard_all):
                    found, _, error = run_ga(
                        population_size=4, mutation_probability=1,
                        mutation_percent=45, tournament_size=2,
                        max_iterations=1, plateau_limit=2,
                        plateau_diff=0, elitist_percent=0, matrix=matrix
                    )
        self.assertTrue(found)
        self.assertEqual(error, 0)

    def test_zero_optimum_reports_infinite_relative_gap_without_division_error(self):
        # The exact assignment can cost zero, while a stochastic GA may miss it.
        matrix = [[0, 10], [10, 0]]
        with patch("random.sample", return_value=[1, 0]):
            found, generations, gap = run_ga(
                population_size=1, mutation_probability=0,
                mutation_percent=30, tournament_size=2, max_iterations=1,
                plateau_limit=2, plateau_diff=0,
                elitist_percent=0, matrix=matrix,
            )
        self.assertFalse(found)
        self.assertEqual(generations, 1)
        self.assertEqual(gap, float("inf"))

    def test_scale_benchmark_zero_optimum_uses_finite_or_infinite_gap(self):
        from scale_benchmark import evaluate_pair, BASELINE
        matrix = [[0, 100], [100, 0]]
        settings = dict(BASELINE, total_population=2, max_generations=1,
                        mutation_probability=0.0, elitist_percent=0.0)
        rows = evaluate_pair(matrix, settings,
                             dict(settings, num_islands=2),
                             seed=3, workers=2)
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["exact_cost"] for r in rows}, {0})
        self.assertTrue(all(r["relative_gap"] in (0.0, float("inf"))
                            for r in rows))

    def test_plateau_sweep_contains_strict_thresholds(self):
        actual = HYPERPARAMETERS["plateau_diff"]
        self.assertIn(0.0, actual)
        self.assertTrue(any(0 < x <= 1e-6 for x in actual))
        self.assertTrue(max(actual) < .005)

    def test_new_research_persists_exact_settings_per_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            report = run_experiment(
                mode="smoke", repeats=1, workers=2,
                seed=9876, output_dir=output,
                parameter_source=PARAMS, parameters=("mutation_probability",)
            )
            rows = list(csv.DictReader((output / "raw.csv").open()))
            self.assertTrue(rows)
            self.assertEqual(len(rows), len(report["rows"]))
            bases = json.loads(PARAMS.read_text())["best_parameters"]
            for row in rows:
                self.assertIn("settings_json", row)
                actual = json.loads(row["settings_json"])
                baseline = _settings_for(
                    row["class"], row["parameter"], bases, "smoke")
                baseline[row["parameter"]] = float(row["value"])
                self.assertEqual(actual, baseline)
                self.assertEqual(int(row["num_islands"]), actual["num_islands"])


if __name__ == "__main__":
    unittest.main()
