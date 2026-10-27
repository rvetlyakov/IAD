"""Scale study: fairness, exact references, and real multiprocessing islands."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab1"))
from scale_benchmark import (
    BASELINE,
    SIZE_CLASSES,
    evaluate_pair,
    generate_matrix,
    run_study,
    run_variant,
    split_population,
)


class ScaleBenchmarkTests(unittest.TestCase):
    def test_requested_classes(self):
        self.assertEqual(SIZE_CLASSES["medium"], (100, 300, 500))
        self.assertEqual(SIZE_CLASSES["large"], (1000, 1500, 2000))

    def test_population_is_conserved_across_islands(self):
        for total in (4, 10, 80, 101):
            for islands in (1, 2, 3, 4):
                with self.subTest(total=total, islands=islands):
                    parts = split_population(total, islands)
                    self.assertEqual(len(parts), islands)
                    self.assertEqual(sum(parts), total)
                    self.assertLessEqual(max(parts) - min(parts), 1)

    def test_matrix_seed_reproducible(self):
        a = generate_matrix(8, 123)
        b = generate_matrix(8, 123)
        self.assertEqual(a.shape, (8, 8))
        self.assertTrue((a == b).all())
        self.assertGreaterEqual(a.min(), 1)
        self.assertLessEqual(a.max(), 100)

    def test_equal_budget_and_valid_costs_with_actual_workers(self):
        matrix = generate_matrix(6, 77)
        params = dict(BASELINE, total_population=12, max_generations=4,
                      migration_interval=2, migration_percent=25)
        simple = run_variant(matrix, dict(params, num_islands=1), seed=9, workers=2)
        islands = run_variant(matrix, dict(params, num_islands=3), seed=9, workers=2)
        self.assertEqual(simple.total_population, 12)
        self.assertEqual(islands.total_population, 12)
        self.assertEqual(simple.generations, 4)
        self.assertEqual(islands.generations, 4)
        self.assertEqual(islands.workers, 2)
        self.assertGreaterEqual(simple.cost, 6)
        self.assertGreaterEqual(islands.cost, 6)
        self.assertTrue(islands.parallel)

    def test_same_initial_budget_preserves_optimal_initial_chromosome(self):
        # Elitism 100% makes this a controlled test of initial chromosomes,
        # not a test of probabilistic selection or mutation.
        matrix = generate_matrix(6, 1234)
        params = dict(BASELINE, total_population=12, max_generations=1,
                      mutation_probability=0.0, elitist_percent=1.0)
        single = run_variant(matrix, dict(params, num_islands=1), seed=333)
        islands = run_variant(matrix, dict(params, num_islands=3), seed=333)
        self.assertEqual(single.cost, islands.cost)

    def test_comparison_uses_same_reference_and_matrix(self):
        matrix = generate_matrix(5, 91)
        params = dict(BASELINE, total_population=12, max_generations=4,
                      migration_interval=2)
        rows = evaluate_pair(matrix, params, dict(params, num_islands=2),
                             seed=11, workers=2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["exact_cost"], rows[1]["exact_cost"])
        self.assertEqual(rows[0]["matrix_seed"], rows[1]["matrix_seed"])
        self.assertEqual(rows[0]["total_population"], 12)
        self.assertEqual(rows[1]["total_population"], 12)
        self.assertTrue(all(row["relative_gap"] >= -1e-12 for row in rows))

    def test_smoke_writes_no_fabricated_full_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_study(mode="smoke", output_dir=Path(tmp), seed=22,
                               workers=2)
            self.assertTrue((Path(tmp) / "best_parameters.json").exists())
            self.assertTrue((Path(tmp) / "comparison.csv").exists())
            self.assertTrue((Path(tmp) / "tuning.csv").exists())
            self.assertIn("medium", report["best_parameters"])
            self.assertIn("large", report["best_parameters"])
            self.assertEqual(report["mode"], "smoke")
            self.assertNotEqual(report["tested_sizes"]["large"], [1000, 1500, 2000])
            self.assertTrue(all(row["num_islands"] > 1 for row in
                                report["comparison"] if row["variant"] == "islands"))


if __name__ == "__main__":
    unittest.main()
