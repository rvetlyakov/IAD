"""Tests for optional three-assignment cyclic local improvement."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lab1"))
from three_cycle import improve_three_cycle
from hybrid_ga import Config, best_columns, costs, run


class TestThreeCycle(unittest.TestCase):
    def test_three_cycle_escapes_a_strict_two_swap_local_minimum(self):
        a = np.array([[3, 0, 10], [10, 3, 0], [0, 10, 3]], dtype=np.int32)
        start = np.arange(3, dtype=np.int32)
        result, moves = improve_three_cycle(a, start, best_columns(a, 3), passes=2)
        self.assertEqual(int(costs(a, start)[0]), 9)
        self.assertEqual(int(costs(a, result)[0]), 0)
        self.assertGreater(moves, 0)

    def test_permutation_validity_and_nonincreasing_cost(self):
        for n in (5, 20, 100):
            for seed in (1, 11, 21):
                a = np.random.default_rng(seed).integers(1, 101, (n,n), dtype=np.int32)
                original = np.random.default_rng(n+seed).permutation(n).astype(np.int32)
                for passes in (1, 2, 4):
                    improved, _ = improve_three_cycle(a, original, best_columns(a, 8), passes)
                    self.assertEqual(sorted(improved.tolist()), list(range(n)))
                    self.assertLessEqual(int(costs(a, improved)[0]),
                                         int(costs(a, original)[0]))

    def test_diagonal_preserved(self):
        for n in (5, 30, 100):
            a = np.full((n,n), 100, dtype=np.int32)
            np.fill_diagonal(a, 1)
            v = np.random.default_rng(n).permutation(n).astype(np.int32)
            improved, _ = improve_three_cycle(a, v, best_columns(a, 8), passes=4)
            self.assertEqual(sorted(improved.tolist()), list(range(n)))
            self.assertGreaterEqual(int(costs(a, improved)[0]), n)

    def test_disabled_operator_matches_unmodified_baseline(self):
        a = np.random.default_rng(6).integers(1, 101, (30,30), dtype=np.int32)
        c = Config(population=40, generations=20, local_every=5, local_candidates=8)
        first = run(a, c, seed=555, workers=2)
        second = run(a, replace(c, cycle3=False, cycle3_passes=4), seed=555, workers=2)
        self.assertEqual(first["cost"], second["cost"])
        self.assertEqual(first["local_calls"], second["local_calls"])

    def test_equal_local_search_calls_with_islands_and_cycle3(self):
        a = np.random.default_rng(9).integers(1,101,(30,30),dtype=np.int32)
        c = Config(population=40, generations=15, local_every=5,
                   local_top=2, cycle3=True, cycle3_passes=1)
        for k in (1, 2, 4):
            result = run(a, replace(c,islands=k), seed=42, workers=2)
            self.assertEqual(result["total_population"], 40)
            self.assertEqual(result["local_calls"], 6)
            self.assertEqual(result["generations"], 15)


if __name__ == "__main__":
    unittest.main()
