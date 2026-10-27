"""Regression tests for the assignment genetic algorithm (Python 3.13+)."""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab1"))
from lab1 import Employees


def matrix(size):
    return [[1 + ((i * 17 + j * 23 + i * j) % 101) for j in range(size)]
            for i in range(size)]


class SelectionTests(unittest.TestCase):
    def test_population_size_is_exact_after_selection(self):
        for chromosome_length in (1, 2, 3, 7):
            for n in (1, 2, 3, 10, 50):
                for elitism in (0.0, 0.2, 0.5, 1.0):
                    for seed in range(12):
                        with self.subTest(length=chromosome_length, n=n,
                                          elitism=elitism, seed=seed):
                            random.seed(seed)
                            pop = Employees(n, 0.7, matrix(chromosome_length),
                                            45, elitism)
                            pop.mutate()
                            pop.select()
                            self.assertEqual(len(pop.population), n)
                            self.assertTrue(all(sorted(x.vector) ==
                                                list(range(chromosome_length))
                                                for x in pop.population))

    def test_elitism_preserves_best_cost(self):
        for seed in range(60):
            random.seed(seed)
            pop = Employees(20, 0.8, matrix(8), 45, 0.2)
            old_cost = pop.get_best()[1]
            pop.mutate()
            pop.select()
            self.assertLessEqual(pop.get_best()[1], old_cost)

    def test_fractional_sampling_selects_low_ranked_candidates(self):
        # When all individuals have equal fitness and no elitism,
        # RSS must not systematically exclude the last candidates.
        random.seed(42)
        pop = Employees(8, 0.0, [[1] * 5 for _ in range(5)], 45, 0.0)
        # Exactly one expected copy per candidate: all must survive.
        before = sorted(tuple(x.vector) for x in pop.population)
        pop.select()
        after = sorted(tuple(x.vector) for x in pop.population)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
