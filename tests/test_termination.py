import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab1"))
from lab1_test import run_ga
from lab1_test_migrations import run_ga_with_migrations
from termination import plateau_reached


class ConvergenceTests(unittest.TestCase):
    def test_identical_mean_fitness_counts_as_plateau(self):
        self.assertTrue(plateau_reached([0.1, 0.1, 0.1, 0.1], 3, 0))
        self.assertFalse(plateau_reached([0.1, 0.1, 0.1], 3, 0))

    def test_relative_changes_do_not_depend_on_scale(self):
        series = [1.0, 1.0001, 1.0002, 1.0003]
        tiny = [x / 1000 for x in series]
        self.assertEqual(plateau_reached(series, 3, 0.001),
                         plateau_reached(tiny, 3, 0.001))
        self.assertEqual(plateau_reached(series, 3, 1e-7),
                         plateau_reached(tiny, 3, 1e-7))

    def test_invalid_termination_parameters(self):
        for limit, threshold in ((0, 1e-4), (3, -1)):
            with self.assertRaises(ValueError):
                plateau_reached([1, 1, 1, 1], limit, threshold)

    def test_one_generation_is_executed_for_limit_one(self):
        matrix = [[2, 4], [3, 1]]
        params = (10, 0.5, 45, 2, 1, 5, 1e-4)
        self.assertEqual(run_ga(*params, 0.2, matrix)[1], 1)
        self.assertEqual(
            run_ga_with_migrations(*params, 2, 10, 3, 0.2, matrix)[1], 1
        )


if __name__ == "__main__":
    unittest.main()
