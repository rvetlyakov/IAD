"""Timing consistency between the legacy OFAT study and modern benchmark."""
import os
import random
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,os.path.join(os.path.dirname(__file__),"..","lab1"))
import experiments
from lab1_test import run_ga
from scipy.optimize import linear_sum_assignment


class LegacyTimingRegressions(unittest.TestCase):
    def test_run_ga_does_not_run_exact_solver_when_reference_precomputed(self):
        matrix=[[1,10,11],[11,1,12],[8,13,1]]
        with patch("lab1_test.linear_sum_assignment",
                   side_effect=AssertionError("exact solver called inside timed GA")):
            random.seed(17)
            outcome=run_ga(
                population_size=15,mutation_probability=.7,
                mutation_percent=45,tournament_size=2,max_iterations=5,
                plateau_limit=3,plateau_diff=0,
                elitist_percent=.2,matrix=matrix,reference_cost=3)
        self.assertEqual(len(outcome),3)
        self.assertGreaterEqual(outcome[2],0)

    def test_legacy_ofat_evaluates_reference_once_per_test_matrix(self):
        matrix_count=2
        with patch.object(experiments,"linear_sum_assignment",
                          wraps=linear_sum_assignment) as solver:
            rows=experiments.run_study(
                matrix_count=matrix_count,repeats=2,seed=900,
                classes={"small":(3,)},
                sweeps={"population_size":(5,10)},
            )
        self.assertEqual(len(rows),8)
        self.assertEqual(solver.call_count,matrix_count)


if __name__=="__main__":
    unittest.main()
