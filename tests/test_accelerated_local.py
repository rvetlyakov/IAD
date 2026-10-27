"""Compiled local search must preserve the reference algorithm exactly."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lab1'))
from hybrid_ga import best_columns, improve_swaps
from three_cycle import improve_three_cycle


class AccelerationTests(unittest.TestCase):
    def test_compiled_kernels_match_python(self):
        self.assertTrue(hasattr(improve_swaps, 'py_func'))
        self.assertTrue(hasattr(improve_three_cycle, 'py_func'))
        for n in (3, 20, 100):
            for seed in (2, 7):
                a = np.random.default_rng(seed).integers(1, 101, (n,n), dtype=np.int32)
                p = np.random.default_rng(seed+1).permutation(n).astype(np.int32)
                cols = best_columns(a, 8)
                for fn in (improve_swaps, improve_three_cycle):
                    actual, moves = fn(a, p, cols, 3)
                    expected, reference_moves = fn.py_func(a, p, cols, 3)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(moves, reference_moves)
