"""Hybrid model regression tests. Runs without expensive full experiments."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"lab1"))
from hybrid_ga import (
    Config,
    best_columns,
    costs,
    improve_swaps,
    rss_indices,
    run,
    validate,
)


class HybridGAValidationTests(unittest.TestCase):
    def test_rss_preserves_exact_total_population(self):
        rng=np.random.default_rng(123)
        for candidates in (10,20,50):
            for selector in ("raw","rank"):
                values=rng.integers(1,100,candidates)
                for elitism in (0,.1,.2,1):
                    selected=rss_indices(values,10,elitism,selector,2,rng)
                    self.assertEqual(len(selected),10)
                    self.assertTrue(np.all(selected>=0))
                    self.assertTrue(np.all(selected<candidates))

    def test_rss_supports_zero_cost(self):
        cc=np.array([0,0,10,11],dtype=np.int64)
        selected=rss_indices(cc,4,.2,"raw",2,np.random.default_rng(2))
        self.assertEqual(len(selected),4)

    def test_pair_swap_is_valid_and_monotonic(self):
        for n in (5,10,50):
            matrix=np.random.default_rng(n).integers(1,101,(n,n),dtype=np.int32)
            v=np.random.default_rng(n+2).permutation(n).astype(np.int32)
            result,_=improve_swaps(matrix,v,best_columns(matrix,8),passes=4)
            self.assertEqual(sorted(result.tolist()),list(range(n)))
            self.assertLessEqual(int(costs(matrix,result)[0]),int(costs(matrix,v)[0]))

    def test_diagonal_120_exact_under_local_swap(self):
        n=120
        a=np.full((n,n),100,dtype=np.int32)
        np.fill_diagonal(a,1)
        v=np.random.default_rng(77).permutation(n).astype(np.int32)
        result,_=improve_swaps(a,v,best_columns(a,8),passes=3)
        self.assertEqual(int(costs(a,result)[0]),n)

    def test_island_partition_does_not_multiply_total_n(self):
        matrix=np.random.default_rng(9).integers(1,100,(30,30),dtype=np.int32)
        base=Config(population=31,generations=10,selection="rank",local_every=5)
        for islands in (1,2,4,8):
            result=run(matrix,replace(base,islands=islands),seed=9,workers=2)
            self.assertEqual(result["total_population"],31)
            self.assertEqual(result["generations"],10)
            self.assertEqual(sorted(result["best"].tolist()),list(range(30)))

    def test_invalid_configuration_fails(self):
        a=np.ones((3,3),dtype=np.int32)
        for bad in (Config(population=0),Config(inversion_pct=-1),
                    Config(islands=999),Config(selection="bad"),
                    Config(local_every=-1)):
            with self.assertRaises(ValueError):
                validate(a,bad)

if __name__=="__main__":
    unittest.main()
