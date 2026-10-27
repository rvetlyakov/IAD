"""Regression tests: all previous lab requirements and fair island search budgets."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"lab1"))
from hybrid_ga import Config, best_columns, costs, improve_swaps, rss_indices, run


class LabAudit(unittest.TestCase):
    def test_equal_population_and_global_local_search_across_island_counts(self):
        a=np.random.default_rng(27).integers(1,101,(20,20),dtype=np.int32)
        for k in (1,2,4,8):
            for local_top in (1,3,7):
                c=Config(population=32,generations=12,islands=k,
                         local_every=3,local_top=local_top,local_candidates=4)
                out=run(a,c,seed=42,workers=2)
                self.assertEqual(out["total_population"],32)
                self.assertEqual(out["local_calls"],4*local_top)
                self.assertEqual(sorted(out["best"].tolist()),list(range(20)))

    def test_exact_diagonal_and_valid_permutations(self):
        for n in (5,20,100):
            a=np.full((n,n),100,dtype=np.int32)
            np.fill_diagonal(a,1)
            v=np.random.default_rng(n).permutation(n).astype(np.int32)
            x,_=improve_swaps(a,v,best_columns(a,8),passes=3)
            self.assertEqual(sorted(x.tolist()),list(range(n)))
            self.assertEqual(int(costs(a,x)[0]),n)

    def test_rss_deterministic_under_fixed_seed(self):
        a=np.array([3,7,1,8,2,9],dtype=np.int64)
        for mode in ("rank","raw"):
            def f(mode=mode):
                return rss_indices(a,5,.2,mode,2,np.random.default_rng(299))
            self.assertEqual(f().tolist(),f().tolist())

    def test_diagonal_finds_optimum_with_local_swap(self):
        for n in (10,50):
            a=np.full((n,n),100,dtype=np.int32);np.fill_diagonal(a,1)
            out=run(a,Config(population=40,generations=10,
                       local_every=2,local_candidates=8),seed=12)
            self.assertEqual(out["cost"],n)

if __name__=="__main__":
    unittest.main()
