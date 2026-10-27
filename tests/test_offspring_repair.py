import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lab1'))
from hybrid_ga import Config, run


class OffspringRepairTests(unittest.TestCase):
    def test_seeded_trial_matches_measured_experimental_repair(self):
        n=50
        seed=950000000+n*1009
        a=np.random.default_rng(seed).integers(1,101,(n,n),dtype=np.int32)
        c=Config(population=64,generations=100,selection='rank',rank_power=2,
                 inversion_pct=5,local_every=5,local_top=2,local_candidates=16,
                 cycle3=True,cycle3_passes=3,local_on_offspring=True)
        ans=run(a,c,seed+17)
        self.assertEqual(ans['local_calls'],40)
        self.assertEqual(ans['cost'],206)
        self.assertEqual(sorted(ans['best']),list(range(n)))
        baseline=run(a,replace(c,local_on_offspring=False),seed+17)
        self.assertLess(ans['cost'],baseline['cost'])

    def test_equal_island_budget_and_diagonal_correctness(self):
        a=np.full((30,30),100,dtype=np.int32)
        np.fill_diagonal(a,1)
        c=Config(population=31,generations=10,local_every=2,local_top=3,
                 cycle3=True,local_on_offspring=True)
        for islands in (1,2,4):
            ans=run(a,replace(c,islands=islands),seed=77,workers=2)
            self.assertEqual(ans['cost'],30)
            self.assertEqual(ans['local_calls'],15)
