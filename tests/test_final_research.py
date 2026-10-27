import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lab1'))

class FinalResearchTests(unittest.TestCase):
    def test_new_profile_starts_with_offspring_repair_and_separate_seeds(self):
        from final_research import FINAL_OFFSETS, starting_config
        for group in ('small','medium','large'):
            cfg=starting_config(group)
            self.assertTrue(cfg.local_on_offspring)
            self.assertEqual(cfg.local_every,5)
            self.assertEqual(cfg.population,64)
        self.assertEqual(len(set(FINAL_OFFSETS.values())),4)
        self.assertGreater(min(FINAL_OFFSETS.values()),1_000_000_000)

    def test_plot_points_are_means_of_actual_repeated_runs(self):
        from local_evidence import measured_series
        ns,ys=measured_series([{'n':'10','error_pct':'2'},
                               {'n':'10','error_pct':'6'},
                               {'n':'20','error_pct':'10'}],'error_pct')
        self.assertEqual(ns,[10,20])
        self.assertEqual(ys,[4,10])

    def test_csv_level_matching_preserves_booleans_and_numeric_equivalence(self):
        from local_evidence import value_key
        self.assertEqual(value_key(True),value_key('True'))
        self.assertEqual(value_key(False),value_key('False'))
        self.assertEqual(value_key(2),value_key('2.0'))
        self.assertNotEqual(value_key(True),value_key(1))
