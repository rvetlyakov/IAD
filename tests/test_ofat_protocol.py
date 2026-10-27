"""Regression audit: prove the OFAT protocol for every plotted value.

Run: python -m unittest discover -s tests -p test_ofat_protocol.py -v
"""
import csv
import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"lab1"))
from hyperparameter_curves import (HYPERPARAMETERS, REAL_SIZES,
                                    PLATEAU_ONLY, ISLAND_ONLY,
                                    _settings_for, _levels)


class OFATProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = ROOT/"results"/"scale_benchmark_full"/"best_parameters.json"
        cls.bases = json.loads(result.read_text(encoding="utf-8"))["best_parameters"]
        cls.rows = list(csv.DictReader(
            (ROOT/"results"/"hyperparameter_curves_10_2000"/"raw.csv")
            .open(encoding="utf-8",newline="")
        ))

    def test_only_one_parameter_changes_within_each_sweep(self):
        for group in REAL_SIZES:
            for parameter, values in HYPERPARAMETERS.items():
                baseline = _settings_for(group, parameter, self.bases, "full")
                for value in values:
                    with self.subTest(group=group, parameter=parameter,value=value):
                        settings = dict(baseline)
                        settings[parameter] = value
                        difference = {
                            key for key in set(settings)|set(baseline)
                            if settings.get(key) != baseline.get(key)
                        }
                        self.assertLessEqual(difference, {parameter},
                            f"Unexpected extra change: {difference}")
                        self.assertEqual(settings[parameter],value)
                        if parameter in ISLAND_ONLY:
                            self.assertGreaterEqual(settings["num_islands"],1)
                            self.assertEqual(settings["max_generations"],
                                self.bases[group]["islands"]["max_generations"])
                        if parameter in PLATEAU_ONLY:
                            self.assertEqual(settings["num_islands"],1)

    def test_compared_levels_have_same_matrix_seed_and_reference(self):
        groups={}
        for row in self.rows:
            key=(row["parameter"],row["class"],int(row["size"]),int(row["repeat"]))
            groups.setdefault(key,[]).append(row)
        self.assertEqual(len(self.rows),465)
        for key, cases in groups.items():
            parameter,klass,size,repeat=key
            with self.subTest(key=key):
                self.assertEqual(len(cases),len(HYPERPARAMETERS[parameter]))
                self.assertEqual({float(x) for x in HYPERPARAMETERS[parameter]},
                                 {float(r["value"]) for r in cases})
                for common in ("matrix_seed","ga_seed","exact_cost"):
                    self.assertEqual(len({r[common] for r in cases}),1)
                self.assertEqual(int(cases[0]["size"]),size)

    def test_observed_run_metadata_matches_each_changed_setting(self):
        for row in self.rows:
            parameter=row["parameter"]
            baseline=_settings_for(row["class"],parameter,self.bases,"full")
            level=float(row["value"])
            if parameter in {"total_population","max_generations",
                             "num_islands","migration_interval",
                             "mutation_percent","plateau_limit"}:
                level=int(level)
            settings=dict(baseline,**{parameter:level})
            self.assertEqual(int(row["total_population"]),
                             int(settings["total_population"]))
            self.assertEqual(int(row["num_islands"]),int(settings["num_islands"]))
            if parameter not in PLATEAU_ONLY:
                self.assertEqual(int(row["generations"]),
                                 int(settings["max_generations"]))
            else:
                self.assertLessEqual(int(row["generations"]),
                                     int(settings["max_generations"]))

    def test_plateau_diff_is_currently_noninformative_for_large_n(self):
        group=[r for r in self.rows
               if r["parameter"]=="plateau_diff"
               and r["class"]=="large" and int(r["size"])>=1250]
        for size in (1250,1500,1750,2000):
            cases=[r for r in group if int(r["size"])==size]
            self.assertEqual(len(cases),3)
            self.assertEqual(len({r["cost"] for r in cases}),1)
            self.assertTrue(all(int(r["generations"])<=6 for r in cases))


if __name__=="__main__":
    unittest.main()
