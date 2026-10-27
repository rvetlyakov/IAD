import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab1"))
from experiments import parameter_settings, run_study
from plateau_sensitivity import BASELINE_PLATEAU, PLATEAU_SWEEPS


class PlateauSensitivityTests(unittest.TestCase):
    def test_plateau_study_varies_one_parameter_per_sweep(self):
        for parameter, values in PLATEAU_SWEEPS.items():
            for value in values:
                settings = parameter_settings(parameter, value, BASELINE_PLATEAU)
                differences = {k for k in BASELINE_PLATEAU
                               if settings[k] != BASELINE_PLATEAU[k]}
                self.assertLessEqual(len(differences), 1)
                self.assertTrue(differences <= {parameter})

    def test_plateau_can_stop_before_iteration_limit(self):
        records = run_study(
            matrix_count=1, repeats=1, seed=2,
            classes={"small": (3,), "medium": (8,)},
            sweeps={"plateau_diff": (0.1,)},
            baseline=BASELINE_PLATEAU,
        )
        self.assertTrue(any(r["generations"] < BASELINE_PLATEAU["max_iterations"]
                            for r in records))


if __name__ == "__main__":
    unittest.main()
