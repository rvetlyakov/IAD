import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"lab1"))
from size_curves import DIMENSIONS, LEVELS, running_mean, save_curves


class DimensionCurvesTests(unittest.TestCase):
    def test_contiguous_dimensions_cover_all_three_classes(self):
        self.assertEqual([i for values in DIMENSIONS.values() for i in values],
                         list(range(3,25)))
        self.assertEqual(set(LEVELS), {
            "population_size","mutation_probability","mutation_percent",
            "elitist_percent","plateau_limit","plateau_diff","max_iterations"
        })

    def test_display_mean_never_invents_new_dimensions(self):
        self.assertEqual(running_mean([0,6,9,12]),[3,5,9,10.5])

    def test_one_factor_and_paired_matrices(self):
        with tempfile.TemporaryDirectory() as path:
            rows,summary=save_curves(
                output=path,matrix_count=1,repeats=1,seed=20261021,
                dimensions={"small":(3,), "medium":(8,), "large":(16,)},
                levels={"population_size":(10,20)}
            )
            self.assertEqual(len(rows),6)
            for n in (3,8,16):
                subset=[r for r in rows if r["size"]==n]
                self.assertEqual(len(subset),2)
                self.assertEqual(subset[0]["seed"],subset[1]["seed"])
            self.assertEqual(len(summary),6)
            self.assertTrue((Path(path)/"summary.csv").is_file())
            self.assertTrue((Path(path)/"metadata.json").is_file())


if __name__=="__main__":
    unittest.main()
