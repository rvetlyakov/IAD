"""Regression tests for continuous numeric class axis and OFAT impact metrics."""
import sys,unittest,tempfile,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"lab1"))
from medium_three_cycle_study import SURVEY_SIZES,dimension_x,class_for as medium_class
from staged_tuning import CLASS_GRID,class_for
from staged_charts import x_value,sensitivity

class ContinuousAxisTests(unittest.TestCase):
    def test_real_survey_has_no_transition_group_and_all_35_sizes(self):
        self.assertEqual(set(SURVEY_SIZES),{"small","medium","large"})
        ns=sorted(n for values in SURVEY_SIZES.values() for n in values)
        self.assertEqual(len(ns),35)
        self.assertEqual(len(ns),len(set(ns)))
        self.assertTrue(set(range(550,1000,50)).issubset(ns))
        self.assertTrue(all(medium_class(n)==group
                            for group,values in SURVEY_SIZES.items()
                            for n in values))
    def test_large_staged_class_covers_transition_without_gap(self):
        self.assertEqual(set(CLASS_GRID),{"small","medium","large"})
        self.assertTrue(set(range(550,1000,50)).issubset(CLASS_GRID["large"]))
        self.assertTrue(all(class_for(n)==group
                            for group,values in CLASS_GRID.items()
                            for n in values))
        self.assertTrue(all(a<b for a,b in zip(
            sorted(n for v in CLASS_GRID.values() for n in v),
            sorted(n for v in CLASS_GRID.values() for n in v)[1:])))
    def test_axis_continuous_at_100_and_500_and_monotonic(self):
        for x in (dimension_x,x_value):
            self.assertAlmostEqual(x(100),1.)
            self.assertAlmostEqual(x(500),2.)
            self.assertAlmostEqual(x(2000),3.)
            for n in range(3,2001):
                self.assertLess(x(n-1),x(n))
        self.assertEqual([medium_class(n) for n in (99,100,499,500)],
                         ["small","medium","medium","large"])
    def test_sensitivity_reports_actual_error_spread(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/"small_ofat.csv"
            with p.open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=("class_name","n","parameter",
                                     "value","error_pct","seconds"))
                w.writeheader()
                for n in (10,20):
                    for value,error in (("0.3",50),("0.7",20)):
                        w.writerow(dict(class_name="small",n=n,
                                        parameter="mutation_prob",value=value,
                                        error_pct=error,seconds=.2))
            report=sensitivity(root)
            r=[r for r in report if r["parameter"]=="mutation_prob"]
            self.assertEqual(len(r),1)
            self.assertAlmostEqual(r[0]["error_spread_percentage_points"],30.)
            self.assertTrue((Path(root)/"best_observed_levels.csv").is_file())
if __name__=="__main__":
    unittest.main()
