"""Trajectory replay must reproduce exact plateau stopping without re-running RNG."""
import csv, json, os, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0,os.path.join(os.path.dirname(__file__),"..","lab1"))
from plateau_trace_study import (
    trace_evolution, replay_plateau, run_trace_study,
    PLATEAU_THRESHOLDS, PLATEAU_LIMITS
)
from hyperparameter_curves import run_plateau_variant
from scale_benchmark import generate_matrix


class PlateauTraceTests(unittest.TestCase):
    def test_replay_matches_direct_early_stop_on_identical_seed(self):
        matrix=generate_matrix(8,2121)
        base=dict(total_population=18,mutation_probability=.8,
                  mutation_percent=45,elitist_percent=.2,
                  max_generations=15,plateau_limit=3,plateau_diff=.002)
        trajectory=trace_evolution(matrix,base,seed=121)
        self.assertEqual(len(trajectory),16)
        for limit in (2,3,5):
            for diff in (0.0,.0003,.003,.1):
                with self.subTest(limit=limit,diff=diff):
                    actual=run_plateau_variant(matrix,dict(base,
                                  plateau_limit=limit,plateau_diff=diff),seed=121)
                    replay=replay_plateau(trajectory,limit,diff)
                    self.assertEqual(actual["generations"],replay["generations"])
                    self.assertEqual(actual["cost"],replay["cost"])

    def test_trajectory_is_nonincreasing_best_cost_and_positive_fitness(self):
        matrix=generate_matrix(9,912)
        params=dict(total_population=12,mutation_probability=1.0,
                    mutation_percent=20,elitist_percent=.2,max_generations=10)
        sequence=trace_evolution(matrix,params,seed=99)
        costs=[record["best_cost"] for record in sequence]
        self.assertEqual(costs,sorted(costs,reverse=True))
        self.assertTrue(all(record["mean_fitness"]>0 for record in sequence))

    def test_threshold_brackets_old_too_tight_and_old_too_loose(self):
        self.assertLess(min(PLATEAU_THRESHOLDS),.0001)
        self.assertGreaterEqual(max(PLATEAU_THRESHOLDS),.005)
        self.assertGreaterEqual(len(PLATEAU_THRESHOLDS),5)
        self.assertGreaterEqual(len(PLATEAU_LIMITS),3)

    def test_technical_smoke_saves_independent_real_trajectories_and_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=run_trace_study(output_dir=Path(tmp),mode="smoke",seed=999,repeats=1)
            self.assertEqual(report["mode"],"smoke")
            traces=list(csv.DictReader((Path(tmp)/"trajectories.csv").open()))
            rows=list(csv.DictReader((Path(tmp)/"raw.csv").open()))
            self.assertGreater(len(traces),20)
            self.assertEqual(len(rows),len(report["rows"]))
            self.assertTrue(all("settings_json" in row for row in rows))
            self.assertTrue(all(int(row["size"])<100 for row in rows))
            self.assertEqual(len(list((Path(tmp)/"plots").glob("*.png"))),4)
            meta=json.loads((Path(tmp)/"metadata.json").read_text())
            self.assertEqual(meta["source"],"full-run trajectory replay")
            self.assertTrue(meta["size_classes"].get("large"))


if __name__=="__main__":
    unittest.main()
