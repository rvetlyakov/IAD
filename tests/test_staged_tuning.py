"""Methodology tests for independent train, OFAT and holdout partitions."""
import sys,unittest,json,tempfile
from pathlib import Path
from unittest.mock import patch
from dataclasses import asdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"lab1"))
from staged_tuning import (class_for,CLASS_GRID,TRAIN,HOLDOUT,SEEDS,LEVELS,
    ACTIVE_CONTEXT,_seed,train,ofat,transfer)
from hybrid_ga import Config

class StagedTuningTests(unittest.TestCase):
    def test_nonoverlapping_user_classes(self):
        self.assertEqual(class_for(2),"small")
        self.assertEqual(class_for(99),"small")
        self.assertEqual(class_for(100),"medium")
        self.assertEqual(class_for(499),"medium")
        self.assertEqual(class_for(499),"medium")
        self.assertEqual(class_for(500),"large")
        self.assertEqual(class_for(501),"large")
        self.assertEqual(class_for(550),"large")
        self.assertEqual(class_for(950),"large")
        self.assertEqual(class_for(999),"large")
        self.assertEqual(class_for(1000),"large")
        self.assertEqual(class_for(2000),"large")
        self.assertEqual([len(x) for x in CLASS_GRID.values()],[9,16,21])
        self.assertTrue(set(range(550,1000,50)).issubset(CLASS_GRID["large"]))
        for name,values in CLASS_GRID.items():
            self.assertTrue(all(class_for(n)==name for n in values))

    def test_all_matrix_seeds_are_disjoint_across_stages(self):
        for n in (20,100,150,300,450,500,1000):
            keys={stage:{_seed(n,x,stage) for x in values}
                  for stage,values in SEEDS.items()}
            for first in keys:
                for second in keys:
                    if first!=second:self.assertFalse(keys[first]&keys[second])

    def test_adaptive_mutation_does_not_change_when_generation_budget_changes(self):
        from hybrid_ga import _evolve
        n=10
        rng=np.random.default_rng(123)
        matrix=rng.integers(1,101,(n,n),dtype=np.int32)
        population=np.asarray([rng.permutation(n) for _ in range(12)],
                              dtype=np.int32)
        base=Config(population=12,generations=50,adaptive=True,
                    adaptive_horizon=100,local_every=0)
        seed_state=np.random.default_rng(991).bit_generator.state
        best=(int(np.min(matrix[np.arange(n)[None,:],population].sum(axis=1))),
              population[0].copy())
        short=_evolve((matrix,population.copy(),base,0,50,
                       seed_state,None,best,0))
        long=_evolve((matrix,population.copy(),
                      __import__("dataclasses").replace(base,generations=200),
                      0,50,seed_state,None,best,0))
        np.testing.assert_array_equal(short[0],long[0])
        self.assertEqual(short[2][0],long[2][0])

    def test_conditional_controls_are_active(self):
        self.assertEqual(ACTIVE_CONTEXT["rank_power"],{"selection":"rank"})
        self.assertEqual(ACTIVE_CONTEXT["migration_interval"],{"islands":4})
        self.assertEqual(ACTIVE_CONTEXT["adaptive_long_pct"],{"adaptive":True})

    def test_training_selects_best_from_real_candidate_grid(self):
        base=Config(generations=100,local_candidates=8,population=16)
        class ToyRun:
            def __init__(self,config):
                self.cost=100+abs(config.generations-500)//10 + abs(config.local_candidates-16)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/"source.json"
            source.write_text(json.dumps({"medium":{"single":asdict(base)}}))
            def fake_eval(a,opt,c,seed,workers):
                cost=ToyRun(c).cost
                return {"cost":cost,"seconds":c.generations/1000,
                        "total_population":c.population,
                        "generations":c.generations,"num_islands":c.islands,
                        "local_calls":0},float(cost)
            with patch("staged_tuning.TRAIN",{"medium":(3,4)}), \
                 patch("staged_tuning.SEEDS",dict(SEEDS,train=(2,))), \
                 patch("staged_tuning.problem",side_effect=lambda n,seed:(np.ones((n,n)),10)), \
                 patch("staged_tuning.evaluate",side_effect=fake_eval):
                result=train("medium",root,source,workers=2)
            self.assertEqual(result["generations"],500)
            self.assertEqual(result["local_candidates"],16)
            saved=json.loads((root/"medium_frozen_baseline.json").read_text())
            self.assertEqual(saved["selected_settings"]["generations"],500)
            self.assertEqual(saved["selected_settings"]["local_candidates"],16)

    def test_large_class_parameter_transfers_only_if_it_helps_small_train(self):
        base=Config(population=16,generations=20,mutation_prob=.3)
        medium=__import__("dataclasses").replace(base,mutation_prob=.7)
        large=__import__("dataclasses").replace(base,mutation_prob=1.)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,cfg in (("small",base),("medium",medium),("large",large)):
                (root/f"{name}_frozen_baseline.json").write_text(
                    json.dumps({"selected_settings":asdict(cfg)}))
            def fake_eval(a,opt,cfg,seed,workers):
                # Large donor setting improves target TRAIN; all other
                # factors and the population/generation budget stay fixed.
                return {"cost":100,"seconds":.1},100*(1-cfg.mutation_prob)
            with patch("staged_tuning.TRAIN",
                       {"small":(10,),"medium":(20,),"large":(30,)}), \
                 patch("staged_tuning.SEEDS",dict(SEEDS,train=(7,))), \
                 patch("staged_tuning.problem",
                       side_effect=lambda n,seed:(np.ones((2,2)),2)), \
                 patch("staged_tuning.evaluate",side_effect=fake_eval):
                selected=transfer("small",root,workers=1)
            self.assertEqual(selected["mutation_prob"],1.)
            self.assertEqual(selected["population"],16)
            self.assertEqual(selected["generations"],20)
            rows=list(__import__("csv").DictReader(
                (root/"small_transfer.csv").open()))
            self.assertTrue(rows)
            for row in rows:
                before=json.loads(row["context_json"])
                after=json.loads(row["settings_json"])
                self.assertLessEqual({k for k in before if before[k]!=after[k]},
                                     {row["parameter"]})

    def test_ofat_proves_only_one_field_changed_for_all_factors(self):
        base=Config(generations=150,population=16,islands=1,local_candidates=8)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"medium_frozen_baseline.json").write_text(
                json.dumps({"selected_settings":asdict(base)}))
            def fake_eval(a,opt,c,seed,workers):
                return {"cost":10,"seconds":.01,"total_population":c.population,
                        "generations":c.generations,"num_islands":c.islands,
                        "local_calls":0},1.
            with patch("staged_tuning.CLASS_GRID",{"medium":(4,)}), \
                 patch("staged_tuning.SEEDS",dict(SEEDS,survey=(9,))), \
                 patch("staged_tuning.problem",return_value=(np.ones((4,4)),4)), \
                 patch("staged_tuning.evaluate",side_effect=fake_eval):
                count=ofat("medium",root,workers=2)
            self.assertEqual(count,sum(len(v) for v in LEVELS.values()))
            import csv
            with (root/"medium_ofat.csv").open(newline="") as f:
                rows=list(csv.DictReader(f))
            for row in rows:
                original=json.loads(row["context_json"])
                actual=json.loads(row["settings_json"])
                differences={k for k in original if original[k]!=actual[k]}
                self.assertLessEqual(differences,{row["parameter"]})

if __name__=="__main__":
    unittest.main()
