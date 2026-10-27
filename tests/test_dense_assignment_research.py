"""Integration-level checks for dense OFAT, class tuning and fair island comparison."""
import csv,json,sys,tempfile,unittest
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"lab1"))
from hybrid_ga import Config,run
from ga_dense_study import (DIMENSIONS,SMOKE_DIMS,FACTORS,settings_for,BASE,
                            positions,run_dense)
from ga_tune_classes import (select_best,tune,TRAIN,HOLDOUT)
from ga_fair_compare import compare
import numpy as np

class DenseResearchTests(unittest.TestCase):
    def test_full_dimensions_include_real_transition_measurements(self):
        self.assertEqual(DIMENSIONS["medium"][0],100)
        self.assertEqual(DIMENSIONS["medium"][-1],500)
        self.assertEqual(DIMENSIONS["large"][0],1000)
        self.assertEqual(DIMENSIONS["large"][-1],2000)
        self.assertEqual(DIMENSIONS["transition"],list(range(550,951,50)))
        self.assertGreater(sum(map(len,DIMENSIONS.values())),30)
        self.assertEqual(len(FACTORS),13)
        position=positions(DIMENSIONS)
        self.assertLess(position["medium",500],position["transition",550])
        self.assertLess(position["transition",950],position["large",1000])

    def test_only_one_setting_changes_within_any_factor_sweep(self):
        for name,levels in FACTORS.items():
            for label,changes in levels:
                cfg,basis=settings_for(name,label)
                changes_detected={key for key in asdict(cfg)
                    if asdict(cfg)[key]!=asdict(basis)[key]}
                self.assertLessEqual(changes_detected,set(changes),
                                     (name,label,changes_detected))
                self.assertEqual(len(changes),1)
                if name.startswith("migration_"):self.assertEqual(cfg.islands,4)

    def test_hyperparameter_selection_ignores_noisy_wall_clock(self):
        from ga_tune_classes import select_best
        configs=[Config(generations=40),Config(generations=100)]
        options=[
            ("40", [{"gap_pct":10.0,"elapsed_seconds":100.}], configs[0]),
            ("100",[{"gap_pct":9.9,"elapsed_seconds":10000.}], configs[1]),
        ]
        chosen=select_best(options)
        self.assertEqual(chosen[0],"100")
        self.assertAlmostEqual(chosen[1],9.9)
        options[0][1][0]["elapsed_seconds"]=.001
        options[1][1][0]["elapsed_seconds"]=.0001
        self.assertEqual(select_best(options)[0],"100")

    def test_dense_smoke_uses_real_matrices_and_paired_seeds(self):
        with tempfile.TemporaryDirectory() as temp:
            rows=run_dense(temp,"smoke",1,2,700,["selection","islands"])
            self.assertEqual(len(rows),sum(map(len,SMOKE_DIMS.values()))*(2+4))
            for cls,sizes in SMOKE_DIMS.items():
                for n in sizes:
                    subset=[x for x in rows if x["class"]==cls and x["size"]==n]
                    self.assertEqual(len({v["matrix_seed"] for v in subset}),1)
                    self.assertEqual(len({v["ga_seed"] for v in subset}),1)
                    self.assertEqual(len({v["optimal_cost"] for v in subset}),1)
                    self.assertTrue(all(v["ga_cost"]>=v["optimal_cost"] for v in subset))
            self.assertTrue((Path(temp)/"raw.csv").exists())

    def test_tuning_sequential_uses_independent_holdouts(self):
        with tempfile.TemporaryDirectory() as temp:
            result=tune(temp,"smoke",701,1,1,2,["selection","islands"])
            self.assertEqual(set(result["best"]),{"small","medium","large"})
            self.assertEqual(len(result["validation"]),6)
            train=list(csv.DictReader((Path(temp)/"train_raw.csv").open()))
            held=list(csv.DictReader((Path(temp)/"validation_raw.csv").open()))
            self.assertTrue(set(x["matrix_seed"] for x in train).isdisjoint(
                set(x["matrix_seed"] for x in held)))
            for cls in ("small","medium","large"):
                variants=[v for v in held if v["class"]==cls]
                self.assertEqual(len(variants),2)
                for name in ("population","generations","matrix_seed","ga_seed"):
                    self.assertEqual(len({r[name] for r in variants}),1)
                self.assertEqual({r["variant"] for r in variants},{"single","parallel"})

    def test_fair_comparison_varies_island_count_only(self):
        with tempfile.TemporaryDirectory() as temp:
            rows=compare(temp,"smoke",1,703,2)
            self.assertEqual(len(rows),sum(map(len,SMOKE_DIMS.values()))*4)
            for cls,ns in SMOKE_DIMS.items():
                for n in ns:
                    x=[r for r in rows if r["class"]==cls and r["size"]==n]
                    self.assertEqual({r["islands"] for r in x},{1,2,4,8})
                    self.assertEqual(len({r["population"] for r in x}),1)
                    self.assertEqual(len({r["generations"] for r in x}),1)
                    self.assertEqual(len({r["matrix_seed"] for r in x}),1)
                    c=[json.loads(r["settings_json"]) for r in x]
                    for key in c[0]:
                        if key!="islands":self.assertEqual(len({j[key] for j in c}),1)
            self.assertEqual(len(list((Path(temp)/"plots").glob("*.png"))),2)

    def test_diagonal_assignments_remain_permutations_and_exact_reference(self):
        n=12
        a=np.full((n,n),100,dtype=np.int32)
        np.fill_diagonal(a,1)
        cfg=Config(population=30,generations=40,selection="rank",
                   local_every=5,local_candidates=4)
        result=run(a,cfg,seed=77)
        self.assertEqual(sorted(result["best"].tolist()),list(range(n)))
        self.assertEqual(result["cost"],n)

if __name__=="__main__":unittest.main()
