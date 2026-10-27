"""Protect measured evidence against confounding and accidental relabeling."""
import csv
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lab1'))
from hybrid_ga import Config
from local_evidence import audit_row
from local_research import Study, problem, seed_for


class LocalEvidenceTests(unittest.TestCase):
    def actual_row(self, directory):
        cfg=Config(population=8,generations=4,local_every=2,local_top=1,cycle3=True)
        study=Study(directory,None,1)
        study.measure('train','small',6,0,cfg,parameter='generations',
                      context=replace(cfg,generations=3))
        with (Path(directory)/'train.csv').open(newline='') as f:
            return next(csv.DictReader(f))

    def test_audit_accepts_real_run_and_rejects_bad_error_or_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            row=self.actual_row(directory)
            audit_row(row)
            for field,value in [('error_pct','-10'),('population','999'),
                                ('local_calls','99'),('matrix_seed','0'),
                                ('generations','100'),('permutation_valid','False')]:
                with self.subTest(field=field), self.assertRaises(AssertionError):
                    audit_row({**row,field:value})

    def test_measure_rejects_two_factor_sweep_before_running(self):
        with tempfile.TemporaryDirectory() as directory:
            study=Study(directory,None,1)
            base=Config(population=8,generations=3)
            with self.assertRaises(AssertionError):
                study.measure('train','small',6,0,
                              replace(base,generations=4,mutation_prob=.3),
                              'generations',base)
            self.assertFalse((Path(directory)/'train.csv').exists())

    def test_distinct_stages_have_distinct_matrices_even_at_same_size(self):
        import numpy as np
        stages=('train','survey','holdout','diagonal')
        self.assertEqual(len({seed_for(s,20,0) for s in stages}),4)
        matrices=[problem(s,20,0)[0] for s in stages]
        for i,a in enumerate(matrices):
            for b in matrices[i+1:]:
                self.assertFalse(np.array_equal(a,b))

class ActiveContextTests(unittest.TestCase):
    def test_interval_scan_activates_migration_without_changing_frozen_winner(self):
        from local_research import survey_context
        base=Config()
        parallel=replace(base,islands=8,migration_pct=0.)
        context=survey_context(base,parallel,'migration_interval')
        self.assertGreater(context.migration_pct,0.)
        self.assertEqual(context.islands,8)
        self.assertEqual(parallel.migration_pct,0.)
        self.assertEqual({k for k in context.__dataclass_fields__
                          if getattr(context,k)!=getattr(parallel,k)}, {'migration_pct'})
