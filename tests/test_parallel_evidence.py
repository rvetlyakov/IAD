import os
import sys
import unittest
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lab1'))
from hybrid_ga import Config, run


class ParallelEvidenceTests(unittest.TestCase):
    def test_persistent_executor_and_ring_trace(self):
        a = np.full((20,20), 10, dtype=np.int32)
        np.fill_diagonal(a, 1)
        c = Config(population=31, generations=12, islands=4,
                   migration_interval=4, local_every=3, local_top=2)
        with ProcessPoolExecutor(max_workers=2) as pool:
            result = run(a, c, 42, workers=2, executor=pool)
            again = run(a, c, 42, workers=2, executor=pool)
        self.assertEqual(result['cost'], again['cost'])
        np.testing.assert_array_equal(result['best'], again['best'])
        self.assertTrue(set(result['worker_pids']).isdisjoint({os.getpid()}))
        self.assertEqual(result['migration_rounds'], 2)
        self.assertEqual(result['migration_edges'], [(0,1),(1,2),(2,3),(3,0)])
        single = run(a, replace(c, islands=1), 42)
        self.assertEqual(result['local_calls'], single['local_calls'])
        self.assertEqual(result['local_calls'], 8)

class RingPayloadTests(unittest.TestCase):
    def test_every_receiver_gets_predecessor_snapshot_without_aliases(self):
        import hybrid_ga
        class TraceExecutor:
            def __init__(self):
                self.inputs = []
                self.outputs = []
            def map(self, fn, tasks):
                tasks = list(tasks)
                self.inputs.append([task[1].copy() for task in tasks])
                results = [fn(task) for task in tasks]
                self.outputs.append([result[0][0].copy() for result in results])
                return iter(results)
        a = np.random.default_rng(33).integers(1, 101, (12,12), dtype=np.int32)
        cfg = Config(population=31, generations=3, islands=4,
                     mutation_prob=0, elitism=1., migration_interval=1,
                     migration_pct=1.)
        pool = TraceExecutor()
        run(a, cfg, seed=123, executor=pool)
        changed = 0
        for batch in (1,2):
            previous = pool.outputs[batch-1]
            for receiver in range(4):
                donor = (receiver-1) % 4
                # Independently reconstruct simultaneous ring replacement.
                migrants = previous[donor][np.argsort(hybrid_ga.costs(a,previous[donor]))]
                augmented = np.concatenate((previous[receiver], migrants))
                expected = augmented[np.argsort(hybrid_ga.costs(a,augmented),kind='stable')[:len(previous[receiver])]]
                np.testing.assert_array_equal(pool.inputs[batch][receiver], expected)
                changed += int(not np.array_equal(expected,previous[receiver]))
                self.assertFalse(np.shares_memory(pool.inputs[batch][receiver],previous[donor]))
        self.assertGreater(changed,0)

class UnevenBudgetTests(unittest.TestCase):
    def test_all_individuals_local_budget_with_uneven_islands_without_mutants(self):
        a=np.random.default_rng(6).integers(1,101,(8,8),dtype=np.int32)
        cfg=Config(population=7,generations=4,islands=3,mutation_prob=0,
                   local_every=1,local_top=7,migration_interval=2)
        result=run(a,cfg,seed=6,workers=2)
        self.assertEqual(result['local_calls'],28)
