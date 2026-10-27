"""Bounded local TRAIN -> frozen OFAT -> independent HOLDOUT experiment.

Every CSV row is a completed actual call. No interpolation or imputation.
Persistent spawn workers amortize startup; their warmup is reported separately.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path
from statistics import mean

import numpy as np
from hybrid_ga import Config, best_columns, improve_swaps, run
from scipy.optimize import linear_sum_assignment
from staged_tuning import ACTIVE_CONTEXT, CLASS_GRID
from three_cycle import improve_three_cycle

LEVELS = {
    'generations': (40, 100, 250, 500, 1000),
    'population': (32, 64, 96),
    'selection': ('raw', 'rank'),
    'rank_power': (1., 2., 4.),
    'mutation_prob': (.3, .7, 1.),
    'inversion_pct': (1., 5., 20.),
    'elitism': (0., .2, .4),
    'local_every': (0, 5, 10, 20),
    'local_candidates': (8, 16, 32),
    'local_passes': (1, 2, 4),
    'local_top': (1, 2, 4),
    'cycle3': (False, True),
    'cycle3_passes': (1, 3, 6),
    'adaptive': (False, True),
    'adaptive_horizon': (100, 500, 1000),
    'adaptive_long_prob': (0., .05, .15),
    'adaptive_long_pct': (5., 20., 40.),
    'islands': (1, 2, 4, 8),
    'migration_interval': (5, 15, 30),
    'migration_pct': (0., .1, .2),
}
TRAIN = {'small': (40, 80), 'medium': (175, 375), 'large': (700, 1400)}
HOLDOUT = {'small': (25, 65, 95), 'medium': (125, 275, 475),
           'large': (525, 875, 1150, 1550, 1950, 2000)}
# Separate seed namespaces even when dimensions coincide.
STAGE_OFFSET = {'train': 100_000_000, 'survey': 300_000_000,
                'holdout': 500_000_000, 'diagonal': 700_000_000}

def survey_context(base, parallel, parameter):
    if parameter in ('migration_interval','migration_pct'):
        context=parallel
        if parameter=='migration_interval' and not context.migration_pct:
            context=replace(context,migration_pct=.1)
        return context
    return replace(base,**ACTIVE_CONTEXT.get(parameter,{}))

def seed_for(stage, n, repeat):
    return STAGE_OFFSET[stage] + 20261028 + n*1009 + repeat*100003

@lru_cache(maxsize=8)
def problem(stage, n, repeat):
    seed = seed_for(stage, n, repeat)
    a = np.random.default_rng(seed).integers(1, 101, (n,n), dtype=np.int32)
    if stage == 'diagonal':
        a[:] = 100
        np.fill_diagonal(a, 1)
    rows, cols = linear_sum_assignment(a)
    return a, int(a[rows, cols].sum(dtype=np.int64))

def warm_worker(_):
    a, _ = problem('train', 20, 9)
    cols = best_columns(a, 8)
    p = np.arange(20, dtype=np.int32)
    improve_swaps(a, p, cols, 2)
    improve_three_cycle(a, p, cols, 2)
    time.sleep(.1)
    return os.getpid()

class Study:
    def __init__(self, root, executor, workers):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.executor, self.workers = executor, workers
        self.count = 0

    def measure(self, stage, group, n, repeat, cfg, parameter='', context=None,
                variant='', phase=''):
        context = cfg if context is None else context
        changed = {key for key in asdict(cfg)
                   if getattr(cfg,key) != getattr(context,key)}
        if parameter:
            assert changed <= {parameter}, (parameter, changed)
        a, optimum = problem(stage, n, repeat)
        seed = seed_for(stage, n, repeat)
        ans = run(a, cfg, seed+17, self.workers, executor=self.executor)
        assert np.array_equal(np.sort(ans['best']), np.arange(n))
        assert ans['cost'] == int(a[np.arange(n), ans['best']].sum())
        assert ans['cost'] >= optimum
        assert ans['total_population'] == cfg.population
        assert ans['local_calls'] == (cfg.generations//cfg.local_every)*cfg.local_top if cfg.local_every else ans['local_calls'] == 0
        if cfg.islands > 1:
            assert ans['worker_pids'] and os.getpid() not in ans['worker_pids']
        row = {'stage': stage, 'class_name': group, 'n': n, 'repeat': repeat,
                   'parameter': parameter, 'value': str(getattr(cfg,parameter)) if parameter else '',
                   'variant': variant, 'phase': phase, 'cost': ans['cost'], 'exact': optimum,
                   'error_pct': 100*(ans['cost']-optimum)/optimum,
                   'seconds': ans['seconds'], 'matrix_seed': seed, 'ga_seed': seed+17,
                   'population': cfg.population, 'generations': cfg.generations,
                   'local_calls': ans['local_calls'], 'islands': cfg.islands,
                   'worker_pids': json.dumps(ans['worker_pids']),
                   'migration_rounds': ans['migration_rounds'],
                   'permutation_valid': True,
                   'context_json': json.dumps(asdict(context),sort_keys=True),
                   'settings_json': json.dumps(asdict(cfg),sort_keys=True)}
        path = self.root / f'{stage}.csv'
        exists = path.exists()
        with path.open('a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
        self.count += 1
        return row

    def tune_factor(self, group, cfg, parameter, values, phase):
        trials = []
        # Always include the current value. Do not regress due to an omitted baseline.
        for value in dict.fromkeys((getattr(cfg,parameter),)+tuple(values)):
            candidate = replace(cfg, **{parameter:value})
            rows = [self.measure('train', group, n, rep, candidate,
                                 parameter, cfg, phase=phase)
                    for n in TRAIN[group] for rep in (0,1)]
            trials.append((mean(r['error_pct'] for r in rows), value))
        best_error = min(error for error,_ in trials)
        current_error = trials[0][0]
        # Retain current on ties; time noise must not choose scientific settings.
        value = getattr(cfg,parameter) if current_error <= best_error+1e-12 else next(v for e,v in trials if e == best_error)
        print('TRAIN', group, phase, parameter, value, 'error', best_error, flush=True)
        return replace(cfg, **{parameter:value}), trials

    def train(self):
        selected = {}
        for group in TRAIN:
            cfg = Config(population=64, generations=100, selection='rank',
                         inversion_pct=1., elitism=.2, local_every=5,
                         local_top=2, local_passes=2, local_candidates=16,
                         cycle3=True, cycle3_passes=3, migration_interval=15)
            starting = asdict(cfg)
            history = []
            # G selected last: local-search settings are tested with fixed G first.
            order = [p for p in LEVELS if p not in ('islands','migration_interval','migration_pct','generations')]+['generations']
            for parameter in order:
                if parameter == 'rank_power' and cfg.selection != 'rank':
                    continue
                if parameter.startswith('adaptive_') and not cfg.adaptive:
                    continue
                if parameter == 'cycle3_passes' and not cfg.cycle3:
                    continue
                levels = LEVELS[parameter]
                if parameter == 'generations' and group != 'medium':
                    levels = (40,100,250,500)
                cfg, scores = self.tune_factor(group,cfg,parameter,levels,'single')
                history.append({'parameter': parameter,'scores': scores})
            # Include 1 to identify whether islands improve quality at all.
            island, scores = self.tune_factor(group,cfg,'islands',LEVELS['islands'],'islands')
            recommended = island.islands
            if island.islands == 1:
                # Best truly parallel candidate, still reported separately from recommendation.
                _, best_parallel = min((e,v) for e,v in scores if v > 1)
                island = replace(cfg,islands=best_parallel)
            for parameter in ('migration_interval','migration_pct'):
                island, _ = self.tune_factor(group,island,parameter,LEVELS[parameter],'islands')
            selected[group] = {'single': asdict(cfg), 'parallel': asdict(island),
                                   'starting': starting, 'recommended_islands': recommended,
                                   'coordinate_history': history}
            (self.root / 'selected.json').write_text(json.dumps(selected,indent=2))
        return selected

    def survey(self, selected):
        for group, sizes in CLASS_GRID.items():
            base = Config(**selected[group]['single'])
            parallel = Config(**selected[group]['parallel'])
            for n in sizes:
                for parameter, values in LEVELS.items():
                    context = survey_context(base,parallel,parameter)
                    for value in dict.fromkeys((getattr(context,parameter),)+tuple(values)):
                        self.measure('survey',group,n,0,replace(context,**{parameter:value}),parameter,context)
                for label,cfg in [('single',base),('parallel',parallel)]:
                    self.measure('survey',group,n,0,cfg,variant=label)
                print('SURVEY',group,n,'completed runs',self.count,flush=True)

    def holdout(self, selected):
        for group, sizes in HOLDOUT.items():
            base = Config(**selected[group]['single'])
            parallel = Config(**selected[group]['parallel'])
            for n in sizes:
                for rep in (0,1,2):
                    pair=[]
                    for label,cfg in [('single',base),('parallel',parallel),
                                      ('without_cycle3',replace(base,cycle3=False)),
                                      ('original_rss_inversion',replace(base,local_every=0,selection='raw',adaptive=False))]:
                        pair.append(self.measure('holdout',group,n,rep,cfg,variant=label))
                    assert pair[0]['local_calls'] == pair[1]['local_calls']
                    a,b=asdict(base),asdict(parallel)
                    assert {key for key in a if a[key] != b[key]} <= {'islands','migration_interval','migration_pct'}
                print('HOLDOUT',group,n,flush=True)
            for n in (sizes[0],sizes[-1]):
                for label,cfg in [('single',base),('parallel',parallel)]:
                    row=self.measure('diagonal',group,n,0,cfg,variant=label)
                    assert row['cost'] == n, row

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',default='results/local_work_20261028')
    p.add_argument('--stage',choices=('all','train','survey','holdout'),default='all')
    p.add_argument('--workers',type=int,default=4)
    args = p.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    # Refuse accidental duplicate observations when resuming a completed stage.
    stages=('train','survey','holdout','diagonal') if args.stage=='all' else (('holdout','diagonal') if args.stage=='holdout' else (args.stage,))
    for stage in stages:
        if (root/f'{stage}.csv').exists():
            raise SystemExit(f'{stage}.csv exists; choose a new output or explicitly resume another stage')
    warm_worker(0)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pids = sorted(set(pool.map(warm_worker,range(args.workers*2))))
        metadata={'status': 'running','python': sys.version,'platform': platform.platform(),
                      'worker_pids': pids,'workers': args.workers,
                      'pool_warmup_seconds': time.perf_counter()-start,
                      'train_sizes': TRAIN,'survey_sizes': CLASS_GRID,'holdout_sizes': HOLDOUT,
                      'levels': LEVELS,'stage_offsets': STAGE_OFFSET,
                      'repeats': {'train': 2,'survey': 1,'holdout': 3},
                      'timings': 'warm kernels and persistent workers; includes matrix serialization, migration, GA initialization, excludes exact solver and pool startup',
                      'objective': 'lowest mean TRAIN relative error; ties retain current; one coordinate pass, no global optimum claim',
                      'parallel_budget': 'identical single settings except islands and migration; equal global local calls'}
        (root/'metadata.json').write_text(json.dumps(metadata,indent=2))
        study=Study(root,pool,args.workers)
        if args.stage in ('all','train'):
            selected=study.train()
        else:
            selected=json.loads((root/'selected.json').read_text())
        if args.stage in ('all','survey'):
            study.survey(selected)
        if args.stage in ('all','holdout'):
            study.holdout(selected)
        metadata.update(status='completed',stage=args.stage,actual_new_runs=study.count,
                        wall_seconds=time.perf_counter()-start)
        (root/'metadata.json').write_text(json.dumps(metadata,indent=2))
        print('COMPLETE',study.count,metadata['wall_seconds'],flush=True)

if __name__=='__main__':
    main()
