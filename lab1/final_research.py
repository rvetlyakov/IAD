"""Final offspring-repair study on fresh TRAIN/SURVEY/HOLDOUT matrices."""
import argparse
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import local_research as lr
from hybrid_ga import Config

FINAL_OFFSETS={'train':1_100_000_000,'survey':1_300_000_000,
               'holdout':1_500_000_000,'diagonal':1_700_000_000}
FINAL_TRAIN={'small':(20,55,85),'medium':(150,300,450),'large':(650,1100,1700)}


def starting_config(group):
    return Config(population=64,generations=100,selection='rank',rank_power=2,
                  inversion_pct={'small':50.,'medium':5.,'large':5.}[group],
                  local_every=5,local_top=2,local_candidates=16,local_passes=2,
                  cycle3=True,cycle3_passes=3,local_on_offspring=True,migration_interval=15)


def configure():
    lr.STAGE_OFFSET.update(FINAL_OFFSETS)
    lr.TRAIN.update(FINAL_TRAIN)
    lr.problem.cache_clear()
    lr.LEVELS['generations']=(1,2,5,10,20,40,100)
    lr.LEVELS['inversion_pct']=(1.,5.,20.,50.,100.)
    lr.LEVELS['local_on_offspring']=(False,True)


class FinalStudy(lr.Study):
    def initial_config(self,group):
        return starting_config(group)

    def generation_levels(self,group):
        return lr.LEVELS['generations']

    def survey(self,selected):
        common=starting_config('medium')
        for group,sizes in lr.CLASS_GRID.items():
            base=Config(**selected[group]['single'])
            parallel=Config(**selected[group]['parallel'])
            for n in sizes:
                for rep in (0,1):
                    for parameter,values in lr.LEVELS.items():
                        context=lr.survey_context(base,parallel,parameter)
                        # Every ablation is active, even if TRAIN disabled local repair.
                        if parameter=='local_on_offspring' and not context.local_every:
                            context=replace(context,local_every=5)
                        for value in dict.fromkeys((getattr(context,parameter),)+tuple(values)):
                            self.measure('survey',group,n,rep,replace(context,**{parameter:value}),parameter,context)
                    for label,cfg in [('single',base),('parallel',parallel),('common',common)]:
                        self.measure('survey',group,n,rep,cfg,variant=label)
                print('SURVEY',group,n,'completed',self.count,flush=True)

    def holdout(self,selected):
        for group,sizes in lr.HOLDOUT.items():
            base=Config(**selected[group]['single'])
            parallel=Config(**selected[group]['parallel'])
            for n in sizes:
                for rep in range(5):
                    pair=[]
                    variants=[('single',base),('parallel',parallel),
                              ('without_cycle3',replace(base,cycle3=False)),
                              ('without_offspring',replace(base,local_on_offspring=False)),
                              ('original_rss_inversion',replace(base,local_every=0,selection='raw',adaptive=False)),
                              ('common',starting_config('medium'))]
                    for label,cfg in variants:
                        pair.append(self.measure('holdout',group,n,rep,cfg,variant=label))
                    assert pair[0]['local_calls']==pair[1]['local_calls']==pair[3]['local_calls']
                    a,b=asdict(base),asdict(parallel)
                    assert {k for k in a if a[k]!=b[k]}<={'islands','migration_interval','migration_pct'}
                print('HOLDOUT',group,n,flush=True)
            for n in (sizes[0],sizes[-1]):
                for label,cfg in [('single',base),('parallel',parallel)]:
                    row=self.measure('diagonal',group,n,0,cfg,variant=label)
                    assert row['cost']==n,row


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results/final_offspring_20261028')
    p.add_argument('--stage',choices=('all','train','survey','holdout'),default='all')
    args=p.parse_args()
    configure()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    stages=('train','survey','holdout','diagonal') if args.stage=='all' else (args.stage,)
    for stage in stages:
        if (root/f'{stage}.csv').exists():raise SystemExit(f'{stage}.csv already exists')
    lr.warm_worker(0)
    begin=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        pids=sorted(set(pool.map(lr.warm_worker,range(8))))
        meta={'status':'running','profile':'final_offspring','python':sys.version,
              'platform':platform.platform(),'repeats':{'train':2,'survey':2,'holdout':5},
              'train_sizes':lr.TRAIN,'survey_sizes':lr.CLASS_GRID,'holdout_sizes':lr.HOLDOUT,
              'stage_offsets':lr.STAGE_OFFSET,'levels':lr.LEVELS,'workers':4,'worker_pids':pids,
              'pool_warmup_seconds':time.perf_counter()-begin,
              'selection':'one coordinate TRAIN pass; minimum mean error; retain current on ties',
              'timings':'warm persistent workers, includes serialization and migration, excludes SciPy and pool startup',
              'common_settings':asdict(starting_config('medium'))}
        (root/'metadata.json').write_text(json.dumps(meta,indent=2))
        study=FinalStudy(root,pool,4)
        selected=study.train() if args.stage in ('all','train') else json.loads((root/'selected.json').read_text())
        if args.stage in ('all','survey'):study.survey(selected)
        if args.stage in ('all','holdout'):study.holdout(selected)
        meta.update(status='completed',stage=args.stage,actual_runs=study.count,wall_seconds=time.perf_counter()-begin)
        (root/'metadata.json').write_text(json.dumps(meta,indent=2))
        print('COMPLETE',meta['actual_runs'],meta['wall_seconds'],flush=True)


if __name__=='__main__':
    main()
