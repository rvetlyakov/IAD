"""Measured acceleration and simultaneous process execution, outside tuning."""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from hybrid_ga import Config, best_columns, improve_swaps, run
from local_research import warm_worker
from three_cycle import improve_three_cycle


def timed_task(payload):
    fn,state=payload
    start=time.perf_counter()
    result=fn(state)
    return result, {'pid': os.getpid(),'start': start,'end': time.perf_counter()}

class TracePool:
    def __init__(self,pool):
        self.pool=pool
        self.events=[]
    def map(self,fn,tasks):
        outcomes=list(self.pool.map(timed_task,[(fn,task) for task in tasks]))
        self.events.extend(event for _,event in outcomes)
        return [result for result,_ in outcomes]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/local_work_20261028')
    args=parser.parse_args()
    root=Path(args.output)
    a=np.random.default_rng(91021).integers(1,101,(300,300),dtype=np.int32)
    p=np.random.default_rng(882).permutation(300).astype(np.int32)
    columns=best_columns(a,16)
    kernels=[]
    for fn in (improve_swaps,improve_three_cycle):
        fn(a,p,columns,3)
        t=time.perf_counter()
        actual,moves=fn(a,p,columns,3)
        compiled=time.perf_counter()-t
        t=time.perf_counter()
        reference,reference_moves=fn.py_func(a,p,columns,3)
        python=time.perf_counter()-t
        assert np.array_equal(actual,reference) and moves==reference_moves
        kernels.append({'kernel': fn.__name__,'n': 300,'k': 16,'passes': 3,
                            'python_seconds': python,'compiled_seconds': compiled,
                            'speed_ratio': python/compiled,'identical': True})
    selected=json.loads((root/'selected.json').read_text())
    cfg=Config(**selected['medium']['parallel'])
    # Separate performance probe, never used for tuning.
    a=np.random.default_rng(918292).integers(1,101,(400,400),dtype=np.int32)
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        warm_pids=sorted(set(pool.map(warm_worker,range(8))))
        warmup=time.perf_counter()-start
        trace=TracePool(pool)
        parallel=run(a,cfg,9101,workers=4,executor=trace)
        single=run(a,replace(cfg,islands=1),9101)
    events=trace.events
    overlap=[(i,j) for i,first in enumerate(events) for j,second in enumerate(events)
             if i<j and first['pid']!=second['pid']
             and max(first['start'],second['start'])<min(first['end'],second['end'])]
    assert overlap, 'No simultaneous work observed in distinct processes'
    # Compare the identical seeded parallel algorithm with a fresh pool.
    start=time.perf_counter()
    cold=run(a,cfg,9101,workers=4)
    cold_seconds=time.perf_counter()-start
    assert cold['cost']==parallel['cost']
    np.testing.assert_array_equal(cold['best'],parallel['best'])
    assert single['local_calls']==parallel['local_calls']
    result={'kernels': kernels,'worker_warmup_seconds': warmup,'warmup_worker_pids': warm_pids,
                'independent_probe_matrix_seed': 918292,'n': 400,'ga_seed': 9101,
                'settings': selected['medium']['parallel'],
                'warm_parallel_seconds': parallel['seconds'],'warm_single_seconds': single['seconds'],
                'cold_parallel_wall_seconds': cold_seconds,
                'warm_parallel_cost': parallel['cost'],'cold_parallel_cost': cold['cost'],
                'simultaneous_distinct_process_pairs': len(overlap),'events': events,
                'total_population': parallel['total_population'],'generations': parallel['generations'],
                'single_local_calls': single['local_calls'],'parallel_local_calls': parallel['local_calls'],
                'note': 'One performance probe; ratios are hardware/load dependent, not a statistical speed claim'}
    (root/'execution_probe.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='events'},indent=2))

if __name__=='__main__':
    main()
