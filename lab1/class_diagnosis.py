"""Independent diagnosis of class differences and hybrid stagnation."""
import csv
import inspect
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean

import hybrid_ga as ga
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path('results/class_diagnosis_20261028')


def save(name, rows):
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def traced(a,c,seed):
    # Instrument the ACTUAL evolution source; do not substitute another GA.
    source=inspect.getsource(ga._evolve)
    source=source.replace('pop=pool[selected].copy()', '''pop=pool[selected].copy()
        trace.append(dict(generation=gen+1,best_cost=best[0],
                          unique_individuals=len(np.unique(pop,axis=0)),
                          local_calls=local_calls))''')
    namespace=dict(ga.__dict__);namespace['trace']=[]
    exec(source,namespace)  # noqa: S102 - instrument trusted local GA source
    rng=np.random.default_rng(seed)
    pop=np.asarray([rng.permutation(len(a)) for _ in range(c.population)],dtype=np.int32)
    cc=ga.costs(a,pop);ib=int(np.argmin(cc));best=(int(cc[ib]),pop[ib].copy())
    state=(a,pop,c,0,c.generations,np.random.default_rng(seed+1).bit_generator.state,
           ga.best_columns(a,c.local_candidates) if c.local_every else None,best,0)
    result=namespace['_evolve'](state)
    reference=ga.run(a,c,seed)
    assert result[2][0]==reference['cost']
    np.testing.assert_array_equal(result[2][1],reference['best'])
    return result,namespace['trace']


def main():
    selected=json.loads(Path('results/local_work_20261028/selected.json').read_text())
    common=ga.Config(population=64,generations=100,selection='rank',rank_power=2,
                     inversion_pct=1,elitism=.2,local_every=5,local_top=2,
                     local_candidates=16,cycle3=True,cycle3_passes=3)
    configs={'common':common,'large_transfer':ga.Config(**selected['large']['single']),
             'medium_transfer':ga.Config(**selected['medium']['single'])}
    rows=[];structure=[];traces=[]
    for high in (100,100000):
        for n in (20,50,90,150,300,450,700,1000,1600,2000):
            for rep in range(3):
                seed=810000000+n*1009+rep*100003
                a=np.random.default_rng(seed).integers(1,high+1,(n,n),dtype=np.int32)
                r,c=linear_sum_assignment(a);opt=int(a[r,c].sum())
                structure.append({'n': n,'rep': rep,'high': high,'seed': seed,'exact': opt,
                                 'exact_per_row': opt/n,'ones_per_row': float((a==1).sum()/n),
                                 'zero_one_rows': int(np.sum(np.sum(a==1,axis=1)==0))})
                group='small' if n<100 else ('medium' if n<500 else 'large')
                for name,cfg in {**configs,'class_selected':ga.Config(**selected[group]['single'])}.items():
                    ans=ga.run(a,cfg,seed+17)
                    rows.append({'high': high,'n': n,'rep': rep,'variant': name,'cost': ans['cost'],'exact': opt,
                                     'error_pct': 100*(ans['cost']-opt)/opt,
                                     'excess_per_row': (ans['cost']-opt)/n,'seconds': ans['seconds'],
                                     'local_calls': ans['local_calls'],'settings_json': json.dumps(asdict(cfg))})
            print('DIAGNOSE',high,n,flush=True)
    for n in (50,300,1000):
        seed=910000000+n
        a=np.random.default_rng(seed).integers(1,101,(n,n),dtype=np.int32)
        _result,trace=traced(a,replace(common,generations=500),seed+17)
        for t in trace:traces.append(dict(n=n,**t))
    save('comparison.csv',rows);save('matrix_structure.csv',structure);save('generation_trace.csv',traces)
    summary=[]
    for high in (100,100000):
        for n in sorted({r['n'] for r in rows}):
            for v in configs.keys()|{'class_selected'}:
                data=[r for r in rows if r['high']==high and r['n']==n and r['variant']==v]
                summary.append({'high': high,'n': n,'variant': v,'mean_error_pct': mean(r['error_pct'] for r in data),
                                    'mean_excess_per_row': mean(r['excess_per_row'] for r in data)})
    save('summary.csv',summary)
    mutations=[]
    for n in (20,50,90,150,300,450,700,1000,2000):
        for pct in (1,5,20):
            mutations.append({'n': n,'inversion_pct': pct,'max_segment': min(n,max(2,math.ceil(n*pct/100)))})
    save('mutation_lengths.csv',mutations)
    print('COMPLETE',len(rows),'paired diagnostic runs',flush=True)


if __name__=='__main__':
    main()
