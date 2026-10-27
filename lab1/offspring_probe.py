"""TRAIN-only experimental repair of mutated offspring before RSS."""
import csv
import inspect
from dataclasses import replace
from pathlib import Path
from statistics import mean
from unittest.mock import patch

import hybrid_ga as ga
import numpy as np
from scipy.optimize import linear_sum_assignment


def experimental_evolve(fraction):
    source=inspect.getsource(ga._evolve)
    original='selected=np.argsort(pc)[:min(quota,len(pc))]'
    replacement='''selected=np.argsort(pc)[:min(quota,len(pc))]
            offspring_quota=min(int(round(quota*offspring_fraction)),len(mutants))
            if offspring_quota:
                offspring_indices=rng.choice(np.arange(len(pop),len(pop)+len(mutants)),offspring_quota,replace=False)
                selected=np.concatenate((selected[:len(selected)-offspring_quota],offspring_indices))'''
    assert original in source
    namespace=dict(ga.__dict__);namespace['offspring_fraction']=fraction
    exec(source.replace(original,replacement),namespace)  # noqa: S102 - controlled local algorithm probe
    return namespace['_evolve']


def main():
    out=Path('results/class_diagnosis_20261028');out.mkdir(exist_ok=True)
    rows=[]
    base=ga.Config(population=64,generations=100,selection='rank',rank_power=2,
                   inversion_pct=1,local_every=5,local_top=2,local_candidates=16,
                   cycle3=True,cycle3_passes=3)
    for n in (50,150,300,450):
        for rep in range(3):
            seed=950000000+n*1009+rep*100003
            a=np.random.default_rng(seed).integers(1,101,(n,n),dtype=np.int32)
            r,c=linear_sum_assignment(a);opt=int(a[r,c].sum())
            for pct in (1,5,20,50):
                for fraction in (0.,.5,1.):
                    cfg=replace(base,inversion_pct=pct)
                    with patch.object(ga,'_evolve',experimental_evolve(fraction)):
                        ans=ga.run(a,cfg,seed+17)
                    assert sorted(ans['best'])==list(range(n))
                    assert ans['local_calls']==40
                    rows.append({'n': n,'rep': rep,'seed': seed,'inversion_pct': pct,
                                'offspring_fraction': fraction,'cost': ans['cost'],'exact': opt,
                                'error_pct': 100*(ans['cost']-opt)/opt,'seconds': ans['seconds']})
        print('PROBE',n,flush=True)
    with (out/'offspring_repair_train.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for n in (50,150,300,450):
        for pct in (1,5,20,50):
            print(n,pct,[(q,round(mean(r['error_pct'] for r in rows if r['n']==n and r['inversion_pct']==pct and r['offspring_fraction']==q),2)) for q in (0.,.5,1.)])


if __name__=='__main__':
    main()
