"""New control matrices; TRAIN-chosen inversion lengths, paired repair ablation."""
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean

import numpy as np
from hybrid_ga import Config, run
from scipy.optimize import linear_sum_assignment


def main():
    root=Path('results/class_diagnosis_20261028')
    old=json.loads(Path('results/local_work_20261028/selected.json').read_text())
    rows=[]
    for group,sizes,pct in [('small',(30,70,95),50),('medium',(125,275,475),5)]:
        cfg=Config(population=64,generations=100,selection='rank',rank_power=2,
                   inversion_pct=pct,local_every=5,local_top=2,local_candidates=16,
                   cycle3=True,cycle3_passes=3,local_on_offspring=True)
        for n in sizes:
            for rep in range(5):
                seed=970000000+n*1009+rep*100003
                a=np.random.default_rng(seed).integers(1,101,(n,n),dtype=np.int32)
                r,c=linear_sum_assignment(a);opt=int(a[r,c].sum())
                for name,c in [('previous_class',Config(**old[group]['single'])),
                               ('matched_elite_repair',replace(cfg,local_on_offspring=False)),
                               ('offspring_repair',cfg)]:
                    ans=run(a,c,seed+17)
                    assert int(a[np.arange(n),ans['best']].sum())==ans['cost']
                    assert sorted(ans['best'])==list(range(n))
                    rows.append({'group': group,'n': n,'rep': rep,'seed': seed,'ga_seed': seed+17,'variant': name,
                                     'cost': ans['cost'],'exact': opt,'error_pct': 100*(ans['cost']-opt)/opt,
                                     'local_calls': ans['local_calls'],'seconds': ans['seconds'],
                                     'settings_json': json.dumps(asdict(c),sort_keys=True)})
            print('VALIDATE',group,n,flush=True)
    with (root/'offspring_repair_holdout.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for group in ('small','medium'):
        for variant in ('previous_class','matched_elite_repair','offspring_repair'):
            data=[r for r in rows if r['group']==group and r['variant']==variant]
            summary.append({'group': group,'variant': variant,'runs': len(data),
                        'mean_error_pct': mean(r['error_pct'] for r in data),
                        'mean_seconds': mean(r['seconds'] for r in data),
                        'local_calls': data[0]['local_calls']})
    (root/'offspring_repair_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
