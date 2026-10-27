"""Replay six independent control rows after final source changes."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from hybrid_ga import Config, run
from local_evidence import read
from local_research import problem, warm_worker


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/local_work_20261028')
    args=parser.parse_args()
    root=Path(args.output)
    metadata=json.loads((root/'metadata.json').read_text())
    if metadata.get('profile')=='final_offspring':
        from final_research import configure
        configure()
    rows=read(root/'holdout.csv')
    selected=[]
    for group in ('small','medium','large'):
        n=max(int(r['n']) for r in rows if r['class_name']==group)
        selected.extend(r for r in rows if r['class_name']==group and int(r['n'])==n
                        and r['repeat']=='0' and r['variant'] in ('single','parallel'))
    results=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(warm_worker,range(8)))
        for row in selected:
            n,rep=int(row['n']),int(row['repeat'])
            a,optimum=problem('holdout',n,rep)
            cfg=Config(**json.loads(row['settings_json']))
            ans=run(a,cfg,int(row['ga_seed']),workers=4,executor=pool)
            assert ans['cost']==int(row['cost'])
            assert ans['local_calls']==int(row['local_calls'])
            assert np.array_equal(np.sort(ans['best']),np.arange(n))
            results.append({'class_name':row['class_name'],'n':n,'variant':row['variant'],
                            'cost':ans['cost'],'original_cost':int(row['cost']),
                            'exact':optimum,'local_calls':ans['local_calls'],
                            'reproduced':True})
    (root/'replay.json').write_text(json.dumps(results,indent=2))
    print('Reproduced',len(results),'original measured control costs exactly')


if __name__=='__main__':
    main()
