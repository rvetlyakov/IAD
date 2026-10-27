"""Fast reproducible paired medium-GA verification; 5 independent matrices per size.

For diagnostic fair N/G comparison, toggle only cycle3, keep ALL other
parameters including the exact matrix and random seed identical.
"""
from __future__ import annotations
import argparse,csv,json
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from hybrid_ga import Config,run

BASE=Config(population=120,generations=250,mutation_prob=.7,
            inversion_pct=2,elitism=.2,selection="rank",rank_power=2,
            adaptive=False,local_every=25,local_top=1,
            local_passes=2,local_candidates=16,cycle3=False,cycle3_passes=2)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results/medium_cycle_holdout_verified_20261028")
    p.add_argument("--workers",type=int,default=2)
    args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for n in (100,300,500):
        for repeat in range(5):
            seed=20261201+n*1001+repeat*100003
            a=np.random.default_rng(seed).integers(1,101,(n,n),dtype=np.int32)
            r,c=linear_sum_assignment(a)
            exact=int(a[r,c].sum(dtype=np.int64))
            for cycle3 in (False,True):
                settings=replace(BASE,cycle3=cycle3)
                result=run(a,settings,seed=seed+11,workers=args.workers)
                rows.append({
                   "size":n,"repeat":repeat,"cycle3":cycle3,
                   "matrix_seed":seed,"ga_seed":seed+11,
                   "exact_cost":exact,"cost":result["cost"],
                   "error_percent":100*(result["cost"]-exact)/exact,
                   "seconds":result["seconds"],"population":result["total_population"],
                   "generations":result["generations"],
                   "local_calls":result["local_calls"],
                   "settings_json":json.dumps(asdict(settings),sort_keys=True)
                })
            print(f"Verified n={n}, matrix {repeat+1}/5",flush=True)
    with (out/"raw.csv").open("w",newline="",encoding="utf8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    groups={}
    for n in (100,300,500):
        for cycle3 in (False,True):
            a=[r for r in rows if r["size"]==n and r["cycle3"]==cycle3]
            groups[f"{n}_cycle3_{cycle3}"]={
                "mean_error_percent":sum(r["error_percent"] for r in a)/len(a),
                "mean_seconds":sum(r["seconds"] for r in a)/len(a),
                "runs":len(a)
            }
    (out/"metadata.json").write_text(json.dumps({
      "method":"Paired seed and matrix, only Config.cycle3 changed",
      "full_config":asdict(BASE),"comparison":groups,
      "source":"actual executed GA with 3-cycle inside generations",
      "class":"medium n=100/300/500",
      "limits":"five independent matrices per size; further sizes in dense study"
    },indent=2,ensure_ascii=False),encoding="utf8")
    assert len(rows)==30
    for k in range(0,len(rows),2):
        x,y=rows[k:k+2]
        assert x["matrix_seed"]==y["matrix_seed"] and x["ga_seed"]==y["ga_seed"]
        assert x["population"]==y["population"] and x["generations"]==y["generations"]
        assert x["local_calls"]==y["local_calls"]
    print("Completed 30 real paired medium experiments",flush=True)

if __name__=="__main__":
    main()
