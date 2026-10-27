"""Dense measured OFAT: 35 actual sizes and independent class holdouts.

Run:
  python lab1/dense_research.py --output results/dense_ofat_20261027
Requirements: numpy, scipy, matplotlib.
"""
from __future__ import annotations
from dataclasses import asdict,replace
from pathlib import Path
from collections import defaultdict
from statistics import mean
import argparse,csv,json,time
import numpy as np
from scipy.optimize import linear_sum_assignment
from hybrid_ga import Config,run

SIZES={
    "small":[5,10,20,30,40,50],
    "medium":list(range(100,501,50)),
    "transition":list(range(550,1000,50)),
    "large":list(range(1000,2001,100))
}
BASE=Config(population=120,generations=150,mutation_prob=1.,inversion_pct=2.,
            elitism=.2,selection="rank",rank_power=2.,adaptive=False,
            local_every=15,local_top=1,local_passes=2,local_candidates=8,
            islands=1,migration_interval=15,migration_pct=.1)
LEVELS={
    "population":[60,120,240],
    "generations":[40,100,150,250,500],
    "mutation_prob":[.3,.7,1.],
    "inversion_pct":[1.,2.,5.,20.],
    "elitism":[0.,.1,.2,.4],
    "selection":["raw","rank"],
    "rank_power":[.5,1.,2.,4.],
    "adaptive":[False,True],
    "local_every":[0,5,15,30],
    "local_candidates":[4,8,16,32],
    "local_passes":[1,2,4],
    "local_top":[1,2,4],
    "islands":[1,2,4,8],
    "migration_interval":[5,15,30,50],
    "migration_pct":[0.,.05,.1,.2]
}
CONTEXT={p:replace(BASE,islands=4) if p in
         ("migration_interval","migration_pct") else BASE for p in LEVELS}
def matrix(seed,n):
    return np.random.default_rng(seed).integers(1,101,(n,n),dtype=np.int32)

def save_csv(path,rows):
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def load_csv(path):
    with open(path,newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))

def experiment(directory,seed=20261028,workers=2):
    out=Path(directory);out.mkdir(parents=True,exist_ok=True)
    rows=[];start=time.perf_counter()
    expected=sum(len(v) for v in SIZES.values())*sum(len(v) for v in LEVELS.values())
    for cls,sizes in SIZES.items():
        for n in sizes:
            matrix_seed=seed+n*2711
            a=matrix(matrix_seed,n)
            rr,cc=linear_sum_assignment(a)
            exact=int(a[rr,cc].sum())
            ga_seed=matrix_seed+19
            for parameter,levels in LEVELS.items():
                reference=CONTEXT[parameter]
                for value in levels:
                    cfg=replace(reference,**{parameter:value})
                    differences={k for k in asdict(cfg) if
                                 getattr(cfg,k)!=getattr(reference,k)}
                    assert differences <= {parameter},(parameter,differences)
                    ans=run(a,cfg,seed=ga_seed,workers=workers)
                    rows.append(dict(class_name=cls,n=n,parameter=parameter,
                         value=str(value),cost=ans["cost"],exact=exact,
                         error_pct=100*(ans["cost"]-exact)/exact,
                         seconds=ans["seconds"],total_population=ans["total_population"],
                         generations=ans["generations"],
                         num_islands=ans["num_islands"],local_calls=ans["local_calls"],
                         matrix_seed=matrix_seed,ga_seed=ga_seed,
                         context_json=json.dumps(asdict(reference),sort_keys=True),
                         settings_json=json.dumps(asdict(cfg),sort_keys=True)))
            save_csv(out/"sweep_raw.csv",rows)
            print(f"{cls} n={n}: {len(rows)}/{expected} measured",flush=True)
    assert len(rows)==expected
    grouped=defaultdict(list)
    for row in rows:
        if row["class_name"]!="transition":
            grouped[(row["class_name"],row["parameter"],row["value"])].append(
                float(row["error_pct"]))
    ranked=[dict(class_name=cls,parameter=p,value=v,count=len(a),
                 mean_error_pct=mean(a)) for (cls,p,v),a in grouped.items()]
    ranked.sort(key=lambda x:(x["class_name"],x["parameter"],x["mean_error_pct"]))
    save_csv(out/"ofat_rankings.csv",ranked)
    (out/"methodology.json").write_text(json.dumps(dict(
        seed=seed,grid=SIZES,baseline=asdict(BASE),levels=LEVELS,
        number_of_real_runs=len(rows),
        note="Per level one measured matrix per dimension. 550–950: transition, not in training class.",
        baseline_context="Four islands held constant ONLY in migration sweeps",
        ofat="Exactly one dataclass field differs inside a sweep",
        graph_curves="PCHIP between measured dots only; smoothing does not create measured rows"),
        ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"OFAT done: {len(rows)} verified, {time.perf_counter()-start:.1f}s",flush=True)

def holdout(directory,seed=20261028,workers=2):
    out=Path(directory);ranks=load_csv(out/"ofat_rankings.csv")
    configs={}
    for cls in ("small","medium","large"):
        c=BASE
        for p in LEVELS:
            if p in ("islands","migration_interval","migration_pct"):continue
            best=min((r for r in ranks if r["class_name"]==cls and r["parameter"]==p),
                     key=lambda r:float(r["mean_error_pct"]))
            value=next(v for v in LEVELS[p] if str(v)==best["value"])
            c=replace(c,**{p:value})
        island_best=min((r for r in ranks
                         if r["class_name"]==cls and r["parameter"]=="islands"
                         and int(r["value"])>=2),
                        key=lambda r:float(r["mean_error_pct"]))
        k=int(island_best["value"])
        configs[cls]={"single":c,"islands":replace(c,islands=max(2,k))}
    rows=[]
    for cls,sizes in {"small":[30,50],"medium":[100,300,500],
                      "large":[1000,1500,2000]}.items():
        c=configs[cls]["single"];N,G=c.population,c.generations
        variants={
            "classic":Config(population=N,generations=G,
                      mutation_prob=1,inversion_pct=2,elitism=.2,
                      selection="raw",local_every=0),
            "hybrid_reference":replace(BASE,population=N,generations=G),
            "tuned_single":c,
            "tuned_islands":configs[cls]["islands"]
        }
        for n in sizes:
            for rep in range(3):
                matrix_seed=seed+888888+n*911+rep*509
                a=matrix(matrix_seed,n)
                rr,cc=linear_sum_assignment(a)
                exact=int(a[rr,cc].sum())
                for name,cfg in variants.items():
                    ans=run(a,cfg,seed=matrix_seed+41,workers=workers)
                    rows.append(dict(class_name=cls,n=n,repeat=rep,
                       variant=name,cost=ans["cost"],exact=exact,
                       error_pct=100*(ans["cost"]-exact)/exact,
                       seconds=ans["seconds"],population=ans["total_population"],
                       generations=G,num_islands=ans["num_islands"],
                       local_calls=ans["local_calls"],matrix_seed=matrix_seed,
                       settings_json=json.dumps(asdict(cfg),sort_keys=True)))
            save_csv(out/"holdout_raw.csv",rows)
            print(f"HOLDOUT {cls} n={n}, {len(rows)} observations",flush=True)
    (out/"best_observed_config.json").write_text(
        json.dumps({cls:{name:asdict(cfg) for name,cfg in cc.items()}
                    for cls,cc in configs.items()},indent=2),encoding="utf-8")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results/dense_ofat_20261027")
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--stage",choices=("sweep","holdout","all"),default="all")
    args=p.parse_args()
    if args.stage in ("sweep","all"):experiment(args.output,workers=args.workers)
    if args.stage in ("holdout","all"):holdout(args.output,workers=args.workers)
