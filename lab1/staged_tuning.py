"""Two-stage, leakage-resistant GA tuning followed by strict one-factor plots.

Stage A: evaluate a G x K factorial on TRAIN matrices and choose a BASELINE.
Stage B: freeze that selected baseline, change ONE parameter at a time on
distinct SURVEY matrices, and store all source parameters/seed.
Stage C: evaluate the chosen baseline on INDEPENDENT HOLDOUT matrices.

Training and survey must not share matrices. Survey results must NOT
be used to rewrite the selected baseline without another held-out stage.

Supports all classes, medium includes a real 1000-generation candidate.
No claim of global optimality is made.
"""
from __future__ import annotations
import argparse, csv, json
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean
import numpy as np
from scipy.optimize import linear_sum_assignment
from hybrid_ga import Config, run

CLASS_GRID={
 "small":tuple(range(10,100,10)),
 "medium":tuple(range(100,500,25)),
 "large":tuple(range(500,2001,125)),
}
# Boundaries are [2,100), [100,500), [500,2000] -- no overlap at 500.
TRAIN={"small":(20,60,90),"medium":(150,300,450),
       "large":(500,1000,1500,2000)}
HOLDOUT={"small":(30,70),"medium":(200,400),
         "large":(625,1250,1750)}
GENERATIONS={
 "small":(100,250,500),
 "medium":(100,250,500,750,1000),
 "large":(250,500,750,1000),
}
CANDIDATES=(8,16,32)
SEEDS={"train":(33,44),"survey":(55,),"holdout":(77,88,99)}
LEVELS={
 "population":(60,120,240),
 "generations":(100,250,500,750,1000),
 "mutation_prob":(.3,.7,1.0),
 "inversion_pct":(1.,2.,5.,20.),
 "elitism":(0.,.1,.2,.4),
 "selection":("raw","rank"),
 "rank_power":(.5,1.,2.,4.),
 "adaptive":(False,True),
 "local_every":(0,5,15,30),
 "local_candidates":(4,8,16,32),
 "local_passes":(1,2,4),
 "local_top":(1,2,4),
 "islands":(1,2,4,8),
 "migration_interval":(5,15,30,50),
 "migration_pct":(0.,.05,.1,.2),
 "adaptive_long_prob":(0.,.05,.15),
 "adaptive_long_pct":(5.,20.,40.)
}
# Only scan active variants: meaningless inactive parameters are excluded.
ACTIVE_CONTEXT={
 "rank_power":{"selection":"rank"},
 "migration_interval":{"islands":4},
 "migration_pct":{"islands":4},
 "adaptive_long_prob":{"adaptive":True},
 "adaptive_long_pct":{"adaptive":True}
}

def _seed(n,repeat,stage):
    return 20261029+n*3011+repeat*977+{"train":0,"survey":100_000_000,"holdout":200_000_000}[stage]

def problem(n,matrix_seed):
    a=np.random.default_rng(matrix_seed).integers(1,101,(n,n),dtype=np.int32)
    r,c=linear_sum_assignment(a)
    return a,int(a[r,c].sum(dtype=np.int64))

def save(path,records):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not records:return
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

def read(path):
    with Path(path).open(encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))

def class_for(n):
    if 2<=n<100:return "small"
    if 100<=n<500:return "medium"
    if 500<=n<=2000:return "large"
    raise ValueError(f"n outside research classes: {n}")

def evaluate(a,opt,cfg,seed,workers):
    result=run(a,cfg,seed=seed,workers=workers)
    assert result["total_population"]==cfg.population
    assert result["generations"]==cfg.generations
    assert result["num_islands"]==cfg.islands
    return result,100*(result["cost"]-opt)/opt

def starter(group,source):
    data=json.loads(Path(source).read_text(encoding="utf-8"))
    return Config(**data[group]["single"])

def train(group,root,source,workers=2):
    """Compare all G and K pairs with identical test instances, seeds, and
    all OTHER knobs fixed; choose after aggregating ALL training matrices.
    """
    if group not in TRAIN:raise ValueError(group)
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    base=starter(group,source)
    records=[]
    for n in TRAIN[group]:
        for repeat,token in enumerate(SEEDS["train"]):
            mat_seed=_seed(n,token,"train")
            a,opt=problem(n,mat_seed)
            ga_seed=mat_seed+9
            for gens in GENERATIONS[group]:
                for k in CANDIDATES:
                    cfg=replace(base,generations=gens,local_candidates=k)
                    result,err=evaluate(a,opt,cfg,ga_seed,workers)
                    records.append({
                      "class_name":group,"n":n,"repeat":repeat,
                      "generations":gens,"local_candidates":k,
                      "cost":result["cost"],"exact":opt,"error_pct":err,
                      "seconds":result["seconds"],
                      "matrix_seed":mat_seed,"ga_seed":ga_seed,
                      "settings_json":json.dumps(asdict(cfg),sort_keys=True)
                    })
        save(root/f"{group}_train.csv",records)
        print(f"TRAIN {group} n={n}, observations={len(records)}",flush=True)
    ranked=[]
    for gens in GENERATIONS[group]:
        for k in CANDIDATES:
            observed=[r for r in records
                      if r["generations"]==gens and r["local_candidates"]==k]
            ranked.append({"generations":gens,"local_candidates":k,
                           "mean_error_pct":mean(r["error_pct"] for r in observed),
                           "mean_seconds":mean(r["seconds"] for r in observed),
                           "samples":len(observed)})
    ranked.sort(key=lambda r:(r["mean_error_pct"],r["mean_seconds"]))
    save(root/f"{group}_grid_rankings.csv",ranked)
    winner=ranked[0]
    tuned=replace(base,generations=int(winner["generations"]),
                  local_candidates=int(winner["local_candidates"]))
    (root/f"{group}_frozen_baseline.json").write_text(
       json.dumps({"phase":"selected exclusively on training matrices",
         "class":group,"starting_settings":asdict(base),
         "selected_settings":asdict(tuned),"winner":winner,
         "train_sizes":TRAIN[group],"train_repeats":len(SEEDS["train"]),
         "training_grid":ranked},indent=2,ensure_ascii=False),encoding="utf-8")
    return winner

def ofat(group,root,workers=2):
    """Freeze TRAIN winner, perturb exactly one active field per trial.

    Survey matrices have never participated in hyperparameter selection.
    """
    root=Path(root)
    meta=json.loads((root/f"{group}_frozen_baseline.json").read_text())
    base=Config(**meta["selected_settings"])
    records=[]
    for n in CLASS_GRID[group]:
        for repeat,token in enumerate(SEEDS["survey"]):
            mat_seed=_seed(n,token,"survey")
            a,opt=problem(n,mat_seed)
            ga_seed=mat_seed+9
            for parameter,levels in LEVELS.items():
                context=replace(base,**ACTIVE_CONTEXT.get(parameter,{}))
                for value in levels:
                    cfg=replace(context,**{parameter:value})
                    changed={key for key in asdict(context)
                             if getattr(context,key)!=getattr(cfg,key)}
                    assert changed<={parameter},(parameter,changed)
                    result,err=evaluate(a,opt,cfg,ga_seed,workers)
                    records.append({
                        "class_name":group,"n":n,"repeat":repeat,
                        "parameter":parameter,"value":str(value),
                        "cost":result["cost"],"exact":opt,
                        "error_pct":err,"seconds":result["seconds"],
                        "matrix_seed":mat_seed,"ga_seed":ga_seed,
                        "population":result["total_population"],
                        "generations":result["generations"],
                        "local_calls":result["local_calls"],
                        "num_islands":result["num_islands"],
                        "context_json":json.dumps(asdict(context),sort_keys=True),
                        "settings_json":json.dumps(asdict(cfg),sort_keys=True)
                    })
        save(root/f"{group}_ofat.csv",records)
        print(f"OFAT {group} n={n}, samples={len(records)}",flush=True)
    return len(records)

def validate(group,root,workers=2):
    """Holdout data is NEVER used to choose settings in this module."""
    root=Path(root)
    m=json.loads((root/f"{group}_frozen_baseline.json").read_text())
    base=Config(**m["starting_settings"])
    selected=Config(**m["selected_settings"])
    rows=[]
    for n in HOLDOUT[group]:
        for repeat,token in enumerate(SEEDS["holdout"]):
            mat_seed=_seed(n,token,"holdout")
            a,opt=problem(n,mat_seed)
            for name,cfg in [("before",base),("after",selected)]:
                ans,err=evaluate(a,opt,cfg,mat_seed+9,workers)
                rows.append({"class_name":group,"n":n,"repeat":repeat,
                    "variant":name,"cost":ans["cost"],"exact":opt,
                    "error_pct":err,"seconds":ans["seconds"],
                    "matrix_seed":mat_seed,"ga_seed":mat_seed+9,
                    "settings_json":json.dumps(asdict(cfg),sort_keys=True)})
        save(root/f"{group}_holdout.csv",rows)
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--class-name",choices=tuple(TRAIN),default="medium")
    p.add_argument("--output",default="results/staged_tuning_v2")
    p.add_argument("--source",default="results/dense_ofat_20261027/best_observed_config.json")
    p.add_argument("--stage",choices=("train","ofat","holdout","all"),default="all")
    p.add_argument("--workers",type=int,default=2)
    args=p.parse_args()
    if args.stage in ("train","all"):
        print("SELECTED",train(args.class_name,args.output,args.source,args.workers),flush=True)
    if args.stage in ("ofat","all"):ofat(args.class_name,args.output,args.workers)
    if args.stage in ("holdout","all"):validate(args.class_name,args.output,args.workers)

if __name__=="__main__":
    main()
