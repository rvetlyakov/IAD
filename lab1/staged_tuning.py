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

# Actual non-overlapping measured sizes, with NO transition category.
CLASS_GRID={
 "small":tuple(range(10,100,10)),
 "medium":tuple(range(100,500,25)),
 "large":(500,)+tuple(range(550,1000,50))+tuple(range(1000,2001,100)),
}
# Class boundaries: [2,100), [100,500), [500,2000].
TRAIN={"small":(20,60,90),"medium":(150,300,450),
       "large":(550,750,950,1250,1750)}
HOLDOUT={"small":(30,70),"medium":(200,400),
         "large":(600,800,1100,1600,1900)}
GENERATIONS={
 "small":(100,250,500),
 "medium":(100,250,500,750,1000),
 "large":(250,500,750,1000),
}
CANDIDATES=(8,16,32)
SEEDS={"train":(33,44),"survey":(55,66),"holdout":(77,88,99)}
LEVELS={
 "population":(60,120,240),
 "generations":(100,250,500,750,1000),
 "mutation_prob":(.3,.7,1.0),
 "inversion_pct":(1.,2.,5.,20.),
 "elitism":(0.,.1,.2,.4),
 "selection":("raw","rank"),
 "rank_power":(.5,1.,2.,4.),
 "adaptive":(False,True),
 "adaptive_horizon":(100,250,500,1000),
 "local_every":(0,5,15,30),
 "local_candidates":(4,8,16,32),
 "local_passes":(1,2,4),
 "local_top":(1,2,4),
 "islands":(1,2,4,8),
 "migration_interval":(5,15,30,50),
 "migration_pct":(0.,.05,.1,.2),
 "adaptive_long_prob":(0.,.05,.15),
 "adaptive_long_pct":(5.,20.,40.),
 "cycle3":(False,True),
 "cycle3_passes":(1,2,3),
}
# Only scan active variants: meaningless inactive parameters are excluded.
ACTIVE_CONTEXT={
 "rank_power":{"selection":"rank"},
 "migration_interval":{"islands":4},
 "migration_pct":{"islands":4},
 "adaptive_long_prob":{"adaptive":True},
 "adaptive_long_pct":{"adaptive":True},
 "adaptive_horizon":{"adaptive":True},
 "cycle3_passes":{"cycle3":True}
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
    """Strict one-factor training: choose G with K fixed, then K with G fixed.

    Every comparison is on the same TRAIN matrices and random seeds; no
    two-dimensional grid is used to choose hyperparameters.
    """
    if group not in TRAIN:raise ValueError(group)
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    base=starter(group,source)
    records=[]
    def measure_candidates(parameter,values,context):
        ranked=[]
        for value in values:
            cfg=replace(context,**{parameter:value})
            assert {key for key in asdict(context)
                    if getattr(cfg,key)!=getattr(context,key)}<={parameter}
            errors=[];times=[]
            for n in TRAIN[group]:
                for repeat,token in enumerate(SEEDS["train"]):
                    mat_seed=_seed(n,token,"train")
                    a,opt=problem(n,mat_seed)
                    ga_seed=mat_seed+9
                    result,err=evaluate(a,opt,cfg,ga_seed,workers)
                    errors.append(err);times.append(result["seconds"])
                    records.append({
                        "class_name":group,"n":n,"repeat":repeat,
                        "parameter":parameter,"value":str(value),
                        "cost":result["cost"],"exact":opt,"error_pct":err,
                        "seconds":result["seconds"],"matrix_seed":mat_seed,
                        "ga_seed":ga_seed,
                        "context_json":json.dumps(asdict(context),sort_keys=True),
                        "settings_json":json.dumps(asdict(cfg),sort_keys=True)})
            ranked.append(dict(parameter=parameter,value=value,
                               mean_error_pct=mean(errors),
                               mean_seconds=mean(times),samples=len(errors)))
        ranked.sort(key=lambda x:(x["mean_error_pct"],x["mean_seconds"]))
        return ranked

    first=measure_candidates("generations",GENERATIONS[group],base)
    after_g=replace(base,generations=int(first[0]["value"]))
    second=measure_candidates("local_candidates",CANDIDATES,after_g)
    tuned=replace(after_g,local_candidates=int(second[0]["value"]))
    rankings=first+second
    save(root/f"{group}_train.csv",records)
    save(root/f"{group}_train_rankings.csv",rankings)
    winner={"generations":tuned.generations,
            "local_candidates":tuned.local_candidates,
            "mean_error_pct":second[0]["mean_error_pct"]}
    (root/f"{group}_frozen_baseline.json").write_text(
       json.dumps({"phase":"trained exclusively on training matrices",
         "class":group,"starting_settings":asdict(base),
         "selected_settings":asdict(tuned),"winner":winner,
         "train_sizes":TRAIN[group],"train_repeats":len(SEEDS["train"]),
         "method":"sequential fixed-context one-parameter-at-a-time (G, then K)",
         "training_rankings":rankings},indent=2,ensure_ascii=False),encoding="utf-8")
    return winner

def refine(group,root,workers=2,rounds=1,parameters=None):
    """Coordinate search on TRAIN data only. Each comparison changes one
    parameter from CURRENT context; repeat passes if interactions matter.
    Island count/migration are tuned separately after the single GA.
    """
    root=Path(root)
    meta=json.loads((root/f"{group}_frozen_baseline.json").read_text())
    current=Config(**meta["selected_settings"])
    order=list(parameters) if parameters is not None else [
      "selection","rank_power","adaptive","adaptive_horizon",
      "adaptive_long_prob","adaptive_long_pct",
      "mutation_prob","inversion_pct","elitism","population",
      "local_every","local_candidates","local_passes","local_top",
      "cycle3","cycle3_passes","generations"]
    history=[]
    valid=set(LEVELS)-{"islands","migration_interval","migration_pct"}
    if any(p not in valid for p in order):raise ValueError("Unsupported tuning factor")
    def score(parameter,candidate,baseline,phase):
        errors=[];times=[]
        for n in TRAIN[group]:
            for rep,token in enumerate(SEEDS["train"]):
                seed=_seed(n,token,"train")
                a,opt=problem(n,seed)
                result,error=evaluate(a,opt,candidate,seed+9,workers)
                errors.append(error);times.append(result["seconds"])
                history.append({
                    "round":phase,"parameter":parameter,
                    "value":str(getattr(candidate,parameter)),
                    "n":n,"rep":rep,"error_pct":error,
                    "seconds":result["seconds"],"matrix_seed":seed,
                    "ga_seed":seed+9,
                    "context_json":json.dumps(asdict(baseline),sort_keys=True),
                    "settings_json":json.dumps(asdict(candidate),sort_keys=True)
                })
        return mean(errors),mean(times)

    for round_index in range(rounds):
        for parameter in order:
            if parameter=="rank_power" and current.selection!="rank":
                continue
            if parameter in ("adaptive_horizon","adaptive_long_prob",
                              "adaptive_long_pct") and not current.adaptive:
                continue
            if parameter=="cycle3_passes" and not current.cycle3:
                continue
            candidates=LEVELS[parameter]
            context=current
            ranked=[]
            for value in candidates:
                candidate=replace(context,**{parameter:value})
                loss,seconds=score(parameter,candidate,context,round_index)
                ranked.append((loss,seconds,value))
            ranked.sort(key=lambda x:(x[0],x[1]))
            current=replace(current,**{parameter:ranked[0][2]})
            save(root/f"{group}_refine_raw.csv",history)
            print("REFINE",group,round_index,parameter,"best",ranked[0],flush=True)

    # Separate parallel configuration; do not let island tuning change the
    # selected SINGLE-GA config or confound the class OFAT baseline.
    parallel=replace(current,islands=2)
    for parameter in ("islands","migration_interval","migration_pct"):
        context=parallel
        values=(2,4,8) if parameter=="islands" else LEVELS[parameter]
        ranked=[]
        for value in values:
            candidate=replace(context,**{parameter:value})
            loss,seconds=score(parameter,candidate,context,"islands")
            ranked.append((loss,seconds,value))
        ranked.sort(key=lambda x:(x[0],x[1]))
        parallel=replace(parallel,**{parameter:ranked[0][2]})
        save(root/f"{group}_refine_raw.csv",history)
        print("PARALLEL",group,parameter,"best",ranked[0],flush=True)

    meta["selected_settings"]=asdict(current)
    meta["island_settings"]=asdict(parallel)
    meta["refine_rounds"]=rounds
    meta["refine_order"]=order
    meta["refinement_method"]="strict one-coordinate training search; no test or survey matrices"
    (root/f"{group}_frozen_baseline.json").write_text(
        json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    return {"single":asdict(current),"islands":asdict(parallel)}


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
                context=(Config(**meta.get("island_settings",asdict(replace(base,islands=4))))
                         if parameter in ("migration_interval","migration_pct")
                         else replace(base,**ACTIVE_CONTEXT.get(parameter,{})))
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
    island_cfg=Config(**m["island_settings"]) if "island_settings" in m else None
    if island_cfg is not None:
        for key in ("population","generations","local_every","local_top",
                    "local_passes","local_candidates","cycle3","cycle3_passes"):
            if getattr(selected,key)!=getattr(island_cfg,key):
                raise AssertionError(f"Unfair single/island comparison: {key}")
    rows=[]
    for n in HOLDOUT[group]:
        for repeat,token in enumerate(SEEDS["holdout"]):
            mat_seed=_seed(n,token,"holdout")
            a,opt=problem(n,mat_seed)
            variants=[("before",base),("after",selected)]
            if island_cfg is not None:variants.append(("after_islands",island_cfg))
            pair={}
            for name,cfg in variants:
                ans,err=evaluate(a,opt,cfg,mat_seed+9,workers)
                if name in ("after","after_islands"):
                    pair[name]=ans["local_calls"]
                rows.append({"class_name":group,"n":n,"repeat":repeat,
                    "variant":name,"cost":ans["cost"],"exact":opt,
                    "error_pct":err,"seconds":ans["seconds"],
                    "population":ans["total_population"],
                    "generations":ans["generations"],
                    "local_calls":ans["local_calls"],
                    "num_islands":ans["num_islands"],
                    "matrix_seed":mat_seed,"ga_seed":mat_seed+9,
                    "settings_json":json.dumps(asdict(cfg),sort_keys=True)})
            if "after_islands" in pair:
                assert pair["after"]==pair["after_islands"],(n,repeat,pair)
        save(root/f"{group}_holdout.csv",rows)
    aggregate=[]
    for name in ("before","after","after_islands"):
        sample=[r for r in rows if r["variant"]==name]
        if sample:
            aggregate.append(dict(class_name=group,variant=name,
                observed_runs=len(sample),
                mean_error_pct=mean(r["error_pct"] for r in sample),
                mean_seconds=mean(r["seconds"] for r in sample),
                mean_local_calls=mean(r["local_calls"] for r in sample),
                total_population=sample[0]["population"],
                generations=sample[0]["generations"]))
    save(root/f"{group}_holdout_comparison.csv",aggregate)
    return rows

# Values found on larger/smaller training groups are candidates, never
# automatically transplanted without TARGET-class training evidence.
TRANSFER_FIELDS=("selection","rank_power","adaptive","adaptive_horizon",
    "adaptive_long_prob","adaptive_long_pct","mutation_prob","inversion_pct",
    "elitism","local_every","local_candidates","local_passes","local_top",
    "cycle3","cycle3_passes")

def transfer(group,root,workers=2):
    """One-coordinate cross-size transfer tested only on TARGET TRAIN data.

    Equal population and generations are held in the target context.
    Accept a donor's value only if its paired target TRAIN mean error drops.
    No survey/holdout observations enter selection; only one config field
    changes per test. Run this after refining all three classes, before OFAT.
    """
    root=Path(root)
    metas={}
    for name in TRAIN:
        file=root/f"{name}_frozen_baseline.json"
        if not file.is_file():
            raise FileNotFoundError(f"Complete refinement for {name} before transfer")
        metas[name]=json.loads(file.read_text(encoding="utf-8"))
    meta=metas[group]
    current=Config(**meta["selected_settings"])
    sources={name:Config(**m["selected_settings"]) for name,m in metas.items()}
    cache={}
    for n in TRAIN[group]:
        for token in SEEDS["train"]:
            seed=_seed(n,token,"train")
            cache[(n,token)]=(seed,)+problem(n,seed)
    def average(c):
        return mean(evaluate(cache[(n,t)][1],cache[(n,t)][2],c,
                   cache[(n,t)][0]+9,workers)[1]
                   for n in TRAIN[group] for t in SEEDS["train"])
    trials=[]
    for field in TRANSFER_FIELDS:
        if field=="rank_power" and current.selection!="rank":continue
        if field.startswith("adaptive_") and not current.adaptive:continue
        if field=="cycle3_passes" and not current.cycle3:continue
        context=current
        candidates={getattr(context,field):"target"}
        for name,source in sources.items():
            if name!=group:candidates.setdefault(getattr(source,field),name)
        scores=[]
        for value,origin in candidates.items():
            c=replace(context,**{field:value})
            changed={k for k in asdict(context) if getattr(context,k)!=getattr(c,k)}
            assert changed<={field}
            score=average(c)
            scores.append((score,value,origin))
            trials.append(dict(target_class=group,parameter=field,
                value=str(value),source_class=origin,mean_train_error_pct=score,
                context_json=json.dumps(asdict(context),sort_keys=True),
                settings_json=json.dumps(asdict(c),sort_keys=True)))
        winner=min(scores,key=lambda z:z[0])
        # Require strict improvement, rather than moving on ties.
        baseline=next(score for score,value,_ in scores
                      if value==getattr(context,field))
        if winner[0]<baseline-1.e-9:
            current=replace(context,**{field:winner[1]})
    save(root/f"{group}_transfer.csv",trials)
    meta["selected_settings"]=asdict(current)
    if "island_settings" in meta:
        island=Config(**meta["island_settings"])
        island=replace(island,**{name:getattr(current,name)
                               for name in TRANSFER_FIELDS})
        meta["island_settings"]=asdict(island)
    meta["cross_class_transfer"]={
        "method":"one-factor on target TRAIN with fixed N,G; lower mean accepted",
        "candidate_sources":list(TRAIN),"trials":len(trials),
        "no_holdout_selection":True}
    (root/f"{group}_frozen_baseline.json").write_text(
        json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
    return asdict(current)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--class-name",choices=tuple(TRAIN)+("all",),default="medium")
    p.add_argument("--output",default="results/staged_tuning_v2")
    p.add_argument("--source",default="results/dense_ofat_20261027/best_observed_config.json")
    p.add_argument("--stage",choices=("train","refine","transfer","ofat","holdout","all"),default="all")
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--rounds",type=int,default=1)
    p.add_argument("--parameters",nargs="*",default=None)
    args=p.parse_args()
    groups=tuple(TRAIN) if args.class_name=="all" else (args.class_name,)
    # Complete the TRAIN-only phases for ALL groups before any transfer.
    if args.stage in ("train","all"):
        for group in groups:
            print("SELECTED",group,train(group,args.output,args.source,args.workers),
                  flush=True)
    if args.stage in ("refine","all"):
        for group in groups:
            print("REFINED",group,refine(group,args.output,args.workers,
                                    args.rounds,args.parameters),flush=True)
    if args.stage in ("transfer","all") and args.class_name=="all":
        for group in groups:
            print("TRANSFER",group,transfer(group,args.output,args.workers),flush=True)
    elif args.stage=="transfer":
        print("TRANSFER",groups[0],transfer(groups[0],args.output,args.workers),
              flush=True)
    if args.stage in ("ofat","all"):
        for group in groups:ofat(group,args.output,args.workers)
    if args.stage in ("holdout","all"):
        for group in groups:validate(group,args.output,args.workers)

if __name__=="__main__":
    main()
