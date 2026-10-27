"""Sequential coordinate search with independent holdout validation.

For each matrix-size class, each step evaluates ONE parameter at a time,
reusing the just-selected best configuration in the next step. All levels
at a step are tested with the same exact matrices, GA seeds and budget
of evaluation per level. SciPy is external reference, not a GA operator.
"""
from __future__ import annotations
import argparse, csv, json
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, pstdev
import numpy as np
from scipy.optimize import linear_sum_assignment
from hybrid_ga import Config,run
from ga_dense_study import BASE,FACTORS,write_csv

TRAIN={"small":[10,30,50],"medium":[100,300,500],
       "large":[1000,1500,2000]}
HOLDOUT={"small":[20,40],"medium":[200,400],
         "large":[1250,1750]}
SMOKE_TRAIN={"small":[10],"medium":[100],"large":[200]}
SMOKE_HOLDOUT={"small":[20],"medium":[140],"large":[300]}

# Evaluate selection before its conditional rank_power; evaluate the island
# count before migration, and the local search interval before candidate width.
ORDER=["selection","rank_power","local_every","local_candidates",
       "inversion_pct","adaptive","mutation_prob","elitism",
       "population","generations","islands","migration_interval","migration_pct"]

def matrices(kind,size_class,sizes,repeats,seed):
    """Paired cases. Holdout uses different sizes AND different matrix seeds."""
    index={"small":0,"medium":1,"large":2}[size_class]
    for n in sizes:
        for rep in range(repeats):
            matrix_seed=seed+index*10_000_000+n*8191+rep*17117+(
                3_000_000 if kind=="holdout" else 0)
            a=np.random.default_rng(matrix_seed).integers(1,101,(n,n),dtype=np.int32)
            i,j=linear_sum_assignment(a)
            yield n,rep,matrix_seed,matrix_seed+71,a,int(a[i,j].sum(dtype=np.int64))

def evaluate_on_cases(config,cases,phase,param,value,size_class,workers=2):
    rows=[]
    for n,rep,matrix_seed,ga_seed,a,exact in cases:
        result=run(a,config,seed=ga_seed,workers=workers)
        gap=100*(result["cost"]-exact)/exact
        rows.append(dict(phase=phase,**{"class":size_class},
             factor=param,value=value,size=n,repeat=rep,matrix_seed=matrix_seed,
             ga_seed=ga_seed,exact_cost=exact,ga_cost=result["cost"],
             gap_pct=float(gap),elapsed_seconds=float(result["seconds"]),
             population=config.population,generations=config.generations,
             islands=config.islands,settings_json=json.dumps(asdict(config),sort_keys=True)))
    return rows

def select_best(options):
    """Deterministically minimize the paired mean relative error.

    Time is reported but deliberately excluded from noisy selection.
    """
    evaluated=[]
    for label,rows,config in options:
        evaluated.append((label,mean(r["gap_pct"] for r in rows),
                          mean(r["elapsed_seconds"] for r in rows),config))
    # Runtime is noisy (especially on shared CI runners) and MUST NOT be
    # used as a primary/near-tie selector. Minimize measured error only;
    # stable Python min keeps the earliest grid level for exact ties.
    label,gap,sec,config=min(evaluated,key=lambda x:x[1])
    return (label,gap,sec,config,evaluated)

def tune(output="results/ga_tuned_classes",mode="smoke",seed=20261105,
         training_repeats=2,holdout_repeats=3,workers=2,factors=None):
    if mode not in ("smoke","full"):raise ValueError(mode)
    if training_repeats<1 or holdout_repeats<1:raise ValueError("positive repeats required")
    training=SMOKE_TRAIN if mode=="smoke" else TRAIN
    holdout=SMOKE_HOLDOUT if mode=="smoke" else HOLDOUT
    factors=list(ORDER) if not factors else list(factors)
    if any(x not in ORDER for x in factors):raise ValueError("unknown factor")
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    all_rows=[];step_rows=[];best_by_class={};validation=[]
    for cls,sizes in training.items():
        config=BASE
        train_cases=list(matrices("train",cls,sizes,training_repeats,seed))
        for factor in factors:
            # Conditional factors must only be tuned when active. Otherwise
            # zero-effect parameters cannot be scientifically identified.
            if (factor=="rank_power" and config.selection!="rank") or (
                factor=="local_candidates" and config.local_every==0) or (
                factor in ("migration_interval","migration_pct") and config.islands==1
            ):
                step_rows.append(dict(**{"class":cls},factor=factor,
                       selected="not applicable",mean_gap_pct="",mean_seconds="",
                       reason="operator disabled in selected configuration"))
                continue
            options=[]
            for level,patch in FACTORS[factor]:
                cfg=replace(config,**patch)
                rows=evaluate_on_cases(cfg,train_cases,"train",factor,level,cls,workers)
                options.append((level,rows,cfg))
                all_rows.extend(rows)
            chosen,gap,sec,config,score=select_best(options)
            step_rows.append(dict(**{"class":cls},factor=factor,
                        selected=chosen,mean_gap_pct=round(gap,4),
                        mean_seconds=round(sec,5),
                        reason="minimum paired error; identical errors use earliest grid value"))
            print(f"TUNED {cls} {factor}={chosen} gap={gap:.2f}%",flush=True)
            write_csv(out/"train_raw.csv",all_rows)
            write_csv(out/"selection_steps.csv",step_rows)
        best_by_class[cls]=asdict(config)
        for n,rep,matrix_seed,ga_seed,a,exact in matrices(
                 "holdout",cls,holdout[cls],holdout_repeats,seed):
            # Compare matched configuration (same N,G and all other controls),
            # only the number of independently evolving islands changes.
            normal=replace(config,islands=1)
            parallel=replace(config,islands=max(2,config.islands))
            for label,cfg in [("single",normal),("parallel",parallel)]:
                result=run(a,cfg,seed=ga_seed,workers=workers)
                validation.append(dict(**{"class":cls},size=n,repeat=rep,
                   variant=label,matrix_seed=matrix_seed,ga_seed=ga_seed,
                   exact_cost=exact,ga_cost=int(result["cost"]),
                   gap_pct=100*(result["cost"]-exact)/exact,
                   elapsed_seconds=float(result["seconds"]),
                   population=cfg.population,generations=cfg.generations,
                   islands=cfg.islands,
                   settings_json=json.dumps(asdict(cfg),sort_keys=True)))
        write_csv(out/"validation_raw.csv",validation)
    (out/"best_parameters.json").write_text(json.dumps({
       "method":"one-pass sequential coordinate descent, best observed on paired training matrices",
       "objective":"minimum paired mean relative error; deterministic first-listed tie breaker",
       "classes":best_by_class,"train_dimensions":training,
       "holdout_dimensions":holdout,"training_repeats":training_repeats,
       "holdout_repeats":holdout_repeats,"training_seed":seed,
       "holdout_seed_offset":3_000_000,
       "scientific_caveat":"local optimum within tested discrete levels, not global hyperparameter optimum"
    },indent=2,ensure_ascii=False),encoding="utf8")
    summary=[]
    for cls in best_by_class:
        for variant in ("single","parallel"):
            sub=[r for r in validation if r["class"]==cls and r["variant"]==variant]
            summary.append(dict(**{"class":cls},variant=variant,
                 runs=len(sub),mean_gap_pct=mean(r["gap_pct"] for r in sub),
                 std_gap_pct=pstdev(r["gap_pct"] for r in sub),
                 mean_seconds=mean(r["elapsed_seconds"] for r in sub),
                 population=sub[0]["population"],generations=sub[0]["generations"],
                 islands=sub[0]["islands"]))
    write_csv(out/"validation_summary.csv",summary)
    return {"best":best_by_class,"validation":summary,"train_rows":len(all_rows)}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results/ga_tuned_classes")
    p.add_argument("--mode",choices=("smoke","full"),default="smoke")
    p.add_argument("--seed",type=int,default=20261105)
    p.add_argument("--train-repeats",type=int,default=2)
    p.add_argument("--holdout-repeats",type=int,default=3)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--factors",nargs="*")
    a=p.parse_args()
    print(json.dumps(tune(a.output,a.mode,a.seed,
          a.train_repeats,a.holdout_repeats,a.workers,a.factors),
          ensure_ascii=False,indent=2))
