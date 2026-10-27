"""Fair ordinary-versus-parallel GA comparison across all measured sizes.

Only num_islands is changed. Total N, generations, all evolutionary
operators, matrix, initial chromosome ensemble and seed are identical.
Process overhead is included in real wall-clock timing.
"""
import argparse,json
from dataclasses import asdict,replace
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from hybrid_ga import run
from ga_dense_study import BASE, DIMENSIONS, SMOKE_DIMS, write_csv, plot

ISLAND_COUNTS=(1,2,4,8)

def compare(output="results/ga_fair_compare",mode="smoke",
            repeats=2,seed=20261106,workers=2):
    if mode not in ("full","smoke"):raise ValueError(mode)
    sizes=DIMENSIONS if mode=="full" else SMOKE_DIMS
    dest=Path(output);dest.mkdir(parents=True,exist_ok=True)
    all_rows=[]
    for cls,dimlist in sizes.items():
        for n in dimlist:
            for rep in range(repeats):
                matrix_seed=seed+10007*n+1000003*rep
                a=np.random.default_rng(matrix_seed).integers(1,101,(n,n),dtype=np.int32)
                ii,jj=linear_sum_assignment(a)
                exact=int(a[ii,jj].sum(dtype=np.int64))
                for islands in ISLAND_COUNTS:
                    cfg=replace(BASE,islands=islands)
                    result=run(a,cfg,seed=matrix_seed+37,workers=workers)
                    all_rows.append(dict(**{"class":cls},size=n,repeat=rep,
                         matrix_seed=matrix_seed,ga_seed=matrix_seed+37,
                         variant=f"{islands} остров(а)",islands=islands,
                         population=cfg.population,generations=cfg.generations,
                         exact=exact,cost=result["cost"],
                         gap_pct=100*(result["cost"]-exact)/exact,
                         elapsed_seconds=result["seconds"],
                         settings_json=json.dumps(asdict(cfg),sort_keys=True)))
            print(f"FAIR {cls} {n}: {repeats} paired matrices",flush=True)
            write_csv(dest/"raw.csv",all_rows)
    summary=[]
    for cls,values in sizes.items():
        for n in values:
            for islands in ISLAND_COUNTS:
                rows=[r for r in all_rows if r["class"]==cls
                       and r["size"]==n and r["islands"]==islands]
                summary.append(dict(factor="islands",value=str(islands),
                    **{"class":cls},size=n,runs=len(rows),
                    mean_gap=float(np.mean([r["gap_pct"] for r in rows])),
                    std_gap=float(np.std([r["gap_pct"] for r in rows],ddof=1)) if len(rows)>1 else 0.,
                    mean_seconds=float(np.mean([r["elapsed_seconds"] for r in rows]))))
    write_csv(dest/"summary.csv",summary)
    plot(summary,dest/"plots",mode=mode)
    (dest/"metadata.json").write_text(json.dumps({
        "mode":mode,"seed":seed,"repeats":repeats,"dimensions":sizes,
        "population":BASE.population,"generations":BASE.generations,
        "variants":list(ISLAND_COUNTS),
        "protocol":"exactly one setting changes: number of islands; same total population, generations, initial seeded permutations",
        "timing":"wall clock including worker and migration overhead",
        "chart":"PCHIP presentation interpolation only; scatter points are empirical"
    },ensure_ascii=False,indent=2),encoding="utf8")
    return all_rows

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results/ga_fair_compare")
    p.add_argument("--mode",choices=("full","smoke"),default="smoke")
    p.add_argument("--seed",type=int,default=20261106)
    p.add_argument("--repeats",type=int,default=2)
    p.add_argument("--workers",type=int,default=2)
    a=p.parse_args()
    compare(a.output,a.mode,a.repeats,a.seed,a.workers)
