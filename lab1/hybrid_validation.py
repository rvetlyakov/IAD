"""Independent 3-matrix validation for each n=100,500,2000 at matched N and G.

Do not select winner on this holdout. Compare original operators against
rank RSS, and optional local search, including true multiprocess islands.
"""
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
from hybrid_ga import Config, run
from scipy.optimize import linear_sum_assignment


def validate_repeats(path="results/hybrid_ga_study/validation.csv"):
    base=Config(population=120,generations=250,mutation_prob=1,
                inversion_pct=2,elitism=.2,selection="raw",local_every=0)
    variants={
        "classic RSS+inversion":base,
        "rank RSS":replace(base,selection="rank",rank_power=2),
        "hybrid":replace(base,selection="rank",rank_power=2,
                         local_every=25,local_passes=2),
        "hybrid 4 islands":replace(base,selection="rank",rank_power=2,
                                   local_every=25,local_passes=2,
                                   islands=4,migration_interval=25)
    }
    rows=[]
    for n in (100,500,2000):
        for sample in range(3):
            matrix_seed=20261027+n*2381+sample*991
            a=np.random.default_rng(matrix_seed).integers(
                1,101,(n,n),dtype=np.int32)
            r,c=linear_sum_assignment(a)
            exact=int(a[r,c].sum(dtype=np.int64))
            for name,setting in variants.items():
                value=run(a,setting,seed=matrix_seed+1,workers=2)
                rows.append({
                    "size":n,"matrix_seed":matrix_seed,"variant":name,
                    "exact":exact,"cost":value["cost"],
                    "error_percent":100*(value["cost"]-exact)/exact,
                    "seconds":value["seconds"],
                    "generations":value["generations"],
                    "total_population":value["total_population"],
                    "islands":value["num_islands"]
                })
            print("Independent validation",n,sample+1,"complete",flush=True)
    dest=Path(path)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open("w",newline="",encoding="utf8") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows

if __name__=="__main__":
    validate_repeats()
