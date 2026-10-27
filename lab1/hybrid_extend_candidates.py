"""Append a measured 11th OFAT sweep to the archived 10-factor results.

Only local_candidates is changed; all other settings are BASE.
Reloads previous result artifact; adds real measurements without re-running them.
"""
import argparse,csv,json
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from hybrid_ga import run
from hybrid_experiments import BASE,SWEEPS,SIZES,save_csv,summarize,draw_all

def extend(root="results/hybrid_ga_study",seed=20261101):
    root=Path(root)
    with (root/"raw.csv").open(encoding="utf-8",newline="") as file:
        old=list(csv.DictReader(file))
    if any(x["parameter"]=="local_candidates" for x in old):
        raise ValueError("local candidate sweep already exists")
    added=[]
    for group,sizes in SIZES.items():
        for n in sizes:
            matrix_seed=seed+n*1093
            matrix=np.random.default_rng(matrix_seed).integers(
                1,101,(n,n),dtype=np.int32)
            r,c=linear_sum_assignment(matrix)
            exact=int(matrix[r,c].sum(dtype=np.int64))
            for label,updates in SWEEPS["local_candidates"]:
                settings=replace(BASE,**updates)
                result=run(matrix,settings,seed=matrix_seed+11,workers=2)
                added.append({
                    "parameter":"local_candidates","level":label,
                    "class":group,"size":n,"repeat":0,
                    "matrix_seed":matrix_seed,"ga_seed":matrix_seed+11,
                    "exact":exact,"cost":result["cost"],
                    "error_percent":100*(result["cost"]-exact)/exact,
                    "elapsed_seconds":result["seconds"],
                    "total_population":result["total_population"],
                    "num_islands":result["num_islands"],
                    "generations":result["generations"],
                    "settings_json":json.dumps(asdict(settings),
                                               ensure_ascii=False,sort_keys=True)
                })
            print("Candidate width",group,n,"done",flush=True)
    allrows=old+added
    save_csv(root/"raw.csv",allrows)
    for row in allrows:
        for field in ("size","repeat","exact","cost","total_population",
                      "num_islands","generations"):
            row[field]=int(row[field])
        for field in ("error_percent","elapsed_seconds"):
            row[field]=float(row[field])
    summarized=summarize(allrows)
    save_csv(root/"summary.csv",summarized)
    metadata=json.loads((root/"metadata.json").read_text(encoding="utf8"))
    metadata["factors"]={key:[label for label,_ in variants]
                         for key,variants in SWEEPS.items()}
    (root/"metadata.json").write_text(json.dumps(
        metadata,ensure_ascii=False,indent=2),encoding="utf8")
    draw_all(summarized,root/"plots",SIZES)
    return allrows

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",default="results/hybrid_ga_study")
    parser.add_argument("--seed",type=int,default=20261101)
    args=parser.parse_args()
    out=extend(root=args.root,seed=args.seed)
    print("TOTAL",len(out),"measured observations",flush=True)
