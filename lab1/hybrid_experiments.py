"""Measured OFAT sweeps; all three size classes on each plot.

Plots contain only axes, measured curves, legend and three background bands.
CSV includes every run's full parameter snapshot and paired matrix seeds.
Usage: python lab1/hybrid_experiments.py --output results/hybrid_ga_study --diagonal
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from hybrid_ga import Config, run
from scipy.optimize import linear_sum_assignment

SIZES={"small":[5,10,30,50],"medium":[100,200,300,400,500],
       "large":[1000,1250,1500,1750,2000]}
SMOKE={"small":[5,10],"medium":[100,200],"large":[300,400]}
BASE=Config(population=120,generations=250,mutation_prob=1.0,inversion_pct=2,
            elitism=.2,selection="rank",rank_power=2,local_every=25,
            local_top=1,local_candidates=8,local_passes=2,islands=1)
SWEEPS={
 "generations":[("40",{"generations":40}),("100",{"generations":100}),("250",{"generations":250}),("500",{"generations":500}),("1000",{"generations":1000})],
 "selection":[("1/C",{"selection":"raw"}),("rank 1",{"selection":"rank","rank_power":1}),("rank 2",{"selection":"rank","rank_power":2}),("rank 4",{"selection":"rank","rank_power":4})],
 "inversion":[("2%",{"inversion_pct":2,"adaptive":False}),("5%",{"inversion_pct":5,"adaptive":False}),("20%",{"inversion_pct":20,"adaptive":False}),("adaptive",{"inversion_pct":20,"adaptive":True})],
 "local_search":[("off",{"local_every":0}),("every 5",{"local_every":5}),("every 10",{"local_every":10}),("every 25",{"local_every":25})],
 "islands":[("1",{"islands":1}),("2",{"islands":2}),("4",{"islands":4}),("8",{"islands":8})],
 "population":[("60",{"population":60}),("120",{"population":120}),("240",{"population":240})],
 "elitism":[("0%",{"elitism":0}),("10%",{"elitism":.1}),("20%",{"elitism":.2}),("40%",{"elitism":.4})],
 "mutation_probability":[("0.3",{"mutation_prob":.3}),("0.7",{"mutation_prob":.7}),("1.0",{"mutation_prob":1.0})],
 "migration_interval":[("5",{"islands":4,"migration_interval":5}),("25",{"islands":4,"migration_interval":25}),("50",{"islands":4,"migration_interval":50})],
 "migration_fraction":[("5%",{"islands":4,"migration_pct":.05}),("10%",{"islands":4,"migration_pct":.10}),("20%",{"islands":4,"migration_pct":.20})],
 "local_candidates":[("4",{"local_candidates":4}),("8",{"local_candidates":8}),("16",{"local_candidates":16}),("32",{"local_candidates":32})]
}
COLORS=["#1d4f91","#258b7b","#e08449","#88478e","#bd6259"]
DISPLAY={"off":"без поиска","every 5":"каждые 5","every 10":"каждые 10","every 25":"каждые 25","rank 1":"ранг 1","rank 2":"ранг 2","rank 4":"ранг 4","adaptive":"адаптивная"}

def save_csv(path,rows):
    if not rows:return
    with open(path,"w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def summarize(rows):
    grouped=defaultdict(list)
    for row in rows:
        grouped[(row["parameter"],row["level"],row["class"],row["size"])].append(row)
    return [{"parameter":p,"level":v,"class":g,"size":n,
             "runs":len(a),"mean_error_percent":float(np.mean([q["error_percent"] for q in a])),
             "mean_seconds":float(np.mean([q["elapsed_seconds"] for q in a]))}
            for (p,v,g,n),a in grouped.items()]

def draw_all(summary,out,sizes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts=[f for f in font_manager.findSystemFonts() if "Montserrat" in f]
    if fonts:
        font_manager.fontManager.addfont(fonts[0])
        family="Montserrat"
    else:family="DejaVu Sans"
    plt.rcParams.update({"font.family":family,"font.size":11,
                         "axes.unicode_minus":False,"svg.fonttype":"none"})
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    bands={"small":(0,5,"#e9f5f2"),"medium":(6,11,"#ecf2f9"),
           "large":(12,17,"#fcf0ed")}
    pos={}
    for group,values in sizes.items():
        left,right,_=bands[group]
        for i,n in enumerate(values):
            pos[group,n]=left+(right-left)*i/max(1,len(values)-1)
    for parameter,levels in SWEEPS.items():
        for metric,suffix in (("mean_error_percent","error"),("mean_seconds","runtime")):
            fig,ax=plt.subplots(figsize=(13,6.8),dpi=165,facecolor="white")
            fig.subplots_adjust(left=.10,right=.985,top=.955,bottom=.285)
            for group in sizes:
                l,r,color=bands[group]
                ax.axvspan(l-.35,r+.35,color=color,alpha=.87,zorder=0)
            ax.grid(axis="y",color="#d9dfe7",lw=.8)
            ax.set_axisbelow(True)
            for index,(label,updates) in enumerate(levels):
                subset=sorted((q for q in summary
                               if q["parameter"]==parameter and q["level"]==label),
                              key=lambda q:pos[q["class"],q["size"]])
                xx=[pos[q["class"],q["size"]] for q in subset]
                yy=[q[metric] for q in subset]
                if not yy:continue
                ax.plot(xx,yy,lw=2.65,color=COLORS[index],
                        label=DISPLAY.get(label,label),solid_capstyle="round")
                ax.scatter(xx,yy,s=23,color=COLORS[index],alpha=.35,zorder=5)
            ax.set_xlim(-.45,17.45)
            ax.set_xticks([pos[g,n] for g in sizes for n in sizes[g]])
            ax.set_xticklabels([str(n) for g in sizes for n in sizes[g]],fontsize=9)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("Размерность матрицы n × n",labelpad=64,fontsize=12)
            ax.set_ylabel("Средняя относительная ошибка, %" if suffix=="error"
                          else "Время выполнения, с",fontsize=12)
            for spine in ax.spines.values():spine.set_visible(False)
            ax.tick_params(colors="#495a6b",length=0,pad=7)
            ax.legend(title="Значение параметра",ncol=min(3,len(levels)),
                      facecolor="white",edgecolor="#d6d6d6",fontsize=10,loc="upper left")
            for group in sizes:
                l,r,_=bands[group]
                ax.text((l+r)/2,-.095,
                        {"small":"малые","medium":"средние","large":"большие"}[group],
                        transform=ax.get_xaxis_transform(),ha="center",va="top",
                        fontsize=11,fontweight="bold",
                        color={"small":"#32917e","medium":"#3c70a2",
                               "large":"#bd735e"}[group])
            fig.savefig(str(out/f"{parameter}_{suffix}")+".png",dpi=165,facecolor="white")
            fig.savefig(str(out/f"{parameter}_{suffix}")+".svg",facecolor="white")
            plt.close(fig)

def study(output,seed=20261101,repeats=1,workers=2,smoke=False):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    sizes=SMOKE if smoke else SIZES
    raw=[]
    for group,levels in sizes.items():
        for n in levels:
            for repetition in range(repeats):
                matrix_seed=seed+n*1093+repetition*65537
                matrix=np.random.default_rng(matrix_seed).integers(
                    1,101,(n,n),dtype=np.int32)
                r,c=linear_sum_assignment(matrix)
                exact=int(matrix[r,c].sum(dtype=np.int64))
                ga_seed=matrix_seed+11
                for parameter,variants in SWEEPS.items():
                    for label,changes in variants:
                        cfg=replace(BASE,**changes)
                        result=run(matrix,cfg,seed=ga_seed,workers=workers)
                        raw.append({
                            "parameter":parameter,"level":label,"class":group,
                            "size":n,"repeat":repetition,"matrix_seed":matrix_seed,
                            "ga_seed":ga_seed,"exact":exact,"cost":result["cost"],
                            "error_percent":100*(result["cost"]-exact)/exact,
                            "elapsed_seconds":result["seconds"],
                            "total_population":result["total_population"],
                            "num_islands":result["num_islands"],
                            "generations":result["generations"],
                            "settings_json":json.dumps(asdict(cfg),ensure_ascii=False,
                                                       sort_keys=True)
                        })
            print(f"{group} {n}: {len(raw)} measurements recorded",flush=True)
            save_csv(out/"raw.csv",raw)
            save_csv(out/"summary.csv",summarize(raw))
            (out/"metadata.json").write_text(json.dumps({
                "seed":seed,"sizes":sizes,"repeats":repeats,
                "factors":{k:[v for v,_ in x] for k,x in SWEEPS.items()},
                "method":"paired test matrices/seeds, one factor at a time",
                "note":"Migration sweeps fix islands=4; other sweeps use one island unless varying islands."
            },ensure_ascii=False,indent=2))
    draw_all(summarize(raw),out/"plots",sizes)
    return raw

def diagonal_trials(output,seed=20261101):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for n in (5,100,500,2000):
        a=np.full((n,n),100,dtype=np.int32)
        np.fill_diagonal(a,1)
        for label,c in (("baseline",replace(BASE,generations=40,
                                            local_every=0,selection="raw")),
                        ("hybrid",replace(BASE,generations=250,
                                          local_every=25,selection="rank"))):
            for rep in range(3):
                result=run(a,c,seed+rep,workers=2)
                rows.append({"size":n,"variant":label,"seed":seed+rep,
                             "optimal_cost":n,"ga_cost":result["cost"],
                             "error_percent":100*(result["cost"]-n)/n,
                             "runtime":result["seconds"]})
        print("diagonal",n,flush=True)
    save_csv(out/"diagonal.csv",rows)
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results/hybrid_ga_study")
    p.add_argument("--repeats",type=int,default=1)
    p.add_argument("--seed",type=int,default=20261101)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--diagonal",action="store_true")
    args=p.parse_args()
    rows=study(args.output,args.seed,args.repeats,args.workers,args.smoke)
    if args.diagonal:diagonal_trials(args.output,args.seed)
    print("COMPLETE",len(rows),flush=True)

if __name__=="__main__":
    main()
