"""Dense measured OFAT sensitivity curves for the assignment genetic algorithm.

Every plotted anchor is a real run. PCHIP is optional presentation-only
interpolation, never a fabricated measurement. Matrix dimensions 550--950
are explicitly marked as transitional, not reclassified as medium/large.
"""
from __future__ import annotations
import argparse, csv, json, os
from dataclasses import asdict, replace
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from hybrid_ga import Config, run

DIMENSIONS = {
    "small": list(range(10,51,10)),
    "medium": list(range(100,501,40)),
    "transition": list(range(550,951,50)),
    "large": list(range(1000,2001,100)),
}
SMOKE_DIMS = {"small":[10,20],"medium":[100,140],
              "transition":[550,600],"large":[1000,1100]}
BASE=Config(population=120,generations=250,mutation_prob=1.0,
            inversion_pct=2.,elitism=.2,selection="rank",rank_power=2.,
            local_every=25,local_candidates=8,local_top=1,local_passes=2,
            islands=1,migration_interval=25,migration_pct=.1)

# NOTE: Every level of each factor changes exactly ONE Config field.
# selection=rank is fixed for rank_power; islands=4 is fixed for migration.
FACTORS={
 "population": [(str(x), {"population":x}) for x in (60,120,240)],
 "generations": [(str(x), {"generations":x}) for x in (40,100,250,500,1000)],
 "mutation_prob": [(str(x), {"mutation_prob":x}) for x in (.3,.7,1.)],
 "inversion_pct": [(f"{x}%", {"inversion_pct":x}) for x in (1.,2.,5.,20.)],
 "elitism": [(f"{x:g}%", {"elitism":x/100}) for x in (0,10,20,40)],
 "selection": [("1/C",{"selection":"raw"}),("ранговый RSS",{"selection":"rank"})],
 "rank_power": [(str(x),{"rank_power":x}) for x in (1.,2.,4.)],
 "adaptive": [("выключена",{"adaptive":False}),("включена",{"adaptive":True})],
 "local_every": [("без поиска",{"local_every":0})]+[(f"каждые {x}",{"local_every":x}) for x in (5,10,25)],
 "local_candidates": [(str(x),{"local_candidates":x}) for x in (4,8,16,32)],
 "islands": [(str(x),{"islands":x}) for x in (1,2,4,8)],
 "migration_interval": [(str(x),{"migration_interval":x}) for x in (5,25,50)],
 "migration_pct": [(f"{x}%",{"migration_pct":x/100}) for x in (5,10,20)],
}
BASIS={"migration_interval":{"islands":4},"migration_pct":{"islands":4}}
BANDS={"small":(0,4,"#E6F4EF","малые"),
       "medium":(5,11,"#EBF2FB","средние 100–500"),
       "transition":(12,16,"#F5F4ED","переход 550–950"),
       "large":(17,23,"#FBEEEE","большие 1000–2000")}
COLORS=["#12539A","#159384","#E57C33","#8A4296","#B94C52"]

def grid(mode):
    return SMOKE_DIMS if mode=="smoke" else DIMENSIONS

def positions(groups):
    result={}
    for name,sizes in groups.items():
        left,right,_,_=BANDS[name]
        for i,n in enumerate(sizes):
            result[name,n]=left+(right-left)*i/max(1,len(sizes)-1)
    return result

def settings_for(name,label,base=BASE):
    items=dict(FACTORS[name])
    if label not in items:raise KeyError((name,label))
    start=replace(base,**BASIS.get(name,{}))
    setting=replace(start,**items[label])
    before=asdict(start);after=asdict(setting)
    diffs={key for key in before if before[key]!=after[key]}
    field=list(items[label])
    if len(field)!=1 or not diffs.issubset(field):
        raise ValueError(f"Not one-factor: {name}, {label}, {diffs}")
    return setting,start

def write_csv(path,rows):
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf8") as f:
        out=csv.DictWriter(f,fieldnames=list(rows[0]))
        out.writeheader();out.writerows(rows)

def summarize(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r["factor"],r["value"],r["class"],int(r["size"]))].append(r)
    summary=[]
    for (factor,value,cls,n),vv in sorted(groups.items()):
        summary.append(dict(factor=factor,value=value,**{"class":cls},size=n,
             runs=len(vv),mean_gap=float(np.mean([float(r["gap_pct"]) for r in vv])),
             std_gap=float(np.std([float(r["gap_pct"]) for r in vv],ddof=1)) if len(vv)>1 else 0.,
             mean_seconds=float(np.mean([float(r["elapsed_seconds"]) for r in vv]))))
    return summary

def plot(summary,path,mode="full",curve="pchip"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import PchipInterpolator
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({"font.family":"Montserrat","font.size":11,
              "axes.unicode_minus":False,"svg.fonttype":"none"})
    groups=grid(mode)
    xs=positions(groups)
    dest=Path(path);dest.mkdir(parents=True,exist_ok=True)
    for param,levels in FACTORS.items():
        for metric,suffix in (("mean_gap","error"),("mean_seconds","time")):
            fig,ax=plt.subplots(figsize=(13.4,7.1),dpi=180,facecolor="white")
            fig.subplots_adjust(left=.10,right=.98,top=.96,bottom=.28)
            for cls in groups:
                left,right,color,_=BANDS[cls]
                ax.axvspan(left-.34,right+.34,facecolor=color,alpha=.86,zorder=0)
            for i,(label,_) in enumerate(levels):
                records={(v["class"],int(v["size"])):v for v in summary
                         if v["factor"]==param and v["value"]==label}
                pts=sorted(((xs[cl,n],float(records[cl,n][metric]))
                     for cl in groups for n in groups[cl] if (cl,n) in records))
                if not pts:continue
                x=np.array([p[0] for p in pts]);y=np.array([p[1] for p in pts])
                if len(x)>2 and curve=="pchip":
                    xf=np.linspace(x[0],x[-1],max(100,8*len(x)))
                    yf=PchipInterpolator(x,y)(xf)
                    ax.plot(xf,np.maximum(0,yf),lw=2.8,color=COLORS[i],
                            label=label,solid_capstyle="round")
                else:
                    ax.plot(x,y,lw=2.8,color=COLORS[i],label=label)
                ax.scatter(x,y,s=24,alpha=.48,color=COLORS[i],edgecolors="white",
                           linewidth=.5,zorder=6)
            ax.grid(axis="y",color="#DEE4EB",linewidth=.8)
            ax.set_axisbelow(True)
            ax.set_xlim(-.48,23.48)
            ax.set_ylim(bottom=0)
            ticks=[]
            for cls,sizes in groups.items():
                for i,n in enumerate(sizes):
                    if cls=="small" or i==0 or i==len(sizes)-1 or i%2==0:
                        ticks.append((xs[cls,n],str(n)))
            ax.set_xticks([a for a,_ in ticks])
            ax.set_xticklabels([b for _,b in ticks],fontsize=8.3)
            ax.tick_params(axis="both",colors="#536477",length=0,pad=8)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
            ax.set_ylabel("Средняя ошибка относительно оптимума, %"
                          if suffix=="error" else "Время работы ГА, с",
                          color="#243E5C",fontsize=12,labelpad=9)
            ax.set_xlabel("Размерность матрицы n × n (классы показаны равными полосами)",
                          color="#243E5C",fontsize=11.5,labelpad=65)
            for edge in ax.spines.values():edge.set_visible(False)
            ax.legend(title="Значение параметра",loc="upper left",
                      ncol=min(3,len(levels)),fontsize=10,title_fontsize=10,
                      frameon=True,framealpha=.97,facecolor="white",
                      edgecolor="#D7E0EA",borderpad=.65)
            for cls in groups:
                left,right,_,label=BANDS[cls]
                ax.text((left+right)/2,-.102,label,
                        ha="center",va="top",transform=ax.get_xaxis_transform(),
                        color={"small":"#238A75","medium":"#326DA8",
                               "transition":"#867B5A","large":"#B56861"}[cls],
                        fontsize=9.5,fontweight="bold")
            file=dest/f"{param}_{suffix}"
            fig.savefig(str(file)+".png",dpi=180,facecolor="white")
            fig.savefig(str(file)+".svg",facecolor="white")
            plt.close(fig)

def run_dense(output="results/dense_ga_study",mode="smoke",repeats=1,
              workers=2,seed=20261104,factors=None):
    if mode not in ("full","smoke") or repeats<1:raise ValueError("invalid mode/repeats")
    factors=list(FACTORS) if not factors else list(factors)
    if set(factors)-set(FACTORS):raise ValueError("unknown factors")
    dest=Path(output);dest.mkdir(parents=True,exist_ok=True)
    records=[]
    for cls,sizes in grid(mode).items():
        for n in sizes:
            for rep in range(repeats):
                matrix_seed=seed+n*2999+rep*100003
                mat=np.random.default_rng(matrix_seed).integers(1,101,(n,n),dtype=np.int32)
                ii,jj=linear_sum_assignment(mat)
                exact=int(mat[ii,jj].sum(dtype=np.int64))
                ga_seed=matrix_seed+101
                for factor in factors:
                    for label,_ in FACTORS[factor]:
                        cfg,basis=settings_for(factor,label)
                        result=run(mat,cfg,seed=ga_seed,workers=workers)
                        records.append(dict(factor=factor,value=label,**{"class":cls},
                             size=n,repeat=rep,matrix_seed=matrix_seed,
                             ga_seed=ga_seed,optimal_cost=exact,ga_cost=int(result["cost"]),
                             gap_pct=100*(int(result["cost"])-exact)/exact,
                             elapsed_seconds=float(result["seconds"]),
                             population=int(result["total_population"]),
                             islands=int(result["num_islands"]),
                             generations=int(result["generations"]),
                             settings_json=json.dumps(asdict(cfg),sort_keys=True),
                             basis_json=json.dumps(asdict(basis),sort_keys=True)))
            print(f"Measured {cls} {n} x {n}; rows={len(records)}",flush=True)
            write_csv(dest/"raw.csv",records)
            write_csv(dest/"summary.csv",summarize(records))
    if set(factors)==set(FACTORS):
        plot(summarize(records),dest/"plots",mode=mode)
    (dest/"metadata.json").write_text(json.dumps({
       "mode":mode,"seed":seed,"repeats":repeats,"sizes":grid(mode),
       "factors":{k:[x for x,_ in FACTORS[k]] for k in factors},
       "x_axis":"piecewise equal-width labeled classes, with genuine transition measurements",
       "curves":"PCHIP is visualization only, every raw dot is a measured n; no intermediate data",
       "experiment":"one-factor-at-a-time, constant named baseline per factor, common matrix/GA seeds",
       "local_search":"optional, SciPy optimal assignment used only for exact benchmark reference"
    },ensure_ascii=False,indent=2),encoding="utf8")
    return records

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results/dense_ga_study")
    p.add_argument("--mode",choices=["smoke","full"],default="smoke")
    p.add_argument("--repeats",type=int,default=1)
    p.add_argument("--seed",type=int,default=20261104)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--factors",nargs="*")
    a=p.parse_args()
    run_dense(a.output,a.mode,a.repeats,a.workers,a.seed,a.factors)
