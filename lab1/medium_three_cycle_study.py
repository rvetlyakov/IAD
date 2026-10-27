"""Reproducible three-cycle training, controlled OFAT and size sweep.

No fabricated samples: charts only connect actually measured points with PCHIP.
Training uses separate seeds from SURVEY and HOLDOUT. The original GA
operators remain available (cycle3=False).
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from dataclasses import asdict,replace
from collections import defaultdict
from statistics import mean
import numpy as np
from scipy.optimize import linear_sum_assignment
from hybrid_ga import Config,run

BASE=Config(population=120,generations=250,mutation_prob=.7,inversion_pct=2,
            elitism=.2,selection="rank",rank_power=2,adaptive=True,
            adaptive_horizon=250,local_every=25,local_top=1,
            local_passes=2,local_candidates=16,cycle3=False,islands=1)
TRAIN_SIZES=(150,300,450)
SURVEY_SIZES={
 "small":(10,30,60,90),
 "medium":tuple(range(100,501,25)),
 "transition":tuple(range(550,1000,50)),
 "large":(1000,1250,1500,1750,2000)
}
HOLDOUT_SIZES=(100,300,500)
TRAIN_K=(8,16,32)
TRAIN_CYCLE_PASSES=(1,2,3)
OFAT={
 "cycle3":(False,True),
 "generations":(100,250,500,1000),
 "islands":(1,2,4,8),
 "local_every":(10,25,50),
}
def save(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
def load(path):
    with Path(path).open(newline="",encoding="utf8") as f:
        return list(csv.DictReader(f))
def problem(n,seed):
    a=np.random.default_rng(seed).integers(1,101,(n,n),dtype=np.int32)
    rows,cols=linear_sum_assignment(a)
    return a,int(a[rows,cols].sum(dtype=np.int64))
def measure(a,exact,c,seed,workers=2):
    out=run(a,c,seed=seed,workers=workers)
    return {"ga_cost":int(out["cost"]),"error_pct":100*(out["cost"]-exact)/exact,
        "seconds":out["seconds"],"population":out["total_population"],
        "generations":out["generations"],"islands":out["num_islands"],
        "local_calls":out["local_calls"]}
def training(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for n in TRAIN_SIZES:
        for repeat in range(2):
            seed=30301001+n*739+repeat*1777
            a,exact=problem(n,seed)
            for k in TRAIN_K:
                for passes in TRAIN_CYCLE_PASSES:
                    c=replace(BASE,cycle3=True,local_candidates=k,cycle3_passes=passes)
                    result=measure(a,exact,c,seed+41)
                    rows.append(dict(size_class="medium",size=n,repeat=repeat,
                         candidate_k=k,cycle_passes=passes,matrix_seed=seed,
                         ga_seed=seed+41,exact=exact,**result,
                         settings_json=json.dumps(asdict(c),sort_keys=True)))
        save(folder/"train.csv",rows)
        print("TRAIN",n,len(rows),flush=True)
    ranking=[]
    for k in TRAIN_K:
        for passes in TRAIN_CYCLE_PASSES:
            selected=[r for r in rows if r["candidate_k"]==k and r["cycle_passes"]==passes]
            ranking.append(dict(k=k,passes=passes,
                mean_error_pct=mean(r["error_pct"] for r in selected),
                mean_seconds=mean(r["seconds"] for r in selected),
                count=len(selected)))
    ranking.sort(key=lambda r:(r["mean_error_pct"],r["mean_seconds"]))
    winner=ranking[0]
    selected=replace(BASE,cycle3=True,local_candidates=int(winner["k"]),
                     cycle3_passes=int(winner["passes"]))
    save(folder/"training_ranking.csv",ranking)
    (folder/"selected.json").write_text(json.dumps({
       "method":"minimum paired TRAIN mean error; ties shortest wall time",
       "selection":winner,"selected_settings":asdict(selected),
       "training_sizes":TRAIN_SIZES,"training_repeats":2,
       "validation_disjoint":"SURVEY and HOLDOUT use different matrix seeds"
       },indent=2,ensure_ascii=False),encoding="utf8")
    return selected

def survey(folder,selected):
    folder=Path(folder);rows=[]
    for group,sizes in SURVEY_SIZES.items():
        for n in sizes:
            seed=40400129+n*877
            a,exact=problem(n,seed)
            for parameter,levels in OFAT.items():
                for value in levels:
                    c=replace(selected,**{parameter:value})
                    diff={key for key in asdict(selected)
                          if getattr(selected,key)!=getattr(c,key)}
                    assert diff<={parameter},diff
                    result=measure(a,exact,c,seed+17)
                    rows.append(dict(size_class=group,size=n,parameter=parameter,
                         value=str(value),matrix_seed=seed,ga_seed=seed+17,exact=exact,
                         **result,baseline_json=json.dumps(asdict(selected),sort_keys=True),
                         settings_json=json.dumps(asdict(c),sort_keys=True)))
            save(folder/"survey.csv",rows)
            print("SURVEY",group,n,len(rows),flush=True)
    return rows

def holdout(folder,selected):
    folder=Path(folder);rows=[]
    for n in HOLDOUT_SIZES:
        for repeat in range(5):
            seed=50500101+n*911+repeat*9619
            a,exact=problem(n,seed)
            for enabled in (False,True):
                c=replace(selected,cycle3=enabled)
                result=measure(a,exact,c,seed+19)
                rows.append(dict(size=n,repeat=repeat,cycle3=enabled,matrix_seed=seed,
                     ga_seed=seed+19,exact=exact,**result,
                     settings_json=json.dumps(asdict(c),sort_keys=True)))
        save(folder/"holdout.csv",rows)
        print("HOLDOUT",n,len(rows),flush=True)
    return rows

def plot(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from scipy.interpolate import PchipInterpolator
    folder=Path(folder)
    rows=load(folder/"survey.csv")
    font=[f for f in font_manager.findSystemFonts() if "Montserrat" in f]
    for f in font:font_manager.fontManager.addfont(f)
    plt.rcParams.update({"font.family":"Montserrat" if font else "DejaVu Sans",
                         "font.size":11,"svg.fonttype":"none",
                         "axes.unicode_minus":False})
    width={g:i for i,g in enumerate(SURVEY_SIZES)}
    positions={}
    borders={}
    cursor=0
    for group,sizes in SURVEY_SIZES.items():
        start=cursor
        for n in sizes:positions[group,n]=cursor;cursor+=1
        borders[group]=(start,cursor-1)
        cursor+=1
    names={"small":"малые","medium":"средние 100–500",
           "transition":"переход 550–950","large":"большие 1000–2000"}
    fills={"small":"#EAF7F2","medium":"#EAF0FA",
           "transition":"#F2F1F7","large":"#FAEFED"}
    colors=("#1C5A98","#129080","#DC823D","#934A96")
    out=folder/"plots";out.mkdir(exist_ok=True,parents=True)
    for parameter,levels in OFAT.items():
        for metric,suffix in (("error_pct","error"),("seconds","runtime")):
            fig,ax=plt.subplots(figsize=(14,7.5),dpi=170,facecolor="white")
            fig.subplots_adjust(left=.095,right=.985,top=.96,bottom=.27)
            ymax=0
            for group,(left,right) in borders.items():
                ax.axvspan(left-.55,right+.55,color=fills[group],alpha=.9,zorder=0)
            for idx,value in enumerate(levels):
                for group,sizes in SURVEY_SIZES.items():
                    selected={int(r["size"]):r for r in rows
                              if r["parameter"]==parameter and r["value"]==str(value)
                              and r["size_class"]==group}
                    xs=np.asarray([positions[group,n] for n in sizes],dtype=float)
                    ys=np.asarray([float(selected[n][metric]) for n in sizes])
                    ymax=max(ymax,*ys)
                    # Lines are graphical interpolation; only dots are measurements.
                    xx=np.linspace(xs[0],xs[-1],len(xs)*12)
                    yy=PchipInterpolator(xs,ys)(xx) if len(xs)>=3 else np.interp(xx,xs,ys)
                    ax.plot(xx,yy,color=colors[idx],lw=2.7,
                            label=str(value) if group=="small" else "_nolegend_")
                    ax.scatter(xs,ys,color=colors[idx],alpha=.5,s=25,
                               edgecolors="white",linewidth=.5,zorder=4)
            ax.set_axisbelow(True);ax.grid(axis="y",color="#E1E8F0")
            ax.set_ylim(0,max(1,ymax*1.2))
            ax.set_xlim(-.65,cursor-1.35)
            if metric=="error_pct" and ymax>1000:
                ax.set_yscale("symlog",linthresh=100,base=10)
            tickpos=[];ticklabels=[]
            for group,sizes in SURVEY_SIZES.items():
                stride=2 if group in ("medium","transition","large") else 1
                for i,n in enumerate(sizes):
                    if i%stride==0 or i==len(sizes)-1:
                        tickpos.append(positions[group,n]);ticklabels.append(str(n))
                l,r=borders[group]
                ax.text((l+r)/2,-.115,names[group],
                        transform=ax.get_xaxis_transform(),ha="center",va="top",
                        fontsize=10.5,fontweight="bold",color="#426A8E")
            ax.set_xticks(tickpos);ax.set_xticklabels(ticklabels,fontsize=9)
            ax.set_xlabel("Размерность матрицы n × n",labelpad=48)
            ax.set_ylabel("Относительная ошибка, %" if suffix=="error" else "Время выполнения, с")
            ax.tick_params(axis="both",length=0,pad=8,colors="#42556A")
            for s in ax.spines.values():s.set_visible(False)
            ax.legend(title="Значение параметра",loc="upper left",ncol=2,
                      facecolor="white",framealpha=.96,edgecolor="#CFD8E2")
            fig.savefig(out/f"{parameter}_{suffix}.png",dpi=170,facecolor="white")
            fig.savefig(out/f"{parameter}_{suffix}.svg",facecolor="white")
            plt.close(fig)
    return len(list(out.glob("*.png")))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="results/medium_3cycle_20261028")
    parser.add_argument("--stage",choices=("all","train","survey","holdout","plots"),default="all")
    opt=parser.parse_args()
    folder=Path(opt.output)
    if opt.stage in ("all","train"):selected=training(folder)
    else:
        meta=json.loads((folder/"selected.json").read_text())
        selected=Config(**meta["selected_settings"])
    if opt.stage in ("all","survey"):survey(folder,selected)
    if opt.stage in ("all","holdout"):holdout(folder,selected)
    if opt.stage in ("all","plots"):print("Rendered",plot(folder),"graphs",flush=True)

if __name__=="__main__":
    main()
