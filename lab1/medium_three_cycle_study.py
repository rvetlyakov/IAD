"""Reproducible three-cycle training, controlled OFAT and size sweep.

No fabricated samples: charts only connect actually measured points, without interpolation.
Training uses separate seeds from SURVEY and HOLDOUT. The original GA
operators remain available (cycle3=False).
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from dataclasses import asdict,replace
from statistics import mean
import numpy as np
from scipy.optimize import linear_sum_assignment
from hybrid_ga import Config,run

BASE=Config(population=120,generations=250,mutation_prob=.7,inversion_pct=2,
            elitism=.2,selection="rank",rank_power=2,adaptive=True,
            adaptive_horizon=250,local_every=25,local_top=1,
            local_passes=2,local_candidates=16,cycle3=False,islands=1)
TRAIN_SIZES=(150,300,450)
# One continuous measured grid: all n=500..2000 are LARGE, including 550..950.
# The lower endpoints 100 and 500 belong to the next class (no overlap).
SURVEY_SIZES={
 "small":(10,30,60,90),
 "medium":tuple(range(100,500,25)),
 "large":(500,)+tuple(range(550,1000,50))+(1000,1250,1500,1750,2000)
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

def dimension_x(n):
    """Continuous piecewise-linear axis: equal visible width per size class.

    Not a category index: n=100 and n=500 have exactly matching boundaries,
    and all true measured n are in increasing order with NO vacant transition.
    """
    if not 2 <= n <= 2000:
        raise ValueError("Supported research sizes: 2..2000")
    if n < 100:
        return (n - 2) / 98.
    if n < 500:
        return 1. + (n - 100) / 400.
    return 2. + (n - 500) / 1500.


def class_for(n):
    if 2 <= n < 100: return "small"
    if 100 <= n < 500: return "medium"
    if 500 <= n <= 2000: return "large"
    raise ValueError(n)


def plot(folder):
    """Plot ONLY measured points with straight line segments; no PCHIP.

    Historical survey.csv rows retain their earlier size_class values, but
    they are reclassified *by numeric n* without rerunning or altering them.
    This preserves the authentic 35-size measurements in the new classification.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    folder=Path(folder)
    rows=load(folder/"survey.csv")
    fonts=[f for f in font_manager.findSystemFonts() if "Montserrat" in f]
    for f in fonts:font_manager.fontManager.addfont(f)
    plt.rcParams.update({"font.family":"Montserrat" if fonts else "DejaVu Sans",
                         "font.size":11,"svg.fonttype":"none",
                         "axes.unicode_minus":False})
    all_sizes=sorted(set(int(r["size"]) for r in rows))
    expected=sorted(n for ns in SURVEY_SIZES.values() for n in ns)
    if all_sizes != expected:
        raise ValueError(f"Survey grid differs from expected: {all_sizes} vs {expected}")
    assert all(a<b for a,b in zip(all_sizes,all_sizes[1:]))
    fills={"small":"#EAF7F2","medium":"#EAF0FA","large":"#FAEFED"}
    colors=("#1C5A98","#129080","#DC823D","#934A96")
    names={"small":"малые · n < 100","medium":"средние · 100–499",
           "large":"большие · 500–2000"}
    out=folder/"plots";out.mkdir(exist_ok=True,parents=True)
    for parameter,levels in OFAT.items():
        for metric,suffix in (("error_pct","error"),("seconds","runtime")):
            fig,ax=plt.subplots(figsize=(14,7.5),dpi=170,facecolor="white")
            fig.subplots_adjust(left=.095,right=.985,top=.95,bottom=.245)
            for group,left,right in (("small",0,1),("medium",1,2),("large",2,3)):
                ax.axvspan(left,right,color=fills[group],alpha=.9,zorder=0)
                ax.text((left+right)/2,-.115,names[group],ha="center",
                        va="top",transform=ax.get_xaxis_transform(),
                        fontsize=10.5,fontweight="bold",color="#426A8E")
            for edge in (1.,2.):
                ax.axvline(edge,color="#B8C8D6",linewidth=.8)
            ymax=0.
            for idx,value in enumerate(levels):
                chosen=[r for r in rows if r["parameter"]==parameter and
                        r["value"]==str(value)]
                sizes=[int(r["size"]) for r in chosen]
                if len(chosen)!=len(expected) or len(set(sizes))!=len(expected):
                    raise ValueError(f"Missing or duplicate observations for {parameter}={value}")
                chosen.sort(key=lambda r:int(r["size"]))
                xs=[dimension_x(int(r["size"])) for r in chosen]
                ys=[float(r[metric]) for r in chosen]
                ymax=max(ymax,max(ys))
                # Every vertex is an actual measurement; no interpolated samples.
                ax.plot(xs,ys,color=colors[idx],lw=2.,marker="o",
                        markersize=3.4,label=str(value),alpha=.92,zorder=3)
            ax.set_axisbelow(True);ax.grid(axis="y",color="#E1E8F0")
            ax.set_ylim(0,max(1,ymax*1.15));ax.set_xlim(-.035,3.035)
            if metric=="error_pct" and ymax>1000:
                ax.set_yscale("symlog",linthresh=100,base=10)
            ticks=(10,30,60,90,100,200,300,400,475,
                   500,600,700,800,900,1000,1250,1500,1750,2000)
            ax.set_xticks([dimension_x(n) for n in ticks])
            ax.set_xticklabels([str(n) for n in ticks],fontsize=8.6)
            ax.set_xlabel("Размерность матрицы n × n · непрерывная шкала по классам",
                          labelpad=48)
            ax.set_ylabel("Относительная ошибка, %" if suffix=="error"
                          else "Время выполнения, с")
            ax.tick_params(axis="both",length=0,pad=8,colors="#42556A")
            for spine in ax.spines.values():spine.set_visible(False)
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
