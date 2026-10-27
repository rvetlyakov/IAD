"""Evidence-only plots and parameter sensitivity for three size classes.

Piecewise-linear X retains continuous numeric order and gives each class
a visible third of the figure. Every vertex is a measured average;
there is no PCHIP, fictitious point or blank transition interval.
"""
import argparse,csv
from pathlib import Path
from collections import defaultdict
from statistics import mean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from staged_tuning import CLASS_GRID,LEVELS,read,save,class_for

COLORS=("#155A9D","#17907C","#E07F39","#914992","#C6535B")
AREAS={"small":"#E9F6F2","medium":"#EAF1FA","large":"#FAEFED"}
LABELS={"small":"малые · n<100","medium":"средние · 100–499",
        "large":"большие · 500–2000"}

def x_value(n):
    """Continuous, piecewise-linear (not equal-distance category) axis."""
    if not 2<=n<=2000:raise ValueError(n)
    if n<100:return (n-2)/98.
    if n<500:return 1+(n-100)/400.
    return 2+(n-500)/1500.

def sensitivity(out):
    """Compare OFAT effect as range of class-mean errors, in percentage points.

    This is a descriptive sensitivity measure, not a causal ANOVA or a
    statistical significance test. For context-dependent factors the
    fixed explicit context recorded in each row applies.
    """
    out=Path(out)
    rows=[]
    for group in CLASS_GRID:
        path=out/f"{group}_ofat.csv"
        if path.exists():rows.extend(read(path))
    groups=defaultdict(list)
    for r in rows:
        groups[(r["class_name"],r["parameter"],r["value"])].append(r)
    means={}
    for key,data in groups.items():
        means[key]=(mean(float(r["error_pct"]) for r in data),
                    mean(float(r["seconds"]) for r in data),len(data))
    best=[];effects=[]
    for group in CLASS_GRID:
        for factor in LEVELS:
            sub=[(v,stat) for (cl,p,v),stat in means.items()
                 if cl==group and p==factor]
            if not sub:continue
            winner=min(sub,key=lambda z:(z[1][0],z[1][1]))
            lo=min(v[1][0] for v in sub)
            hi=max(v[1][0] for v in sub)
            best.append(dict(class_name=group,parameter=factor,
                best_observed_value=winner[0],mean_error_pct=winner[1][0],
                mean_seconds=winner[1][1],observations=winner[1][2],
                note="strict frozen-context OFAT; combinations not optimized"))
            effects.append(dict(class_name=group,parameter=factor,
                error_spread_percentage_points=hi-lo,
                lowest_mean_error_pct=lo,highest_mean_error_pct=hi,
                n_levels=len(sub),
                n_per_level=min(v[1][2] for v in sub)))
    effects.sort(key=lambda x:(x["class_name"],-x["error_spread_percentage_points"]))
    save(out/"parameter_effects.csv",effects)
    save(out/"best_observed_levels.csv",best)
    return effects

def render(out):
    out=Path(out)
    observed={}
    for group in CLASS_GRID:
        path=out/f"{group}_ofat.csv"
        if path.exists():observed[group]=read(path)
    if not observed:raise FileNotFoundError("No OFAT records to plot")
    font_files=[f for f in font_manager.findSystemFonts() if "Montserrat" in f]
    for f in font_files:font_manager.fontManager.addfont(f)
    plt.rcParams.update({"font.family":"Montserrat" if font_files else "DejaVu Sans",
                         "font.size":11,"svg.fonttype":"none",
                         "axes.unicode_minus":False})
    target=out/"plots"
    target.mkdir(parents=True,exist_ok=True)
    # OFAT has a fixed class-specific baseline. Do not imply joint
    # optimization simply because the curves share an axis.
    ticks=(10,30,60,90,100,200,300,400,475,
           500,600,700,800,900,1000,1200,1400,1600,1800,2000)
    for parameter,values in LEVELS.items():
        for metric,suffix,ylabel in (
            ("error_pct","error","Относительная ошибка, %"),
            ("seconds","runtime","Время работы ГА, с")):
            fig,ax=plt.subplots(figsize=(14,7.4),dpi=170,facecolor="white")
            fig.subplots_adjust(left=.10,right=.985,top=.96,bottom=.25)
            for group,left,right in (("small",0,1),("medium",1,2),("large",2,3)):
                ax.axvspan(left,right,color=AREAS[group],alpha=.92,zorder=0)
                ax.text((left+right)/2,-.12,LABELS[group],
                        color="#2E6D8C",fontsize=10.5,fontweight="bold",
                        ha="center",va="top",transform=ax.get_xaxis_transform())
            for bound in (1,2):ax.axvline(bound,color="#AFBDCB",lw=.9)
            ymax=0.
            for i,value in enumerate(values):
                points=[]
                for group,data in observed.items():
                    filtered=[r for r in data if r["parameter"]==parameter
                              and r["value"]==str(value)]
                    by_n=defaultdict(list)
                    for r in filtered:
                        n=int(r["n"])
                        if class_for(n)!=group:raise ValueError((group,n))
                        by_n[n].append(float(r[metric]))
                    for n,values_n in by_n.items():
                        points.append((n,mean(values_n)))
                if not points:continue
                points.sort()
                xs=[x_value(n) for n,y in points]
                ys=[y for n,y in points]
                ymax=max(ymax,max(ys))
                ax.plot(xs,ys,lw=1.9,marker="o",markersize=3.3,
                        color=COLORS[i%len(COLORS)],label=str(value),zorder=3)
            ax.set_axisbelow(True);ax.grid(axis="y",color="#DFE7EF",linewidth=.85)
            ax.set_ylabel(ylabel,fontsize=12)
            if suffix=="error" and ymax>=1200:
                ax.set_yscale("symlog",linthresh=100,base=10)
            ax.set_ylim(0,max(1,ymax*1.15))
            ax.set_xlim(-.03,3.03)
            ax.set_xticks([x_value(n) for n in ticks])
            ax.set_xticklabels([str(n) for n in ticks],fontsize=8.2)
            ax.set_xlabel("Размерность n × n · непрерывные участки шкалы",
                          fontsize=12,labelpad=48)
            for spine in ax.spines.values():spine.set_visible(False)
            ax.tick_params(axis="both",length=0,pad=8,colors="#44556B")
            handles,labels=ax.get_legend_handles_labels()
            if handles:
                ax.legend(title="Значение параметра",loc="upper left",
                    ncol=2,fontsize=9.5,edgecolor="#CDD8E0",
                    framealpha=.97,facecolor="white")
            fig.savefig(target/f"{parameter}_{suffix}.png",dpi=170)
            fig.savefig(target/f"{parameter}_{suffix}.svg")
            plt.close(fig)
    return sensitivity(out)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",default="results/staged_tuning_v2")
    args=p.parse_args()
    effects=render(args.root)
    print(f"Finished verified OFAT charts and {len(effects)} sensitivity comparisons.")

if __name__=="__main__":main()
