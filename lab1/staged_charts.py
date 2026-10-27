"""Presentation-only curves for TWO-STAGE researched GA hyperparameters.

No trend interpolation across independently tuned size classes. Every faint
marker is measured; PCHIP connects points ONLY WITHIN the same class.
Stand-alone white-background plots, no slide headings or side cards.
"""
import argparse,csv
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from staged_tuning import CLASS_GRID,LEVELS,read

COLORS=["#155A9D","#17907C","#E07F39","#914992","#C6535B"]
AREAS={"small":"#E9F6F2","medium":"#EAF1FA","large":"#FAEFED"}
AREA_LABELS={"small":"малые · 2–99","medium":"средние · 100–500",
             "large":"большие · 1000–2000"}
SIZES={key:len(x) for key,x in CLASS_GRID.items()}

def render(out):
    out=Path(out)
    observed={}
    for group in CLASS_GRID:
        path=out/f"{group}_ofat.csv"
        if path.exists():
            observed[group]=read(path)
    if not observed:raise FileNotFoundError("No OFAT records to plot")
    font_files=[x for x in font_manager.findSystemFonts()
                if "Montserrat" in x]
    for f in font_files:
        font_manager.fontManager.addfont(f)
    plt.rcParams.update({"font.family":"Montserrat" if font_files else "DejaVu Sans",
                         "font.size":11,"svg.fonttype":"none",
                         "axes.unicode_minus":False})
    dirs=out/"plots"
    dirs.mkdir(parents=True,exist_ok=True)
    classes=list(observed)
    # Each class is one wide category band with real size tick labels.
    intervals={}
    x=0.
    for group in classes:
        sizes=CLASS_GRID[group]
        left=x
        intervals[group]=(left,left+len(sizes)-1)
        x=left+len(sizes)+1.4
    for parameter,values in LEVELS.items():
        for metric,suffix,ylabel in [
            ("error_pct","error","Относительная ошибка, %"),
            ("seconds","runtime","Время работы ГА, с")]:
            fig,ax=plt.subplots(figsize=(13.3,7.2),dpi=180,facecolor="white")
            fig.subplots_adjust(left=.105,right=.975,top=.965,bottom=.265)
            for group,(left,right) in intervals.items():
                ax.axvspan(left-.45,right+.45,color=AREAS[group],alpha=.9,zorder=0)
            ax.set_axisbelow(True)
            ax.grid(axis="y",color="#DFE7EF",linewidth=.85)
            largest=0.0
            for index,value in enumerate(values):
                for group in classes:
                    sub=[r for r in observed[group]
                         if r["parameter"]==parameter and r["value"]==str(value)]
                    seen={int(r["n"]):r for r in sub}
                    ns=[n for n in CLASS_GRID[group] if n in seen]
                    if not ns:continue
                    ys=np.array([float(seen[n][metric]) for n in ns])
                    xs=np.array([intervals[group][0]+CLASS_GRID[group].index(n)
                                 for n in ns],dtype=float)
                    largest=max(largest,float(ys.max()))
                    # Category-specific interpolation: not a measured observation.
                    if len(xs)>=3:
                        dense=np.linspace(xs[0],xs[-1],len(xs)*12)
                        predicted=PchipInterpolator(xs,ys)(dense)
                        ax.plot(dense,predicted,color=COLORS[index],
                                linewidth=2.6,label=str(value) if group==classes[0]
                                else "_nolegend_",solid_capstyle="round",zorder=4)
                    else:
                        ax.plot(xs,ys,color=COLORS[index],
                                linewidth=2.6,label=str(value) if group==classes[0]
                                else "_nolegend_",zorder=4)
                    ax.scatter(xs,ys,color=COLORS[index],s=28,alpha=.48,zorder=5,
                               edgecolors="white",linewidths=.5)
            if suffix=="error" and largest>=1200:
                ax.set_yscale("symlog",linthresh=100,base=10)
                ax.set_ylabel(ylabel+" · symlog",fontsize=12)
            else:
                ax.set_ylabel(ylabel,fontsize=12)
            ax.set_ylim(0,max(1,largest*1.15))
            tickpos,ticknames=[],[]
            for group in classes:
                left,right=intervals[group]
                ns=CLASS_GRID[group]
                stride=max(1,int(np.ceil(len(ns)/8)))
                for i,n in enumerate(ns):
                    if i%stride==0 or i==len(ns)-1:
                        tickpos.append(left+i);ticknames.append(str(n))
                ax.text((left+right)/2,-.115,AREA_LABELS[group],
                        color="#2E6D8C",fontsize=10.8,weight="bold",
                        ha="center",va="top",transform=ax.get_xaxis_transform())
            ax.set_xticks(tickpos)
            ax.set_xticklabels(ticknames,fontsize=9)
            ax.set_xlim(-.6,x-1.1)
            ax.tick_params(axis="both",length=0,pad=8,colors="#44556B")
            ax.set_xlabel("Размерность матрицы n × n",fontsize=12,labelpad=50)
            for spine in ax.spines.values():spine.set_visible(False)
            ax.legend(title="Значение параметра",loc="upper left",ncol=2,
                      fontsize=10,title_fontsize=10,edgecolor="#CDD8E0",
                      framealpha=.97,facecolor="white")
            fig.savefig(dirs/f"{parameter}_{suffix}.png",dpi=180)
            fig.savefig(dirs/f"{parameter}_{suffix}.svg")
            plt.close(fig)
    return len(list(dirs.glob("*.png")))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",default="results/staged_tuning_v2")
    args=p.parse_args()
    print("Rendered clean charts:",render(args.root))
