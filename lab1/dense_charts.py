"""Only clean scientific charts: no slide titles or explanatory side cards.

Every plotted marker is an actual measurement; the PCHIP connecting curve
is for presentation smoothness only, and never enters CSV or optimization.
"""
from pathlib import Path
import argparse
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager,ticker
from dense_research import SIZES,LEVELS,load_csv

COLORS=("#14599B","#17967F","#D77C33","#914E94","#BF5862")
NAMES={"small":"малые","medium":"средние",
       "transition":"переход 550–950","large":"большие"}
FILLS={"small":"#EAF7F2","medium":"#EBF2FB",
       "transition":"#F3F1FA","large":"#FCF2F0"}
COLORS_BANDS={"small":"#1E8F7A","medium":"#316AA0",
              "transition":"#75709B","large":"#B96857"}
SYMBOLS={"raw":"1/C","rank":"ранговый","True":"да","False":"нет"}

def render(root):
    out=Path(root)
    summary=load_csv(out/"sweep_raw.csv")
    expected=sum(map(len,SIZES.values()))*sum(map(len,LEVELS.values()))
    assert len(summary)==expected,(len(summary),expected)
    fonts=[f for f in font_manager.findSystemFonts() if "Montserrat" in f]
    if fonts:
        for font in fonts:font_manager.fontManager.addfont(font)
    plt.rcParams.update({"font.family":"Montserrat" if fonts else "DejaVu Sans",
             "font.size":11,"svg.fonttype":"none","axes.unicode_minus":False})
    plotdir=out/"plots";plotdir.mkdir(parents=True,exist_ok=True)
    keys=[(cls,n) for cls,ns in SIZES.items() for n in ns]
    lookup={key:i for i,key in enumerate(keys)}
    zones={}
    for cls in SIZES:
        idx=[lookup[cls,n] for n in SIZES[cls]]
        zones[cls]=(min(idx),max(idx))
    for parameter,values in LEVELS.items():
        for field,label,unit in (
            ("error_pct","error","Относительная ошибка, %"),
            ("seconds","runtime","Продолжительность, с")
        ):
            fig,ax=plt.subplots(figsize=(13.8,7.2),dpi=165,facecolor="white")
            fig.subplots_adjust(left=.098,right=.984,top=.962,bottom=.28)
            for cls,(left,right) in zones.items():
                ax.axvspan(left-.49,right+.49,color=FILLS[cls],alpha=.93,zorder=0)
            for j in list(zones.values())[:-1]:
                ax.axvline(j[1]+.5,color="#CCD5DF",lw=.8)
            ax.set_axisbelow(True)
            ax.grid(axis="y",color="#DDE4ED",linewidth=.85)
            biggest=0
            for k,val in enumerate(values):
                record={(r["class_name"],int(r["n"])):r for r in summary
                        if r["parameter"]==parameter and r["value"]==str(val)}
                ys=np.array([float(record[key][field]) for key in keys])
                x=np.arange(len(keys),dtype=float)
                xx=np.linspace(0,len(keys)-1,len(keys)*12)
                smooth=PchipInterpolator(x,ys)(xx)
                biggest=max(biggest,float(np.max(ys)))
                ax.plot(xx,smooth,lw=2.7,color=COLORS[k],
                        label=SYMBOLS.get(str(val),str(val)),
                        zorder=4,solid_capstyle="round")
                ax.scatter(x,ys,s=25,color=COLORS[k],alpha=.43,
                           edgecolors="white",linewidths=.45,zorder=5)
            if field=="error_pct" and biggest>1000:
                ax.set_yscale("symlog",linthresh=100,base=10)
                ticks=[0,20,50,100,300,1000,3000,5000,10000]
                ax.set_yticks([v for v in ticks if v<biggest*1.3])
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
                unit+=" · symlog"
            ax.set_ylim(0,max(.1,biggest*1.22))
            ax.set_xlim(-.65,len(keys)-.35)
            xindices=[]
            for cls,ns in SIZES.items():
                for i,n in enumerate(ns):
                    if i%2==0 or i==len(ns)-1 or n in (50,100,500,550,950,1000,2000):
                        xindices.append(lookup[cls,n])
            xindices=sorted(set(xindices))
            ax.set_xticks(xindices)
            ax.set_xticklabels([str(keys[i][1]) for i in xindices],fontsize=8.7)
            ax.tick_params(axis="both",pad=8,length=0,colors="#43566B")
            ax.set_xlabel("Размерность матрицы n × n",labelpad=49,fontsize=12)
            ax.set_ylabel(unit,fontsize=12,labelpad=12)
            for spine in ax.spines.values():spine.set_visible(False)
            for cls,(left,right) in zones.items():
                ax.text((left+right)/2,-.113,NAMES[cls],
                    fontsize=10.2,color=COLORS_BANDS[cls],fontweight="bold",
                    ha="center",va="top",transform=ax.get_xaxis_transform())
            ax.legend(title="Значение параметра",loc="upper left",
                      ncol=2 if len(values)>3 else 1,fontsize=10,
                      title_fontsize=10,facecolor="white",framealpha=.97,
                      edgecolor="#CDD8E0")
            fig.savefig(plotdir/f"{parameter}_{label}.png",dpi=165)
            fig.savefig(plotdir/f"{parameter}_{label}.svg")
            plt.close(fig)
        print(f"Rendered {parameter}: real dots + PCHIP",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",default="results/dense_ofat_20261027")
    render(p.parse_args().root)
