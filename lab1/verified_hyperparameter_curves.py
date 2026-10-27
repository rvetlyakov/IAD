"""Reclassify and replot the 1855 REAL dense OFAT trials as 3 classes.

This is a reanalysis of a historic (non-3cycle) experiment, NOT a fresh
run or evidence that the new 3cycle hybrid has already been fully tuned.
No interpolation or invented measurements are used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dense_research import LEVELS, SIZES, load_csv, save_csv
from matplotlib import font_manager
from staged_charts import x_value
from staged_tuning import class_for

BACKGROUND={"small":"#EAF7F2","medium":"#EBF2FA","large":"#FBF0EE"}
COLORS=("#165D9C","#16927D","#D78338","#914B96","#BD535A")
NAMES={"small":"малые · n<100","medium":"средние · 100–499",
       "large":"большие · 500–2000"}
TICKS=(5,10,20,30,40,50,100,200,300,400,500,600,700,800,
       900,1000,1200,1400,1600,1800,2000)

def source_rows(source):
    rows=load_csv(source)
    expected=sum(map(len,SIZES.values()))*sum(map(len,LEVELS.values()))
    if len(rows)!=expected:
        raise ValueError(f"Raw row count must be {expected}, got {len(rows)}")
    seen=set()
    for r in rows:
        n=int(r["n"])
        key=(n,r["parameter"],r["value"])
        if key in seen:raise ValueError(f"Duplicate measured trial: {key}")
        seen.add(key)
        # Keep numeric class independent of outdated 'transition' labels.
        r["three_class"]=class_for(n)
        if float(r["error_pct"])< -1e-8:raise ValueError("Negative optimality gap")
    all_n={int(r["n"]) for r in rows}
    original={n for sizes in SIZES.values() for n in sizes}
    if all_n!=original:raise ValueError("Changed or missing true sizes")
    if not set(range(550,1000,50)).issubset(all_n):
        raise ValueError("Transition region observations missing")
    for parameter,levels in LEVELS.items():
        for value in levels:
            sizes={int(r["n"]) for r in rows
                   if r["parameter"]==parameter and r["value"]==str(value)}
            if sizes!=all_n:raise ValueError(f"Missing {parameter}={value}")
    return rows

def analyze(rows,out):
    groups=defaultdict(list)
    for r in rows:
        groups[(r["three_class"],r["parameter"],r["value"])].append(r)
    summaries=[]
    for (cls,par,value),items in groups.items():
        summaries.append({"class_name": cls,"parameter": par,"value": value,
            "measured_sizes": len(items),"mean_error_pct": mean(
               float(r["error_pct"]) for r in items),
            "mean_seconds": mean(float(r["seconds"]) for r in items)})
    best=[];influence=[]
    for cls in ("small","medium","large"):
        for par,values in LEVELS.items():
            pairs=[r for r in summaries
                   if r["class_name"]==cls and r["parameter"]==par]
            if len(pairs)!=len(values):raise ValueError((cls,par,len(pairs)))
            winner=min(pairs,key=lambda r:(r["mean_error_pct"],r["mean_seconds"]))
            lo=min(r["mean_error_pct"] for r in pairs)
            hi=max(r["mean_error_pct"] for r in pairs)
            best.append({"class_name": cls,"parameter": par,
                "best_observed_value": winner["value"],
                "mean_error_pct": winner["mean_error_pct"],
                "mean_seconds": winner["mean_seconds"],
                "measured_sizes": winner["measured_sizes"]})
            influence.append({"class_name": cls,"parameter": par,
                "error_spread_percentage_points": hi-lo,
                "lowest_mean_error_pct": lo,"highest_mean_error_pct": hi,
                "number_of_levels": len(pairs)})
    influence.sort(key=lambda r:(r["class_name"],-r["error_spread_percentage_points"]))
    transfer=[]
    for par in LEVELS:
        donor=next(r for r in best if r["class_name"]=="large" and r["parameter"]==par)
        for target in ("small","medium"):
            own=next(r for r in best if r["class_name"]==target and r["parameter"]==par)
            on_target=next(r for r in summaries if
                    r["class_name"]==target and r["parameter"]==par and
                    r["value"]==donor["best_observed_value"])
            transfer.append({"parameter": par,"source_class": "large",
                "target_class": target,"large_winner_value": donor["best_observed_value"],
                "target_error_using_large_winner_pct": on_target["mean_error_pct"],
                "target_best_observed_value": own["best_observed_value"],
                "target_best_observed_error_pct": own["mean_error_pct"],
                "extra_error_pp": on_target["mean_error_pct"]-own["mean_error_pct"],
                "source": "already measured one-factor comparison, not combined retune"})
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    save_csv(out/"three_class_mean_by_value.csv",summaries)
    save_csv(out/"three_class_best_levels.csv",best)
    save_csv(out/"three_class_parameter_influence.csv",influence)
    save_csv(out/"large_to_small_medium_transfer.csv",transfer)
    return best,influence,transfer

def plot(rows,out):
    font_files=[f for f in font_manager.findSystemFonts() if "Montserrat" in f]
    for f in font_files:font_manager.fontManager.addfont(f)
    plt.rcParams.update({"font.family":"Montserrat" if font_files else "DejaVu Sans",
                         "font.size":10.5,"svg.fonttype":"none",
                         "axes.unicode_minus":False})
    dest=Path(out)/"plots";dest.mkdir(parents=True,exist_ok=True)
    for par,levels in LEVELS.items():
        for metric,label,ylabel in (
            ("error_pct","error","Относительная ошибка, %"),
            ("seconds","runtime","Время работы ГА, с")):
            fig,ax=plt.subplots(figsize=(14,7.5),dpi=165,facecolor="white")
            fig.subplots_adjust(left=.10,right=.985,top=.96,bottom=.25)
            for cls,left,right in (("small",0,1),("medium",1,2),("large",2,3)):
                ax.axvspan(left,right,color=BACKGROUND[cls],alpha=.91,zorder=0)
                ax.text((left+right)/2,-.115,NAMES[cls],
                    ha="center",va="top",fontsize=10,weight="bold",
                    color="#456980",transform=ax.get_xaxis_transform())
            for edge in (1,2):ax.axvline(edge,color="#B5C5D3",lw=.9)
            ymax=0.
            for i,value in enumerate(levels):
                chosen=sorted((r for r in rows
                         if r["parameter"]==par and r["value"]==str(value)),
                         key=lambda r:int(r["n"]))
                xs=[x_value(int(r["n"])) for r in chosen]
                ys=[float(r[metric]) for r in chosen]
                ymax=max(ymax,max(ys))
                # Markers are the actual experimental rows; the line follows
                # actual consecutive measured points, not a spline.
                ax.plot(xs,ys,marker="o",markersize=3.3,lw=1.9,
                        color=COLORS[i%len(COLORS)],label=str(value),zorder=3)
            ax.set_axisbelow(True);ax.grid(axis="y",color="#DFE7EF",linewidth=.8)
            if label=="error" and ymax>1200:
                ax.set_yscale("symlog",linthresh=100,base=10)
            ax.set_ylim(0,max(1,ymax*1.16));ax.set_xlim(-.035,3.035)
            ax.set_xticks([x_value(x) for x in TICKS])
            ax.set_xticklabels([str(x) for x in TICKS],fontsize=8.1)
            ax.set_ylabel(ylabel,fontsize=12)
            ax.set_xlabel("Размерность n × n · непрерывные участки шкалы",
                          fontsize=11.5,labelpad=46)
            ax.tick_params(axis="both",length=0,pad=8,colors="#44556B")
            for spine in ax.spines.values():spine.set_visible(False)
            ax.legend(title="Значение параметра",loc="upper left",ncol=2,
                      fontsize=9.5,edgecolor="#CDD8E0",
                      framealpha=.96,facecolor="white")
            fig.savefig(dest/f"{par}_{label}.png",dpi=165)
            fig.savefig(dest/f"{par}_{label}.svg")
            plt.close(fig)
    return len(list(dest.glob("*.png")))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",default="results/dense_ofat_20261027/sweep_raw.csv")
    ap.add_argument("--out",default="results/continuous_three_class_20261028")
    ap.add_argument("--skip-plots",action="store_true")
    args=ap.parse_args()
    rows=source_rows(args.source)
    best,imp,transfer=analyze(rows,args.out)
    charts=0 if args.skip_plots else plot(rows,args.out)
    print(f"RECLASSIFIED: {len(rows)} real runs, {len(best)} best levels, "
          f"{len(imp)} parameter impacts, {len(transfer)} transfer tests, "
          f"{charts} PNG chart pairs. Old data not called 3cycle results.")

if __name__=="__main__":main()
