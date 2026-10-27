"""Measured OFAT hyperparameter curves, all matrix-size classes on EVERY figure.

The X axis deliberately uses equally wide class bands, not a misleading linear
distance between 10 and 2000. Each labeled X coordinate is an actual test.
Three classes: 10..50, 100..500, 1000..2000. No image-generation data.

Usage:
  python lab1/hyperparameter_curves.py --mode smoke
  python lab1/hyperparameter_curves.py --mode full --repeats 1 --workers 2
Results: results/hyperparameter_curves_10_2000/{raw.csv,summary.csv,plots/*.png}
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from lab1 import Employee, Employees
from termination import plateau_reached
from scale_benchmark import generate_matrix, exact_cost, run_variant

# All chart X coordinates (independent experiments, never interpolated).
REAL_SIZES = {
    "small": (10, 20, 30, 40, 50),
    "medium": (100, 200, 300, 400, 500),
    "large": (1000, 1250, 1500, 1750, 2000),
}
SMOKE_SIZES = {
    "small": (4, 5),
    "medium": (7, 8),
    "large": (10, 11),
}
HYPERPARAMETERS = {
    "total_population": (40, 80, 120),
    "mutation_probability": (0.4, 0.7, 1.0),
    "mutation_percent": (20, 45, 70),
    "elitist_percent": (0, 0.1, 0.2),
    "max_generations": (15, 25, 40),
    "num_islands": (1, 2, 3, 4),
    "migration_interval": (5, 10, 20),
    "migration_percent": (5, 10, 20),
    "plateau_limit": (3, 5, 10),
    "plateau_diff": (0.005, 0.02, 0.05),
}
LABELS = {
    "total_population": ("Размер популяции", "особей"),
    "mutation_probability": ("Вероятность мутации", "вероятность"),
    "mutation_percent": ("Максимальная доля инверсии", "%"),
    "elitist_percent": ("Доля элитизма", "доля"),
    "max_generations": ("Лимит поколений", "поколений"),
    "num_islands": ("Количество островов", "островов"),
    "migration_interval": ("Интервал миграции", "поколений"),
    "migration_percent": ("Доля мигрантов", "%"),
    "plateau_limit": ("Длина окна плато", "поколений"),
    "plateau_diff": ("Порог плато", "отн. величина"),
}
COLORS = ("#1761B0", "#008E81", "#E07B34", "#933D9C")
BANDS = (
    ("small", 0.0, 4.0, "#EDF8F4", "МАЛЫЕ"),
    ("medium", 5.5, 9.5, "#ECF4FE", "СРЕДНИЕ"),
    ("large", 11.0, 15.0, "#FFF2EF", "БОЛЬШИЕ"),
)
ISLAND_ONLY = {"num_islands", "migration_interval", "migration_percent"}
PLATEAU_ONLY = {"plateau_limit", "plateau_diff"}


def axis_position(group, size):
    """Categorical X with equal-width visual bands, real numeric labels."""
    values = REAL_SIZES[group]
    if size in values:
        idx=values.index(size)
        return (0 if group=="small" else 5.5 if group=="medium" else 11)+idx
    smoke=SMOKE_SIZES[group]
    if size in smoke:
        idx=smoke.index(size)
        return (0 if group=="small" else 5.5 if group=="medium" else 11)+idx*4
    raise ValueError(f"unmeasured dimension {group} / {size}")


def run_plateau_variant(matrix, settings, seed):
    """Same operators, starting population and best-so-far as ordinary GA.

    Early stopping is measured on the mean post-selection fitness. This
    experiment is SEPARATE from the equal-generation island comparison.
    """
    n=len(matrix)
    rng=random.Random(seed)
    initial=[rng.sample(range(n),n) for _ in range(int(settings["total_population"]))]
    random.seed(seed+1)
    ga=Employees.__new__(Employees)
    ga.matrix=matrix
    ga.n=len(initial)
    ga.mutation_probability=float(settings["mutation_probability"])
    ga.mutation_percent=float(settings["mutation_percent"])
    ga.elitist_percent=float(settings["elitist_percent"])
    ga.t=2
    ga.population=[Employee(x,n,matrix,ga.mutation_percent) for x in initial]
    best=int(ga.get_best()[1])
    history=[ga.fitness()]
    t0=time.perf_counter()
    generations=0
    for generation in range(int(settings["max_generations"])):
        ga.mutate()
        best=min(best,int(ga.get_best()[1]))
        ga.select()
        best=min(best,int(ga.get_best()[1]))
        history.append(float(ga.fitness()))
        generations=generation+1
        if plateau_reached(history,int(settings["plateau_limit"]),
                           float(settings["plateau_diff"])):
            break
    return {"cost":best, "seconds":time.perf_counter()-t0,
            "generations":generations}


def _settings_for(group, parameter, full_settings, mode):
    strategy="islands" if parameter in ISLAND_ONLY else "single"
    cfg=dict(full_settings[group][strategy])
    if parameter in PLATEAU_ONLY:
        cfg["num_islands"]=1
        cfg["plateau_limit"]=5
        cfg["plateau_diff"]=0.02
    if mode=="smoke":
        cfg.update(total_population=12,max_generations=6,
                   migration_interval=3,mutation_percent=40)
        if parameter in PLATEAU_ONLY:
            cfg.update(plateau_limit=2,plateau_diff=.02)
    return cfg


def _levels(parameter, mode):
    if mode=="smoke":
        if parameter=="num_islands": return (1,2,3)
        if parameter=="plateau_diff": return (.005,.05)
        return HYPERPARAMETERS[parameter][:2]
    return HYPERPARAMETERS[parameter]


def _csv(path, rows):
    if not rows:
        return
    with Path(path).open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(raw):
    grouped=defaultdict(list)
    for r in raw:
        grouped[(r["parameter"], r["class"],int(r["size"]),str(r["value"]))].append(r)
    summary=[]
    for (parameter,group,size,value),items in sorted(grouped.items()):
        summary.append({
            "parameter":parameter, "class":group, "size":size,
            "value":value, "count":len(items),
            "mean_error_percent":statistics.mean(float(x["error_percent"]) for x in items),
            "mean_seconds":statistics.mean(float(x["seconds"]) for x in items),
            "mean_generations":statistics.mean(int(x["generations"]) for x in items),
            "min_error_percent":min(float(x["error_percent"]) for x in items),
            "max_error_percent":max(float(x["error_percent"]) for x in items),
        })
    return summary


def render_parameter(parameter, summaries, output, *, mode, repeats=1):
    """16:9 Montserrat chart with three equally sized size-class regions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    title,unit=LABELS[parameter]
    rows=[r for r in summaries if r["parameter"]==parameter]
    if not rows:
        return
    classes=REAL_SIZES if mode=="full" else SMOKE_SIZES
    plt.rcParams.update({
        "font.family":"Montserrat", "font.size":11,
        "axes.unicode_minus":False, "svg.fonttype":"none",
        "savefig.facecolor":"#FFFFFF",
    })
    fig=plt.figure(figsize=(13.333,7.5),dpi=160,facecolor="white")
    fig.patches.append(FancyBboxPatch(
        (.015,.84),.97,.15,boxstyle="round,pad=0.001,rounding_size=0.027",
        transform=fig.transFigure,facecolor="#1C4B8D",edgecolor="none",zorder=1,
    ))
    fig.text(.055,.935,"ИССЛЕДОВАНИЕ ГЕНЕТИЧЕСКОГО АЛГОРИТМА",
             color="#DDEBFC",fontsize=10,fontweight="medium",zorder=2)
    fig.text(.055,.876,title.upper(),color="white",fontsize=21,
             fontweight="bold",zorder=2)
    fig.text(.955,.893,"IAD  •  RSS + ЭЛИТИЗМ",color="#DDEBFC",
             ha="right",fontsize=10,zorder=2)
    fig.text(.059,.79,
             "Влияние параметра на качество решений для трёх классов размерности",
             color="#4B6787",fontsize=12)

    ax=fig.add_axes([.084,.25,.677,.49])
    ax.set_facecolor("#FFFFFF")
    ax.set_axisbelow(True)
    for _,left,right,color,_ in BANDS:
        ax.axvspan(left-.48,right+.48,color=color,alpha=.93,zorder=0)
    ax.grid(axis="y",color="#D6E2EF",linewidth=.8,alpha=.9)
    ax.set_yscale("symlog",linthresh=100,linscale=1.0,base=10)
    # Lines run across all sampled dimensions; the visual X spacing is
    # categorical and therefore its nonlinearity is declared in the caption.
    max_value=0.0
    for i,value in enumerate(_levels(parameter,mode)):
        selected={(r["class"],int(r["size"])):r for r in rows
                  if str(r["value"])==str(value)}
        x=[];y=[]
        for group,sizes in classes.items():
            for size in sizes:
                point=selected.get((group,size))
                if point is not None:
                    x.append(axis_position(group,size))
                    y.append(float(point["mean_error_percent"]))
        if not x: continue
        max_value=max(max_value,*y)
        ax.plot(x,y,color=COLORS[i],linewidth=2.45,alpha=.97,
                label=str(value),solid_capstyle="round",zorder=4)
        ax.scatter(x,y,color=COLORS[i],s=23,alpha=.36,zorder=5,
                   edgecolors="white",linewidths=.45)
    xtick_positions=[];tick_labels=[]
    for group,sizes in classes.items():
        for size in sizes:
            xtick_positions.append(axis_position(group,size))
            tick_labels.append(str(size))
    ax.set_xlim(-.54,15.54)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(tick_labels,rotation=0,fontsize=8.3)
    ax.tick_params(axis="x",pad=7,length=0,colors="#40546C")
    ax.tick_params(axis="y",pad=6,length=0,colors="#40546C")
    ticks=[0,20,50,100,300,1000,3000,5000,10000]
    ax.set_yticks([t for t in ticks if t<=max(10000,max_value*1.6)])
    ax.set_yticklabels([f"{t:,}".replace(","," ") for t in ticks if t<=max(10000,max_value*1.6)],
                       fontsize=9)
    ax.set_ylim(0,max(180,max_value*1.3))
    ax.set_ylabel("Средняя ошибка относительно оптимума, %",fontsize=10,
                  labelpad=12,color="#233E5A")
    for spine in ax.spines.values(): spine.set_visible(False)
    leg=ax.legend(title="Значение параметра",loc="upper left",ncol=2,
                  fontsize=9.5,title_fontsize=10,framealpha=.96,
                  facecolor="white",edgecolor="#D4DDE8",
                  borderpad=.75,handlelength=2.7)
    for t in leg.get_texts():t.set_color("#243B58")
    for group,left,right,clr,label in BANDS:
        ax.text((left+right)/2,-.155,label,
                ha="center",fontsize=10,fontweight="bold",
                color={"small":"#248673","medium":"#2464A0","large":"#B95D51"}[group],
                transform=ax.get_xaxis_transform())

    # Side explanation based on the ACTUAL plotted series, no invented trend.
    card=FancyBboxPatch((.785,.242),.188,.506,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        transform=fig.transFigure,facecolor="#F8FBFE",
        edgecolor="#D5E3F0",linewidth=1.3,zorder=1)
    fig.patches.append(card)
    fig.text(.808,.694,"КАК ЧИТАТЬ",color="#1C4B8D",
             fontsize=14,fontweight="bold",zorder=2)
    fig.text(.808,.626,"Ось X — размеры\nматриц всех классов:\n10–50, 100–500,\n1000–2000.",
             color="#2E435C",fontsize=10.5,linespacing=1.45,zorder=2)
    fig.text(.808,.474,"Ось Y — отклонение\nот точного решения.\nМеньше — лучше.",
             color="#2E435C",fontsize=10.5,linespacing=1.48,zorder=2)
    fig.text(.808,.354,"Каждая линия —\nодно значение\nпараметра.",
             color="#008C80",fontsize=11,fontweight="bold",
             linespacing=1.45,zorder=2)

    fig.text(.063,.124,
             f"Измерения: {sum(1 for r in rows if r['value']==str(_levels(parameter,mode)[0]))} размеров"
             f"  •  Повторений: {repeats}"
             f"  •  Исследуемый параметр: {unit}",
             fontsize=9.3,color="#385674")
    fig.text(.063,.084,
             "Размерности сгруппированы с равной шириной классов; шкала Y — нелинейная.",
             fontsize=9,color="#627B94")
    fig.text(.063,.055,
             "Точки и линии — реальные результаты. Никаких нарисованных или интерполированных измерений.",
             fontsize=8.6,color="#627B94")
    fig.text(.954,.06,"SMOKE — НЕ ДАННЫЕ 100–2000" if mode=="smoke" else "ГА • РЕАЛЬНЫЙ ЭКСПЕРИМЕНТ",
             fontsize=8.6,color="#9AAFC4",ha="right")
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/f"{parameter}.png",dpi=175)
    fig.savefig(output/f"{parameter}.svg")
    plt.close(fig)


def run_experiment(*, mode, parameters=None, output_dir="results/hyperparameter_curves_10_2000",
                   repeats=1, workers=2, seed=20261027,
                   parameter_source="results/scale_benchmark_full/best_parameters.json"):
    if mode not in ("full","smoke"):raise ValueError("mode must be full or smoke")
    if workers<2:raise ValueError("requires at least 2 workers for genuine island sweeps")
    if repeats<1:raise ValueError("repeats must be positive")
    parameters=tuple(HYPERPARAMETERS) if parameters is None else tuple(parameters)
    if not parameters or any(p not in HYPERPARAMETERS for p in parameters):
        raise ValueError("unknown or empty parameter list")
    records=json.loads(Path(parameter_source).read_text(encoding="utf-8"))
    if records["mode"]!="full":raise ValueError("requires full-sized tuned input")
    base=records["best_parameters"]
    sizes_by_class=REAL_SIZES if mode=="full" else SMOKE_SIZES
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    results=[]
    for parameter in parameters:
        values=_levels(parameter,mode)
        for class_index,(group,sizes) in enumerate(sizes_by_class.items()):
            for dim_index,size in enumerate(sizes):
                for repeat in range(repeats):
                    # Same matrix and GA seed across *all* hyperparameter values.
                    matrix_seed=seed+1000000*class_index+10007*dim_index+113*repeat
                    matrix=generate_matrix(size,matrix_seed)
                    optimal=exact_cost(matrix)
                    for value in values:
                        settings=_settings_for(group,parameter,base,mode)
                        settings[parameter]=value
                        # Carefully isolate ONLY the selected parameter.
                        ga_seed=matrix_seed+31
                        if parameter in PLATEAU_ONLY:
                            trial=run_plateau_variant(matrix,settings,ga_seed)
                            cost=trial["cost"]
                            elapsed=trial["seconds"]
                            generations=trial["generations"]
                        else:
                            measured=run_variant(matrix,settings,ga_seed,workers=workers)
                            cost=measured.cost
                            elapsed=measured.elapsed_seconds
                            generations=measured.generations
                        results.append({
                            "parameter":parameter,"class":group,"size":size,
                            "value":value,"repeat":repeat,"matrix_seed":matrix_seed,
                            "ga_seed":ga_seed,"num_islands":settings["num_islands"],
                            "total_population":settings["total_population"],
                            "cost":int(cost),"exact_cost":int(optimal),
                            "error_percent":100*(cost-optimal)/optimal,
                            "generations":int(generations),
                            "seconds":float(elapsed)
                        })
                print(f"{parameter}: {group} n={size}; {repeats} shared matrices",flush=True)
        _csv(output/"raw.csv",results)
        summary=aggregate(results)
        _csv(output/"summary.csv",summary)
        render_parameter(parameter,summary,output/"plots",mode=mode,repeats=repeats)
        (output/"metadata.json").write_text(
            json.dumps({
                "mode":mode,"seed":seed,"repeats":repeats,
                "requested_parameters":list(parameters),
                "completed_parameters":list(parameters[:parameters.index(parameter)+1]),
                "classes":{g:list(s) for g,s in sizes_by_class.items()},
                "x_axis":"categorical: equal-width small, medium, large class bands",
                "y_axis":"symmetric log for relative error percent, linear from 0 to 100%",
                "series":"one per changed hyperparameter value",
                "method":"per class tuned baseline, OFAT (all other values held fixed)",
                "reference":"scipy.optimize.linear_sum_assignment",
                "seeds":"paired among values of each parameter; independent across sizes/repeats",
                "plateau_experiments":"single GA only; early stop on population mean fitness",
                "island_experiments":"multiprocess islands; total_population is the sum over islands",
                "caveat":"best among observed settings only; not a global optimizer"
            },ensure_ascii=False,indent=2),encoding="utf-8"
        )
        print(f"Finished {parameter}: {len(results)} real trials total",flush=True)
    return {"mode":mode,"parameters":list(parameters),"rows":results}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("full","smoke"),default="smoke")
    p.add_argument("--output-dir",default="results/hyperparameter_curves_10_2000")
    p.add_argument("--repeats",type=int,default=1)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--seed",type=int,default=20261027)
    p.add_argument("--parameters",nargs="*",choices=tuple(HYPERPARAMETERS))
    o=p.parse_args()
    run_experiment(mode=o.mode,output_dir=o.output_dir,
                   repeats=o.repeats,workers=o.workers,seed=o.seed,
                   parameters=o.parameters)
if __name__=="__main__":main()
