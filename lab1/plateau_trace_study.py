"""Scientific plateau calibration using replay of full GA trajectories.

Exactly one full (no early stop) GA trajectory per matrix/seed. Every
threshold/plateau-window counterfactual is replayed from its exact recorded
mean-fitness history and best-so-far costs. Changing plateau parameters has
NO effect on the random evolution *before* their stopping condition; replay
is therefore identical to rerunning with the same initial RNG seed.

Real results: 15 matrix sizes, 3 independent samples per size, all values
plotted on every chart, preserving the original inversion + RSS + elitism.

  python lab1/plateau_trace_study.py --mode smoke
  python lab1/plateau_trace_study.py --mode full --repeats 3
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

from lab1 import Employee, Employees
from scale_benchmark import generate_matrix, exact_cost, relative_gap
from hyperparameter_curves import REAL_SIZES, SMOKE_SIZES, axis_position, BANDS
from termination import plateau_reached

PLATEAU_THRESHOLDS = (0.0, 1e-4, 3e-4, 7e-4, 0.0015, 0.003, 0.005)
PLATEAU_LIMITS = (3, 5, 10, 20)
SMOKE_THRESHOLDS = (0.0, .001, .1)
SMOKE_LIMITS = (2, 3)
COLORS = ("#1761B0", "#008E81", "#E07B34", "#933D9C",
          "#C54A52", "#6478C3", "#AA793C")


def trace_evolution(matrix, settings, seed):
    """Return generation 0..max_generations of identical production GA steps."""
    n = len(matrix)
    total = int(settings["total_population"])
    rng = random.Random(seed)
    initial = [rng.sample(range(n), n) for _ in range(total)]
    random.seed(seed + 1)
    ga = Employees.__new__(Employees)
    ga.matrix = matrix
    ga.n = total
    ga.mutation_probability = float(settings["mutation_probability"])
    ga.mutation_percent = float(settings["mutation_percent"])
    ga.elitist_percent = float(settings["elitist_percent"])
    ga.t = 2
    ga.population = [
        Employee(x, n, matrix, ga.mutation_percent) for x in initial
    ]
    best_cost = int(ga.get_best()[1])
    records = [{
        "generation": 0,
        "mean_fitness": float(ga.fitness()),
        "best_cost": best_cost,
        "elapsed_seconds": 0.0,
    }]
    started = time.perf_counter()
    for generation in range(1, int(settings["max_generations"]) + 1):
        ga.mutate()
        best_cost = min(best_cost, int(ga.get_best()[1]))
        ga.select()
        best_cost = min(best_cost, int(ga.get_best()[1]))
        records.append({
            "generation": generation,
            "mean_fitness": float(ga.fitness()),
            "best_cost": best_cost,
            "elapsed_seconds": float(time.perf_counter() - started),
        })
    return records


def replay_plateau(trajectory, limit, threshold):
    """Exact stop on saved history, without consuming any extra RNG."""
    if limit < 1 or threshold < 0:
        raise ValueError("Invalid plateau control")
    history = []
    for record in trajectory:
        history.append(float(record["mean_fitness"]))
        if plateau_reached(history, limit, threshold):
            return {
                "cost": int(record["best_cost"]),
                "generations": int(record["generation"]),
                "seconds": float(record["elapsed_seconds"]),
                "plateau_triggered": True,
            }
    last = trajectory[-1]
    return {
        "cost": int(last["best_cost"]),
        "generations": int(last["generation"]),
        "seconds": float(last["elapsed_seconds"]),
        "plateau_triggered": False,
    }


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["parameter"], row["class"],
                 int(row["size"]), str(row["value"]))].append(row)
    output = []
    for (parameter, cls, size, value), items in sorted(grouped.items()):
        errors = [float(row["error_percent"]) for row in items]
        output.append({
            "parameter": parameter, "class": cls, "size": size,
            "value": value, "repeats": len(items),
            "mean_error_percent": statistics.mean(errors),
            "std_error_percent": statistics.stdev(errors) if len(errors)>1 else 0,
            "mean_generations": statistics.mean(int(row["generations"])
                                                 for row in items),
            "plateau_activation_rate": statistics.mean(
                int(row["plateau_triggered"]) for row in items),
        })
    return output


def render_chart(rows, parameter, levels, field, path, mode):
    """Presentation-neutral line chart: ONLY lines, axes, bands and legend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "Montserrat", "font.size": 12,
        "axes.unicode_minus": False, "svg.fonttype": "none",
    })
    sizes = REAL_SIZES if mode == "full" else SMOKE_SIZES
    fig, ax = plt.subplots(figsize=(12.8, 6.65), dpi=160, facecolor="white")
    fig.subplots_adjust(left=.095, right=.985, top=.945, bottom=.275)
    for _, left, right, color, label in BANDS:
        ax.axvspan(left-.5, right+.5, color=color, alpha=.90, zorder=0)
    ax.grid(axis="y", color="#DCE5ED", alpha=.85)
    ax.set_axisbelow(True)
    max_value = 0.
    for i, value in enumerate(levels):
        selected = {(r["class"], int(r["size"])):r for r in rows
                    if r["parameter"] == parameter
                    and float(r["value"]) == float(value)}
        xx, yy = [], []
        for cls, cls_sizes in sizes.items():
            for size in cls_sizes:
                point = selected.get((cls, size))
                if point is not None:
                    xx.append(axis_position(cls, size))
                    yy.append(float(point[field]))
        if not yy:
            continue
        max_value = max(max_value,*yy)
        ax.plot(xx,yy,color=COLORS[i],lw=2.5,alpha=.96,
                label=str(value),solid_capstyle="round",zorder=3)
        ax.scatter(xx,yy,s=25,color=COLORS[i],alpha=.5,
                   edgecolors="white",lw=.5,zorder=4)
    axis_x = [axis_position(cls,size) for cls,values in sizes.items()
              for size in values]
    axis_names = [str(size) for values in sizes.values() for size in values]
    ax.set_xlim(-.55,15.55)
    ax.set_xticks(axis_x)
    ax.set_xticklabels(axis_names,fontsize=9)
    ax.tick_params(axis="both",length=0,pad=8,colors="#40546C")
    if field == "mean_error_percent":
        ax.set_yscale("symlog",linthresh=100,linscale=1,base=10)
        ax.set_ylim(0,max(180,max_value*1.2))
        ax.set_ylabel("Отклонение от оптимума, % · symlog",
                      fontsize=12,color="#243D59",labelpad=12)
    else:
        ax.set_ylim(0,max(5,max_value*1.2))
        ax.set_ylabel("Поколений до остановки",fontsize=12,
                      color="#243D59",labelpad=12)
    ax.set_xlabel("Размерность матрицы n × n",
                  fontsize=12,color="#243D59",labelpad=48)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.legend(title="Значение параметра",ncol=3,loc="upper left",
              fontsize=10,title_fontsize=10,framealpha=.96,
              facecolor="white",edgecolor="#CAD6E4")
    for cls,left,right,_,label in BANDS:
        ax.text((left+right)/2,-.16,label,transform=ax.get_xaxis_transform(),
                ha="center",va="top",fontsize=11,fontweight="bold",
                color={"small":"#248673","medium":"#2464A0",
                       "large":"#B95D51"}[cls])
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path.with_suffix(".png"),dpi=180,facecolor="white")
    fig.savefig(path.with_suffix(".svg"),facecolor="white")
    plt.close(fig)


def run_trace_study(*, output_dir="results/plateau_trace_20261027",
                    mode="smoke", seed=20261030, repeats=3,
                    source_params="results/scale_benchmark_full/best_parameters.json"):
    if mode not in ("smoke","full") or repeats<1:
        raise ValueError("Invalid experiment mode or repeats")
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    source=json.loads(Path(source_params).read_text(encoding="utf-8"))
    if source["mode"]!="full":
        raise ValueError("Requires actual full-size tuned baseline")
    reference=source["best_parameters"]
    size_classes=REAL_SIZES if mode=="full" else SMOKE_SIZES
    diff_levels=PLATEAU_THRESHOLDS if mode=="full" else SMOKE_THRESHOLDS
    limit_levels=PLATEAU_LIMITS if mode=="full" else SMOKE_LIMITS
    raw=[]
    trace_rows=[]
    for class_index,(cls,sizes) in enumerate(size_classes.items()):
        settings=dict(reference[cls]["single"])
        settings.update(num_islands=1)
        if mode=="smoke":
            settings.update(total_population=12,max_generations=8,
                            mutation_probability=.7, mutation_percent=40)
        for size_index,size in enumerate(sizes):
            for repeat in range(repeats):
                matrix_seed=seed+1000000*class_index+10007*size_index+113*repeat
                ga_seed=matrix_seed+31
                matrix=generate_matrix(size,matrix_seed)
                optimal=exact_cost(matrix)
                trace=trace_evolution(matrix,settings,ga_seed)
                source_id=f"{cls}:{size}:{matrix_seed}:{ga_seed}"
                for state in trace:
                    trace_rows.append({
                        "trajectory_id":source_id, "class":cls, "size":size,
                        "repeat":repeat, "matrix_seed":matrix_seed,
                        "ga_seed":ga_seed, **state
                    })
                for param,levels in (("plateau_limit",limit_levels),
                                     ("plateau_diff",diff_levels)):
                    for value in levels:
                        trial_config=dict(settings,
                               plateau_limit=int(value) if param=="plateau_limit" else 5,
                               plateau_diff=.001 if param=="plateau_limit" else float(value))
                        stopped=replay_plateau(trace,
                                    trial_config["plateau_limit"],
                                    trial_config["plateau_diff"])
                        raw.append({
                            "parameter":param, "class":cls,"size":size,
                            "value":value,"repeat":repeat,
                            "trajectory_id":source_id,
                            "matrix_seed":matrix_seed,"ga_seed":ga_seed,
                            "exact_cost":optimal,
                            "ga_cost":stopped["cost"],
                            "error_percent":100*relative_gap(stopped["cost"],optimal),
                            "generations":stopped["generations"],
                            "elapsed_seconds":stopped["seconds"],
                            "plateau_triggered":stopped["plateau_triggered"],
                            "settings_json":json.dumps(trial_config,
                                                       sort_keys=True),
                        })
            print(f"Calibrated {cls}: n={size} ({repeats} full trajectories)",flush=True)
            save_csv(output/"trajectories.csv",trace_rows)
            save_csv(output/"raw.csv",raw)
            save_csv(output/"summary.csv",aggregate(raw))
    summary=aggregate(raw)
    for param,levels in (("plateau_limit",limit_levels),("plateau_diff",diff_levels)):
        for field,suffix in (("mean_error_percent","error"),
                             ("mean_generations","generations")):
            render_chart(summary,param,levels,field,
                         output/"plots"/f"{param}_{suffix}",mode)
    meta={
        "source":"full-run trajectory replay",
        "mode":mode,"seed":seed,"repeats":repeats,
        "trajectories":len(trace_rows)//(max(int(reference[c]["single"]["max_generations"])
                     for c in reference)+1) if mode=="full" else
                     len(size_classes)*len(next(iter(size_classes.values())))*repeats,
        "size_classes":{cls:list(s) for cls,s in size_classes.items()},
        "plateau_diff_values":list(diff_levels),
        "plateau_limit_values":list(limit_levels),
        "plateau_limit_fixed_diff":.001,
        "plateau_diff_fixed_limit":5,
        "fairness":"each parameter value replayed from identical full RNG trajectory",
        "notes":"Numbers are replayed counterfactual stop points, NOT independent GA runs per level"
    }
    (output/"metadata.json").write_text(
        json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    return {"mode":mode,"rows":raw,"summary":summary,"metadata":meta}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("smoke","full"),default="smoke")
    p.add_argument("--output-dir",default="results/plateau_trace_20261027")
    p.add_argument("--seed",type=int,default=20261030)
    p.add_argument("--repeats",type=int,default=3)
    args=p.parse_args()
    run_trace_study(mode=args.mode, output_dir=args.output_dir,
                    seed=args.seed,repeats=args.repeats)


if __name__=="__main__":
    main()
