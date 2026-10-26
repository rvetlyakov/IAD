"""Slide-ready measured curves for single GA vs multiprocess islands.

Every x position is independently evaluated. Colored curves use a centered
three-measurement moving average for visual readability, while faint markers
show actual unsmoothed measurements. There are NO invented intermediate data.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from scale_benchmark import evaluate_pair, exact_cost, generate_matrix

GRIDS = {
    "small": tuple(range(10, 51, 4)),
    "medium": tuple(range(100, 501, 40)),
    "large": tuple(range(1000, 2001, 100)),
}
SMOKE_GRIDS = {
    "small": (4, 5),
    "medium": (7, 8),
    "large": (10, 11),
}

TITLES = {
    "small": "МАЛАЯ РАЗМЕРНОСТЬ",
    "medium": "СРЕДНЯЯ РАЗМЕРНОСТЬ",
    "large": "БОЛЬШАЯ РАЗМЕРНОСТЬ",
}
PALETTE = {"single": "#2563EB", "islands": "#00B5AB"}
NAMES = {"single": "Обычный ГА", "islands": "Островной ГА"}


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_curves(parameters_path: Path | str,
                   output_dir: Path | str,
                   mode: str = "smoke", repeats: int = 2,
                   workers: int = 2, seed: int = 20261027):
    if mode not in ("smoke", "full"):
        raise ValueError("mode must be 'smoke' or 'full'")
    if workers < 2:
        raise ValueError("workers must be at least 2 for parallel island plots")
    if repeats < 1:
        raise ValueError("repeats must be positive")

    document = json.loads(Path(parameters_path).read_text(encoding="utf-8"))
    if document["mode"] != "full":
        raise ValueError("curve study needs trained parameters from REAL full-size study")
    settings = document["best_parameters"]
    sizes_by_class = GRIDS if mode == "full" else SMOKE_GRIDS
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for class_index, (class_name, sizes) in enumerate(sizes_by_class.items()):
        single = settings[class_name]["single"]
        islands = settings[class_name]["islands"]
        if single["num_islands"] != 1 or islands["num_islands"] < 2:
            raise ValueError("invalid saved island configuration")
        for size_index, size in enumerate(sizes):
            for rep in range(repeats):
                matrix_seed = seed + class_index * 1000000 + size_index * 7919 + rep * 113
                matrix = generate_matrix(size, matrix_seed)
                target = exact_cost(matrix)
                rows = evaluate_pair(
                    matrix, single, islands, seed=matrix_seed + 17,
                    workers=workers, reference=target, matrix_seed=matrix_seed,
                )
                for row in rows:
                    row["class"] = class_name
                    row["repeat"] = rep
                all_rows.extend(rows)
            print(f"{class_name}: measured {size}x{size}; {repeats} paired cases", flush=True)
            _write_csv(output / "raw.csv", all_rows)
            _write_csv(output / "summary.csv", build_summary(all_rows))
        plot_curves(build_summary(all_rows), output / "plots", mode=mode,
                    only_class=class_name, repeats=repeats)
        (output / "metadata.json").write_text(
            json.dumps({
                "mode": mode, "seed": seed, "repeats_per_size": repeats,
                "size_grids": {k:list(v) for k,v in sizes_by_class.items()},
                "sampled_dimensions": "all x coordinates measured, no interpolation",
                "visual_smoothing": "centered 3-point moving average (presentation only)",
                "comparison": "matched cost matrices, seeds, total population, generations",
                "parallel_workers": workers,
                "source_parameters": str(parameters_path),
                "status": "complete" if class_name == list(sizes_by_class)[-1] else "partial",
            }, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return all_rows


def build_summary(raw):
    grouped={}
    for row in raw:
        key=(row["class"],int(row["size"]),row["variant"])
        grouped.setdefault(key,[]).append(row)
    summary=[]
    for (name,size,variant), rows in sorted(grouped.items()):
        summary.append({
            "class": name, "size": size, "variant": variant,
            "runs": len(rows),
            "mean_relative_gap": float(mean(float(x["relative_gap"]) for x in rows)),
            "std_relative_gap": float(stdev(float(x["relative_gap"]) for x in rows)) if len(rows)>1 else 0.0,
            "mean_seconds": float(mean(float(x["elapsed_seconds"]) for x in rows)),
            "total_population": int(rows[0]["total_population"]),
            "generations": int(rows[0]["generations"]),
        })
    return summary


def centered_running_mean(values):
    """Presentation-only 3-point visual smoothing."""
    return [mean(values[max(0,k-1):min(len(values),k+2)])
            for k in range(len(values))]


def plot_curves(summary, output_dir, *, mode="full", only_class=None, repeats=2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    plt.rcParams.update({
        "font.family": "Montserrat",
        "font.size": 11,
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "savefig.facecolor": "#F6F8FC",
    })
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    class_names = [only_class] if only_class else sorted(set(r["class"] for r in summary))
    for name in class_names:
        rows = [r for r in summary if r["class"]==name]
        if not rows:
            continue
        for field, metric_title, y_title, factor, suffix in (
            ("mean_relative_gap", "КАЧЕСТВО РЕШЕНИЯ", "Отклонение от точного оптимума, %", 100, "%"),
            ("mean_seconds", "ВРЕМЯ ВЫПОЛНЕНИЯ", "Продолжительность, с", 1, "с"),
        ):
            fig = plt.figure(figsize=(13.333,7.5), dpi=160, facecolor="#F6F8FC")
            header = FancyBboxPatch((.035,.83),.93,.135,
                          transform=fig.transFigure,
                          boxstyle="round,pad=0.004,rounding_size=0.018",
                          facecolor="#132A44",edgecolor="none",zorder=1)
            fig.patches.append(header)
            fig.text(.062,.925,metric_title,color="#72D6D3",fontsize=10,
                     fontweight="bold",va="center",zorder=2)
            fig.text(.062,.865,TITLES[name],color="white",fontsize=24,
                     fontweight="bold",va="center",zorder=2)
            fig.text(.922,.875,"IAD  /  GA",color="#A9C6D7",
                     fontsize=10,ha="right",va="center",zorder=2)

            ax = fig.add_axes((.10,.27,.84,.50))
            ax.set_facecolor("white")
            ax.set_axisbelow(True)
            ax.grid(axis="y",color="#E1E9F2",linewidth=1.2)
            ax.grid(axis="x",color="#F0F3F8",linewidth=.8)
            maximum=0.0
            samples=[]
            for variant in ("single","islands"):
                group=sorted((r for r in rows if r["variant"]==variant),
                             key=lambda r:r["size"])
                if not group:
                    continue
                xx=[int(x["size"]) for x in group]
                yy=[float(x[field])*factor for x in group]
                smooth=centered_running_mean(yy) if len(yy)>=3 else yy
                color=PALETTE[variant]
                ax.plot(xx,smooth,color=color,lw=3.4,solid_capstyle="round",
                        label=NAMES[variant],zorder=4)
                # Every faint dot is an observed per-size mean, NOT an interpolation.
                ax.scatter(xx,yy,s=28,facecolor=color,edgecolors="white",
                           linewidths=.6,alpha=.42,zorder=5)
                maximum=max(maximum,*yy,*smooth)
                samples=xx
            ax.set_ylim(0,max(1e-5,maximum*1.15))
            if samples:
                ax.set_xlim(samples[0],samples[-1])
                ax.set_xticks(samples[::2]+([] if len(samples)%2 else [samples[-1]]))
            ax.set_xlabel("Размерность матрицы  n × n", color="#253D5B",
                          labelpad=14, fontsize=12)
            ax.set_ylabel(y_title,color="#253D5B",labelpad=13,fontsize=11)
            ax.tick_params(colors="#526782",labelsize=10,length=0,pad=8)
            for edge in ("top","right","left","bottom"):
                ax.spines[edge].set_visible(False)
            lg=ax.legend(loc="upper left",ncol=2,frameon=True,
                         facecolor="white",edgecolor="#E0EAF5",
                         fontsize=11,borderpad=.85,labelspacing=.85,
                         handlelength=2.6)
            for text in lg.get_texts():
                text.set_color("#233D58")
            fig.text(.062,.124,
                    f"Общая популяция: {int(rows[0]['total_population'])}   •   "
                    f"Поколений: {int(rows[0]['generations'])}   •   "
                    f"Запусков на размерность: {repeats}",
                    color="#29425E",fontsize=10,weight="medium")
            fig.text(.062,.071,
                    "Точки — измеренные значения; линии — скользящее среднее по 3 соседним размерностям",
                    color="#697D94",fontsize=9.5)
            fig.text(.94,.071,"SMOKE" if mode=="smoke" else "РЕАЛЬНЫЕ ИЗМЕРЕНИЯ",
                     color="#8090A2",fontsize=9,ha="right",weight="bold")
            base=out/f"{name}_{field}"
            fig.savefig(base.with_suffix(".png"),dpi=160)
            fig.savefig(base.with_suffix(".svg"))
            plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parameters",default="results/scale_benchmark_full/best_parameters.json")
    p.add_argument("--output",default="results/scale_curves_100_2000")
    p.add_argument("--mode",choices=("smoke","full"),default="smoke")
    p.add_argument("--repeats",type=int,default=2)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--seed",type=int,default=20261027)
    opts=p.parse_args()
    collect_curves(opts.parameters,opts.output,opts.mode,opts.repeats,
                   opts.workers,opts.seed)


if __name__=="__main__":
    main()
