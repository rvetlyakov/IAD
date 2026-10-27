"""Reproducible one-factor-at-a-time GA study for the assignment problem.

Run:
    python lab1/experiments.py --output-dir results --matrix-count 6 --repeats 2
    python lab1/experiments.py --smoke --output-dir /tmp/iad-ga-smoke
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

from lab1_test import run_ga
from scipy.optimize import linear_sum_assignment

# Two representative instance sizes per class; sizes are defined before trials.
SIZE_CLASSES = {
    "small": (3, 5),
    "medium": (8, 12),
    "large": (16, 24),
}

BASELINE = {
    "population_size": 100,
    "mutation_probability": 0.7,
    "mutation_percent": 45,
    "tournament_size": 2,  # unused compatibility parameter
    "max_iterations": 100,
    "plateau_limit": 5,
    "plateau_diff": 1e-4,  # relative mean-fitness change
    "elitist_percent": 0.2,
}

# Change exactly one active parameter at a time.
SWEEPS = {
    "population_size": (50, 100, 200, 300),
    "mutation_probability": (0.2, 0.5, 0.7, 1.0),
    "mutation_percent": (15, 30, 45, 60),
    "elitist_percent": (0.0, 0.1, 0.2, 0.3),
    "plateau_limit": (3, 5, 10, 20),
    "plateau_diff": (0.0, 1e-6, 1e-4, 1e-3),
    "max_iterations": (50, 100, 200, 400),
}

RAW_COLUMNS = (
    "parameter", "value", "class", "size", "matrix_id", "run_id", "seed",
    "relative_gap", "optimal", "generations", "elapsed_seconds"
)
SUMMARY_COLUMNS = (
    "parameter", "value", "class", "size", "runs", "mean_gap",
    "std_gap", "max_gap", "optimal_rate", "mean_generations",
    "mean_elapsed_seconds"
)


def generate_cases(matrix_count=6, seed=20261021, classes=None):
    """Create one immutable-in-practice set of positive cost matrices per size."""
    if matrix_count < 1:
        raise ValueError("matrix_count must be >= 1")
    classes = SIZE_CLASSES if classes is None else classes
    rng = random.Random(seed)
    cases = []
    for class_name, sizes in classes.items():
        for size in sizes:
            if size < 1:
                raise ValueError("matrix size must be >= 1")
            for matrix_id in range(matrix_count):
                matrix = [[rng.randint(1, 100) for _ in range(size)]
                          for _ in range(size)]
                cases.append((class_name, size, matrix_id, matrix))
    return cases


def parameter_settings(parameter, value, baseline=None):
    """A fresh parameter mapping differing from baseline in at most one key."""
    baseline = BASELINE if baseline is None else baseline
    if parameter not in SWEEPS:
        raise ValueError(f"unknown hyperparameter: {parameter}")
    settings = dict(baseline)
    settings[parameter] = value
    return settings


def run_study(matrix_count=6, repeats=2, seed=20261021,
              classes=None, sweeps=None, baseline=None):
    """Run paired instance/seed trials and return all individual measurements."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    cases = generate_cases(matrix_count, seed, classes)
    # Precompute exact references ONCE per matrix, outside all GA timers.
    references = {}
    for class_name, size, matrix_id, matrix in cases:
        rows, cols = linear_sum_assignment(matrix)
        references[(class_name, size, matrix_id)] = sum(
            matrix[int(i)][int(j)] for i, j in zip(rows, cols))
    sweeps = SWEEPS if sweeps is None else sweeps
    baseline = BASELINE if baseline is None else baseline
    rows = []
    for parameter, values in sweeps.items():
        for value in values:
            settings = parameter_settings(parameter, value, baseline)
            for class_name, size, matrix_id, matrix in cases:
                for run_id in range(repeats):
                    # Identical initialization RNG for every hyperparameter
                    # on a given matrix/repetition (paired comparison).
                    ga_seed = seed + size * 100003 + matrix_id * 1009 + run_id * 17
                    random.seed(ga_seed)
                    start = time.perf_counter()
                    optimal, generations, relative_gap = run_ga(
                        **settings, matrix=matrix,
                        reference_cost=references[(class_name, size, matrix_id)]
                    )
                    elapsed = time.perf_counter() - start
                    if relative_gap < -1e-10:
                        raise AssertionError("GA better than exact minimum")
                    rows.append({
                        "parameter": parameter,
                        "value": str(value),
                        "class": class_name,
                        "size": size,
                        "matrix_id": matrix_id,
                        "run_id": run_id,
                        "seed": ga_seed,
                        "relative_gap": relative_gap,
                        "optimal": int(optimal),
                        "generations": generations,
                        # Only the GA; the exact SciPy reference was cached.
                        "elapsed_seconds": elapsed,
                    })
    return rows


def summarize(rows):
    """Return summary metrics by size and pooled class (size='all')."""
    groups = defaultdict(list)
    for r in rows:
        key = (r["parameter"], r["value"], r["class"])
        groups[key + (str(r["size"]),)].append(r)
        groups[key + ("all",)].append(r)

    summary = []
    for (param, value, name, size), observations in groups.items():
        gaps = [float(x["relative_gap"]) for x in observations]
        summary.append({
            "parameter": param, "value": value, "class": name, "size": size,
            "runs": len(gaps),
            "mean_gap": statistics.mean(gaps),
            "std_gap": statistics.stdev(gaps) if len(gaps) > 1 else 0.0,
            "max_gap": max(gaps),
            "optimal_rate": statistics.mean(int(x["optimal"]) for x in observations),
            "mean_generations": statistics.mean(int(x["generations"]) for x in observations),
            "mean_elapsed_seconds": statistics.mean(float(x["elapsed_seconds"])
                                                      for x in observations),
        })
    return sorted(summary, key=lambda x: (x["parameter"], x["class"],
                                          str(x["size"]), x["value"]))


def write_csv(path, data, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(data)


def plot_summary(summary, output_dir, sweeps=None, classes=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweeps = SWEEPS if sweeps is None else sweeps
    classes = SIZE_CLASSES if classes is None else classes
    figures = output_dir / "plots"
    figures.mkdir(exist_ok=True, parents=True)
    measures = (
        ("mean_gap", "Mean relative gap (%)", 100),
        ("optimal_rate", "Optimal solutions (%)", 100),
        ("mean_elapsed_seconds", "Mean end-to-end time (s)", 1),
    )
    for parameter, levels in sweeps.items():
        labels = [str(x) for x in levels]
        for field, label, scale in measures:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for name in classes:
                lookup = {
                    row["value"]: row for row in summary
                    if row["parameter"] == parameter and row["class"] == name
                    and row["size"] == "all"
                }
                ax.plot(labels, [lookup[x][field] * scale for x in labels],
                        marker="o", label=name)
            ax.set_xlabel(parameter)
            ax.set_ylabel(label)
            ax.set_title(f"{parameter}: {label}")
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(figures / f"{parameter}_{field}.png", dpi=130)
            plt.close(fig)


def save_study(output_dir, matrix_count=6, repeats=2, seed=20261021,
               classes=None, sweeps=None, baseline=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = SIZE_CLASSES if classes is None else classes
    sweeps = SWEEPS if sweeps is None else sweeps
    baseline = BASELINE if baseline is None else baseline
    rows = run_study(matrix_count, repeats, seed, classes, sweeps, baseline)
    summary = summarize(rows)
    write_csv(output_dir / "raw_results.csv", rows, RAW_COLUMNS)
    write_csv(output_dir / "summary.csv", summary, SUMMARY_COLUMNS)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump({
            "seed": seed, "matrix_count": matrix_count, "repeats": repeats,
            "classes": classes, "baseline": baseline, "sweeps": sweeps,
            "raw_trials": len(rows),
            "timing": "only GA evolution; each SciPy exact reference computed once outside timers",
            "fitness_plateau": "relative mean-fitness change",
            "sampling": "same random matrices and GA seeds across parameter levels",
        }, f, ensure_ascii=False, indent=2)
    plot_summary(summary, output_dir, sweeps, classes)
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--matrix-count", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        classes = {"small": (3,), "medium": (8,), "large": (16,)}
        sweeps = {"population_size": (10, 20)}
        args.matrix_count, args.repeats = 1, 1
    else:
        classes, sweeps = None, None
    rows, summary = save_study(args.output_dir, args.matrix_count,
                               args.repeats, args.seed, classes, sweeps)
    print(f"Generated {len(rows)} trials, {len(summary)} summary rows "
          f"under {args.output_dir}")


if __name__ == "__main__":
    main()
