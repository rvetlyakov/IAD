"""Reproducible size-stratified GA tuning and *real* multiprocess island comparison.

Small/medium/large refer to assignment matrix dimensions, not population sizes.
Full mode benchmarks 100..500 and 1000..2000. Smoke mode tests machinery
only: its synthetic tiny cases MUST NOT be reported as large-scale results.

Run from repository root:
  python lab1/scale_benchmark.py --mode smoke
  python lab1/scale_benchmark.py --mode full --workers 4
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from lab1 import Employee, Employees

SIZE_CLASSES = {
    "small": (10, 30, 50),
    "medium": (100, 300, 500),
    "large": (1000, 1500, 2000),
}
TRAIN_SIZES = {"small": 30, "medium": 300, "large": 1500}
VALIDATE_SIZES = {
    "small": (10, 20, 30, 40, 50),
    "medium": (100, 200, 300, 400, 500),
    "large": (1000, 1250, 1500, 1750, 2000),
}
SMOKE_TRAIN = {"small": 4, "medium": 7, "large": 10}
SMOKE_VALIDATE = {
    "small": (5,),
    "medium": (8,),
    "large": (12,),
}
BASELINE = {
    "total_population": 80,
    "mutation_probability": 0.7,
    "mutation_percent": 45,
    "elitist_percent": 0.2,
    "max_generations": 25,
    "num_islands": 1,
    "migration_interval": 10,
    "migration_percent": 10,
}
COMMON_SWEEPS = {
    "total_population": (40, 80, 120),
    "mutation_probability": (0.4, 0.7, 1.0),
    "mutation_percent": (20, 45, 70),
    "elitist_percent": (0.0, 0.1, 0.2),
    "max_generations": (15, 25, 40),
}
ISLAND_SWEEPS = {
    "num_islands": (2, 3, 4),
    "migration_interval": (5, 10, 20),
    "migration_percent": (5, 10, 20),
}
_MATRIX = None


def generate_matrix(size: int, seed: int) -> np.ndarray:
    """Dense strictly-positive costs, memory-bounded int32 (avoid uint8 sum overflow)."""
    if size < 2:
        raise ValueError("matrix size must be >=2")
    return np.random.default_rng(seed).integers(
        1, 101, size=(size, size), dtype=np.int32
    )


def split_population(total: int, islands: int) -> list[int]:
    if islands < 1 or total < islands:
        raise ValueError("need at least one individual per island")
    quotient, remainder = divmod(total, islands)
    return [quotient + (index < remainder) for index in range(islands)]


def assignment_cost(matrix, vector) -> int:
    rows = np.arange(len(vector))
    return int(np.sum(matrix[rows, vector], dtype=np.int64))


def exact_cost(matrix: np.ndarray) -> int:
    rows, cols = linear_sum_assignment(matrix)
    return int(np.sum(matrix[rows, cols], dtype=np.int64))


def _install_matrix(matrix) -> None:
    global _MATRIX
    _MATRIX = matrix


def _evolve_island(task):
    """Worker: local stochastic evolution, no shared mutable population."""
    vectors, size, settings, steps, seed = task
    if _MATRIX is None:
        raise RuntimeError("worker matrix not installed")
    random.seed(seed)
    if vectors is None:
        ga = Employees(
            n=size, mutation_probability=settings["mutation_probability"],
            matrix=_MATRIX, mutation_percent=settings["mutation_percent"],
            elitist_percent=settings["elitist_percent"]
        )
    else:
        ga = Employees.__new__(Employees)
        ga.matrix = _MATRIX
        ga.n = size
        ga.mutation_probability = settings["mutation_probability"]
        ga.mutation_percent = settings["mutation_percent"]
        ga.elitist_percent = settings["elitist_percent"]
        ga.t = 2  # legacy unused argument
        ga.population = [
            Employee(vector, len(_MATRIX), _MATRIX, ga.mutation_percent)
            for vector in vectors
        ]
        if len(ga.population) != ga.n:
            # Migration immediately triggers selection, as in the original island GA.
            ga.select()
    best_cost = int(ga.get_best()[1])
    for _ in range(steps):
        ga.mutate()
        # Remember improvements even if a new mutation loses the RSS lottery.
        best_cost = min(best_cost, int(ga.get_best()[1]))
        ga.select()
        best_cost = min(best_cost, int(ga.get_best()[1]))
    return [v.vector for v in ga.population], best_cost


def _migrate(states, sizes, matrix, percent):
    if percent == 0:
        return states
    donors = []
    for state, size in zip(states, sizes):
        count = min(size, math.ceil(size * percent / 100))
        donors.append(sorted(state, key=lambda x: assignment_cost(matrix, x))[:count])
    result = [list(state) for state in states]
    for index, migrants in enumerate(donors):
        receiver = (index + 1) % len(states)
        result[receiver].extend(list(v) for v in migrants)
    return result


@dataclass
class RunResult:
    cost: int
    elapsed_seconds: float
    total_population: int
    generations: int
    workers: int
    parallel: bool


def run_variant(matrix, settings: dict, seed: int, workers: int = 2) -> RunResult:
    """Compare at equal sum of island sizes and equal number of generations.

    With >1 islands and >1 workers, independent islands actually evolve in
    separate OS processes. Pool creation and migration count in wall time.
    No plateau stopping in this benchmark: both variants get the same fixed
    generation budget. Plateau is researched separately in legacy experiments.
    """
    total = int(settings["total_population"])
    islands = int(settings["num_islands"])
    generations = int(settings["max_generations"])
    interval = int(settings["migration_interval"])
    percent = float(settings["migration_percent"])
    if generations < 1 or interval < 1 or workers < 1:
        raise ValueError("generations, migration interval and workers must be positive")
    if not 0 <= percent <= 100:
        raise ValueError("migration_percent must be within [0, 100]")
    sizes = split_population(total, islands)
    actual_workers = min(workers, islands)
    matrix = np.asarray(matrix, dtype=np.int32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("expected square cost matrix")

    started = time.perf_counter()
    # Identical N initial chromosomes for the ordinary and island variants.
    initial_rng = random.Random(seed)
    initial = [initial_rng.sample(range(len(matrix)), len(matrix))
               for _ in range(total)]
    states = []
    cursor = 0
    for size in sizes:
        states.append(initial[cursor:cursor + size])
        cursor += size
    best_seen = min(assignment_cost(matrix, chromosome)
                    for chromosome in initial)
    if islands == 1:
        _install_matrix(matrix)
        states[0], epoch_best = _evolve_island(
            (states[0], sizes[0], settings, generations, seed + 1)
        )
        best_seen = min(best_seen, epoch_best)
    else:
        # Persistent processes, matrix transferred once in pool initializer.
        with ProcessPoolExecutor(
            max_workers=actual_workers,
            initializer=_install_matrix,
            initargs=(matrix,),
        ) as executor:
            offset = 0
            epoch = 0
            while offset < generations:
                steps = min(interval, generations - offset)
                tasks = [
                    (states[i], sizes[i], settings, steps,
                     seed + 1009 * i + 100003 * epoch)
                    for i in range(islands)
                ]
                outcomes = list(executor.map(_evolve_island, tasks))
                states = [population for population, _ in outcomes]
                best_seen = min(best_seen, *(best for _, best in outcomes))
                offset += steps
                if offset < generations:
                    states = _migrate(states, sizes, matrix, percent)
                epoch += 1
    cost = min(best_seen, *(assignment_cost(matrix, chromosome)
                for state in states for chromosome in state))
    elapsed = time.perf_counter() - started
    return RunResult(int(cost), float(elapsed), int(total), int(generations),
                     int(actual_workers), bool(islands > 1 and actual_workers > 1))


def evaluate_pair(matrix, common_settings: dict, island_settings: dict,
                  seed: int, workers: int = 2, *, reference: int | None = None,
                  matrix_seed: int | None = None) -> list[dict]:
    """Holdout comparison with identical matrix, total N, generations and GA ops."""
    single = dict(common_settings, num_islands=1)
    islands = dict(common_settings, **{
        name: island_settings[name] for name in
        ("num_islands", "migration_interval", "migration_percent")
    })
    if islands["num_islands"] < 2:
        raise ValueError("island comparator requires at least 2 islands")
    for key in ("total_population", "mutation_probability", "mutation_percent",
                "elitist_percent", "max_generations"):
        if single[key] != islands[key]:
            raise ValueError(f"unfair comparison: {key} differs")
    exact = exact_cost(matrix) if reference is None else reference
    output = []
    for variant, params in (("single", single), ("islands", islands)):
        result = run_variant(matrix, params, seed=seed, workers=workers)
        output.append({
            "variant": variant,
            "matrix_seed": seed if matrix_seed is None else matrix_seed,
            "seed": seed,
            "size": len(matrix),
            "num_islands": params["num_islands"],
            "total_population": result.total_population,
            "generations": result.generations,
            "exact_cost": exact,
            "ga_cost": result.cost,
            "relative_gap": float((result.cost - exact) / exact),
            "elapsed_seconds": result.elapsed_seconds,
            "workers": result.workers,
            "parallel": result.parallel,
        })
    return output


def _settings_values(mode):
    if mode == "smoke":
        return {
            "mutation_probability": (0.7, 1.0),
        }, {
            "num_islands": (2, 3),
        }
    return COMMON_SWEEPS, ISLAND_SWEEPS


def _mean_gap(matrix, reference, params, seed, workers, repeats):
    gaps = []
    runtimes = []
    for run_index in range(repeats):
        result = run_variant(matrix, params, seed=seed + 37 * run_index,
                             workers=workers)
        gaps.append((result.cost - reference) / reference)
        runtimes.append(result.elapsed_seconds)
    return statistics.mean(gaps), statistics.mean(runtimes)


def _tune_for_class(class_name, matrix, reference, *, mode, seed, workers,
                    repeats):
    """Greedy sequential OFAT; optimum means best OBSERVED, not global optimum."""
    sweeps_common, sweeps_island = _settings_values(mode)
    common = dict(BASELINE)
    if mode == "smoke":
        common.update(total_population=12, max_generations=4,
                      migration_interval=2, mutation_percent=45)
    measurements = []
    for key, values in sweeps_common.items():
        trials = []
        for value in values:
            candidate = dict(common, **{key: value}, num_islands=1)
            gap, seconds = _mean_gap(matrix, reference, candidate, seed,
                                     workers, repeats)
            trials.append((gap, seconds, value))
            measurements.append({
                "class": class_name, "variant": "single", "parameter": key,
                "value": value, "mean_gap": gap, "mean_elapsed_seconds": seconds
            })
        common[key] = min(trials, key=lambda row: (row[0], row[1]))[2]
    island = dict(common, num_islands=2)
    for key, values in sweeps_island.items():
        trials = []
        for value in values:
            candidate = dict(island, **{key: value})
            gap, seconds = _mean_gap(matrix, reference, candidate, seed,
                                     workers, repeats)
            trials.append((gap, seconds, value))
            measurements.append({
                "class": class_name, "variant": "islands", "parameter": key,
                "value": value, "mean_gap": gap, "mean_elapsed_seconds": seconds
            })
        island[key] = min(trials, key=lambda row: (row[0], row[1]))[2]
    return common, island, measurements


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)



def summarize_comparison(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["class"], row["size"], row["variant"]), []).append(row)
    summary = []
    for (name, size, variant), group in sorted(groups.items()):
        summary.append({
            "class": name, "size": size, "variant": variant,
            "runs": len(group),
            "mean_relative_gap": statistics.mean(x["relative_gap"] for x in group),
            "optimal_rate": statistics.mean(int(x["relative_gap"] == 0) for x in group),
            "mean_seconds": statistics.mean(x["elapsed_seconds"] for x in group),
            "total_population": group[0]["total_population"],
            "generations": group[0]["generations"],
        })
    return summary


def plot_comparison(summary, output_dir):
    """Plots reflect measured holdout data only; smoke output is labeled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output_dir / "plots"
    plots.mkdir(exist_ok=True)
    classes = sorted(set(row["class"] for row in summary))
    for name in classes:
        for key, y_label, multiplier in (
            ("mean_relative_gap", "Mean relative error (%)", 100),
            ("mean_seconds", "Wall-clock time including multiprocessing overhead (s)", 1),
        ):
            fig, ax = plt.subplots(figsize=(9, 5))
            for variant in ("single", "islands"):
                subset = sorted(
                    (row for row in summary if row["class"] == name and
                     row["variant"] == variant),
                    key=lambda x: x["size"]
                )
                ax.plot([row["size"] for row in subset],
                        [row[key] * multiplier for row in subset],
                        marker="o", label=variant)
            ax.set_title(f"{name}: matched total population and generations")
            ax.set_xlabel("Assignment matrix size n x n")
            ax.set_ylabel(y_label)
            ax.legend()
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(plots / f"{name}_{key}.png", dpi=140)
            plt.close(fig)


def run_study(mode: str = "smoke", output_dir: Path | str = "results/scale_benchmark",
              seed: int = 20261026, workers: int = 2,
              repeats: int | None = None):
    """Tune on one representative size, validate on independent boundary sizes.

    Full results are only generated by explicitly selecting mode=full.
    The output never labels a tiny smoke test as an experiment on n>=1000.
    """
    if mode not in ("smoke", "full"):
        raise ValueError("mode must be smoke or full")
    if workers < 1:
        raise ValueError("workers must be >=1")
    if repeats is None:
        repeats = 1 if mode == "smoke" else 2
    if repeats < 1:
        raise ValueError("repeats must be >=1")
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    training = SMOKE_TRAIN if mode == "smoke" else TRAIN_SIZES
    validation = SMOKE_VALIDATE if mode == "smoke" else VALIDATE_SIZES
    report = {
        "mode": mode, "seed": seed, "repeats": repeats, "workers": workers,
        "method": "greedy one-factor-at-a-time tuning; independent holdout comparison",
        "criterion": "minimum training mean relative gap, elapsed time tie-break",
        "fairness": "same cost matrix, total population, generations and GA operators; exact SciPy outside GA timer",
        "warning": "smoke sizes are technical tests, not claimed as medium/large research" if mode == "smoke" else
                   "best among tested candidates, not global hyperparameter optimum",
        "tested_sizes": {},
        "best_parameters": {},
        "comparison": [],
    }
    tuning = []
    for class_index, class_name in enumerate(SIZE_CLASSES):
        size = training[class_name]
        matrix_seed = seed + class_index * 100000
        matrix = generate_matrix(size, matrix_seed)
        reference = exact_cost(matrix)
        common, island, rows = _tune_for_class(
            class_name, matrix, reference, mode=mode,
            seed=matrix_seed + 3, workers=workers, repeats=repeats
        )
        tuning.extend(rows)
        report["best_parameters"][class_name] = {
            "single": common, "islands": island,
            "training_size": size, "training_reference_cost": reference,
        }
        report["tested_sizes"][class_name] = [size, *validation[class_name]]
        for validation_index, n in enumerate(validation[class_name]):
            validation_seed = seed + 500000 + class_index * 100000 + validation_index * 100
            holdout = generate_matrix(n, validation_seed)
            holdout_exact = exact_cost(holdout)
            for repeat in range(repeats):
                pair = evaluate_pair(
                    holdout, common, island, seed=validation_seed + repeat + 7,
                    workers=workers, reference=holdout_exact,
                    matrix_seed=validation_seed
                )
                for row in pair:
                    row["class"] = class_name
                    row["repeat"] = repeat
                report["comparison"].extend(pair)
        # Write partial confirmed results per class in case a full run is interrupted.
        _write_csv(folder / "tuning.csv", tuning)
        _write_csv(folder / "comparison.csv", report["comparison"])
        summary = summarize_comparison(report["comparison"])
        _write_csv(folder / "summary.csv", summary)
        plot_comparison(summary, folder)
        (folder / "best_parameters.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"{class_name}: tuned n={size}; validated {validation[class_name]}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output-dir", default="results/scale_benchmark")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()
    run_study(mode=args.mode, output_dir=args.output_dir, seed=args.seed,
              workers=args.workers, repeats=args.repeats)


if __name__ == "__main__":
    main()
