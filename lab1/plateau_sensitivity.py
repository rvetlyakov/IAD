"""Focused, one-factor plateau-sensitivity study.

The primary study uses very strict thresholds, so none of its plateau
settings interrupts an execution before max_iterations. This complementary
study holds a looser baseline threshold fixed and varies one stopping
parameter at a time, on the same deterministic matrices and GA seeds.
"""
from pathlib import Path

from experiments import BASELINE, save_study

BASELINE_PLATEAU = {**BASELINE, "plateau_diff": 0.05}
PLATEAU_SWEEPS = {
    "plateau_limit": (3, 5, 10, 20),
    "plateau_diff": (0.005, 0.02, 0.05, 0.1),
}


def main():
    destination = Path("results") / "plateau_sensitivity"
    rows, summary = save_study(
        destination,
        matrix_count=6,
        repeats=2,
        seed=20261021,
        baseline=BASELINE_PLATEAU,
        sweeps=PLATEAU_SWEEPS,
    )
    print(f"Additional plateau study: {len(rows)} trials, {len(summary)} summaries")


if __name__ == "__main__":
    main()
