"""Release gate: verify measured data, matrix splits and honest island budgets.

These checks fail if plot counts or raw evidence disappear from the repo.
They are independent of Python unit tests and never generate fake runs.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(relative):
    with (ROOT / relative).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plots(folder, expected):
    root = ROOT / folder / "plots"
    png = list(root.glob("*.png"))
    svg = list(root.glob("*.svg"))
    assert len(png) == expected, (folder, "png", len(png), expected)
    assert len(svg) == expected, (folder, "svg", len(svg), expected)
    assert all(p.stat().st_size > 3000 for p in png + svg)


def check_three_cycle():
    from medium_three_cycle_study import OFAT, SURVEY_SIZES
    from staged_tuning import class_for
    base = "results/medium_3cycle_20261028"
    survey = rows(f"{base}/survey.csv")
    train = rows(f"{base}/train.csv")
    hold = rows(f"{base}/holdout.csv")
    sizes = {n for values in SURVEY_SIZES.values() for n in values}
    assert len(sizes) == 35
    assert len(train) == 54 and len(survey) == 35 * sum(map(len, OFAT.values()))
    assert len(hold) == 30
    assert {int(r["size"]) for r in survey} == sizes
    # The immutable historic CSV retains its old medium/transition labels.\n    # Reclassify from numeric n, as the current plots do; never rewrite data.\n    assert {class_for(int(r["size"])) for r in survey} == {\n        "small", "medium", "large"\n    }
    assert all(float(r["error_pct"]) >= -1e-8 for r in train + survey + hold)
    for record in survey:
        before = json.loads(record["baseline_json"])
        after = json.loads(record["settings_json"])
        changed = {key for key in before if before[key] != after[key]}
        assert changed <= {record["parameter"]}, (record["size"], record["parameter"])
        assert int(record["population"]) == after["population"]
    for parameter, values in OFAT.items():
        for value in values:
            measured = [r for r in survey
                        if r["parameter"] == parameter and r["value"] == str(value)]
            assert len(measured) == 35
            assert {int(r["size"]) for r in measured} == sizes
    datasets = [
        {r["matrix_seed"] for r in train},
        {r["matrix_seed"] for r in survey},
        {r["matrix_seed"] for r in hold},
    ]
    assert not (datasets[0] & datasets[1] or datasets[0] & datasets[2]
                or datasets[1] & datasets[2])
    paired = defaultdict(dict)
    for row in hold:
        paired[int(row["size"]), int(row["repeat"])][row["cycle3"]] = row
    assert len(paired) == 15
    for key, group in paired.items():
        assert set(group) == {"True", "False"}, key
        a, b = group["True"], group["False"]
        for field in ("exact", "matrix_seed", "ga_seed", "population",
                      "generations", "islands", "local_calls"):
            assert a[field] == b[field], (key, field)
    plots(base, 8)
    return {"three_cycle_train": len(train), "three_cycle_ofat": len(survey),
            "three_cycle_holdout": len(hold)}


def check_equal_budget():
    sample = rows("results/scale_benchmark_full/comparison.csv")
    assert sample
    pairs = defaultdict(dict)
    for row in sample:
        key = (row["class"], row["size"], row["repeat"])
        assert row["variant"] not in pairs[key], key
        pairs[key][row["variant"]] = row
        assert float(row["relative_gap"]) >= -1e-8
        assert math.isclose(float(row["relative_gap"]),
                            (int(row["ga_cost"]) - int(row["exact_cost"]))
                            / int(row["exact_cost"]), rel_tol=1e-10)
    for key, variants in pairs.items():
        assert set(variants) == {"single", "islands"}, key
        single, island = variants["single"], variants["islands"]
        for field in ("matrix_seed", "seed", "exact_cost",
                      "total_population", "generations"):
            assert single[field] == island[field], (key, field)
        assert single["parallel"] == "False" and island["parallel"] == "True"
        assert int(single["num_islands"]) == 1
        assert int(island["num_islands"]) > 1
    assert len(pairs) >= 20
    return {"paired_equal_budget_trials": len(pairs)}


def check_legacy_ofat():
    root = "results/continuous_three_class_20261028"
    best = rows(f"{root}/three_class_best_levels.csv")
    influence = rows(f"{root}/three_class_parameter_influence.csv")
    assert len(best) == len(influence) == 45
    assert {r["class_name"] for r in best} == {"small", "medium", "large"}
    assert "islands" in {r["parameter"] for r in best}
    assert all(float(r["error_spread_percentage_points"]) >= 0 for r in influence)
    plots(root, 30)
    return {"historical_factors_per_class": 15, "historical_charts": 30}


def main():
    import sys
    sys.path.insert(0, str(ROOT / "lab1"))
    evidence = {}
    for fn in (check_three_cycle, check_equal_budget, check_legacy_ofat):
        evidence.update(fn())
    print(json.dumps({"result": "PASS", "source": "committed empirical CSV",
                      "scope": "existing evidence, not full updated three-class sweep",
                      **evidence}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
