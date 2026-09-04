from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = PROJECT_ROOT / "数据" / "清洗后数据"
SCENARIO_DIR = PROJECT_ROOT / "数据" / "情景数据"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
Q2_CONFIG_PATH = PROJECT_ROOT / "config" / "q2.json"
TOLERANCE = 1e-6

PLANS = {
    "q2": {
        "label": "Q2_lambda_025",
        "solution": "q2_lambda_025_solution_long.csv",
    },
    "q3a": {
        "label": "Q3A_correlation_only",
        "solution": "q3a_solution_long.csv",
    },
    "q3b": {
        "label": "Q3B_correlation_elasticity",
        "solution": "q3b_solution_long.csv",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def empirical_lower_cvar(values: np.ndarray, alpha: float) -> float:
    count = max(1, int(np.ceil((1.0 - alpha) * len(values))))
    return float(np.mean(np.sort(values)[:count]))


def load_area(path: Path) -> dict[tuple[int, str, str, int], float]:
    area: dict[tuple[int, str, str, int], float] = defaultdict(float)
    for row in read_csv(path):
        area[(int(row["year"]), row["plot_id"], row["season"], int(row["crop_id"]))] += float(row["area_mu"])
    return dict(area)


def profits_for_plan(
    area_by_key: dict[tuple[int, str, str, int], float],
    scenario_data: dict[str, np.ndarray],
    years: list[int],
    plot_by_id: dict[str, dict[str, Any]],
    demand_index: dict[tuple[int, str], int],
    parameter_index: dict[tuple[int, str, str], int],
    price_index: dict[tuple[int, str], int],
    crop_seasons: list[tuple[int, str]],
) -> np.ndarray:
    demand_scenarios = scenario_data["demand"]
    yield_scenarios = scenario_data["yield"]
    cost_scenarios = scenario_data["cost"]
    price_scenarios = scenario_data["price"]
    year_index = {year: index for index, year in enumerate(years)}
    profits = np.zeros(demand_scenarios.shape[0], dtype=np.float64)
    production_terms: dict[tuple[int, int, str], list[tuple[int, float]]] = defaultdict(list)
    cost_terms: list[tuple[int, int, float]] = []
    for (year, plot_id, season, crop_id), area in area_by_key.items():
        if area <= TOLERANCE:
            continue
        pindex = parameter_index[(crop_id, plot_by_id[plot_id]["land_type"], season)]
        production_terms[(year, crop_id, season)].append((pindex, area))
        cost_terms.append((year, pindex, area))
    for scenario in range(demand_scenarios.shape[0]):
        revenue = 0.0
        cost = 0.0
        for year, pindex, area in cost_terms:
            cost += area * float(cost_scenarios[scenario, year_index[year], pindex])
        for year in years:
            yi = year_index[year]
            for crop_id, season in crop_seasons:
                production = 0.0
                for pindex, area in production_terms[(year, crop_id, season)]:
                    production += area * float(yield_scenarios[scenario, yi, pindex])
                dindex = demand_index.get((crop_id, season))
                demand = 0.0 if dindex is None else float(demand_scenarios[scenario, yi, dindex])
                sold = min(production, demand)
                revenue += sold * float(price_scenarios[scenario, yi, price_index[(crop_id, season)]])
        profits[scenario] = revenue - cost
    return profits


def summarize(name: str, profits: np.ndarray, alpha: float) -> dict[str, Any]:
    return {
        "plan": name,
        "scenario_count": int(len(profits)),
        "mean_profit_yuan": round(float(np.mean(profits)), 6),
        "std_profit_yuan": round(float(np.std(profits, ddof=0)), 6),
        "p05_profit_yuan": round(float(np.quantile(profits, 0.05)), 6),
        "var_05_profit_yuan": round(float(np.quantile(profits, 0.05)), 6),
        "lower_cvar_090_yuan": round(empirical_lower_cvar(profits, alpha), 6),
        "worst_1pct_mean_yuan": round(empirical_lower_cvar(profits, 0.99), 6),
        "negative_profit_rate": round(float(np.mean(profits < 0.0)), 6),
    }


def paired_stats(left: np.ndarray, right: np.ndarray, label: str) -> dict[str, Any]:
    diff = left - right
    return {
        "comparison": label,
        "mean_difference_yuan": round(float(np.mean(diff)), 6),
        "median_difference_yuan": round(float(np.median(diff)), 6),
        "win_rate": round(float(np.mean(diff > 0.0)), 6),
        "p05_difference_yuan": round(float(np.quantile(diff, 0.05)), 6),
        "lower_cvar_difference_yuan": round(empirical_lower_cvar(diff, 0.90), 6),
    }


def maybe_plot(profits: dict[str, np.ndarray], area_change: list[dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, values in profits.items():
        ax.hist(values / 1e4, bins=40, alpha=0.35, label=name)
    ax.set_xlabel("累计利润（万元）")
    ax.set_ylabel("情景数")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "q2_q3_profit_difference.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(profits)
    means = [float(np.mean(profits[name])) / 1e4 for name in labels]
    cvars = [empirical_lower_cvar(profits[name], 0.90) / 1e4 for name in labels]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, means, width=0.35, label="平均利润")
    ax.bar(x + 0.18, cvars, width=0.35, label="下尾CVaR")
    ax.set_xticks(x, labels)
    ax.set_ylabel("万元")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "q2_q3_risk_comparison.png", dpi=150)
    plt.close(fig)

    top = sorted(area_change, key=lambda row: abs(float(row["q3b_minus_q2_mu"])), reverse=True)[:15]
    fig, ax = plt.subplots(figsize=(8, 6))
    names = [row["crop_name"] for row in reversed(top)]
    deltas = [float(row["q3b_minus_q2_mu"]) for row in reversed(top)]
    ax.barh(names, deltas)
    ax.set_xlabel("Q3-B 相对 Q2 的七年累计面积变化（亩）")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "q3_area_change.png", dpi=150)
    plt.close(fig)


def main() -> None:
    q2_config = json.loads(Q2_CONFIG_PATH.read_text(encoding="utf-8"))
    metadata = json.loads((SCENARIO_DIR / "q2_scenario_metadata.json").read_text(encoding="utf-8"))
    years = [int(year) for year in metadata["years"]]
    alpha = float(q2_config["risk"]["alpha"])
    plots = read_csv(CLEAN_DIR / "plots.csv")
    crops = read_csv(CLEAN_DIR / "crops.csv")
    eligibility = read_csv(CLEAN_DIR / "eligibility.csv")
    plot_by_id = {row["plot_id"]: {"land_type": row["land_type"], "area": float(row["area_mu"])} for row in plots}
    crop_name = {int(row["crop_id"]): row["crop_name"] for row in crops}
    eligible_keys = {
        (row["plot_id"], row["season"], int(row["crop_id"]))
        for row in eligibility if int(row["eligible"]) == 1
    }
    crop_seasons = sorted({(crop_id, season) for _plot_id, season, crop_id in eligible_keys})
    demand_keys = [(int(item[0]), str(item[1])) for item in metadata["demand_keys"]]
    parameter_keys = [(int(item[0]), str(item[1]), str(item[2])) for item in metadata["parameter_keys"]]
    price_keys = [(int(item[0]), str(item[1])) for item in metadata["price_keys"]]
    demand_index = {key: index for index, key in enumerate(demand_keys)}
    parameter_index = {key: index for index, key in enumerate(parameter_keys)}
    price_index = {key: index for index, key in enumerate(price_keys)}

    def evaluate(area_by_key, filename: str) -> np.ndarray:
        return profits_for_plan(
            area_by_key,
            np.load(SCENARIO_DIR / filename),
            years,
            plot_by_id,
            demand_index,
            parameter_index,
            price_index,
            crop_seasons,
        )

    q2_area = load_area(RESULTS_DIR / PLANS["q2"]["solution"])
    q2_worlds = {
        "q2_independent": "q2_test_2000.npz",
        "q3_weak_correlation": "q3_weak_test_2000.npz",
        "q3_medium_correlation": "q3_medium_test_2000.npz",
    }
    q2_world_profits = {}
    q2_risk_rows = []
    for world_name, filename in q2_worlds.items():
        if not (SCENARIO_DIR / filename).exists():
            continue
        values = evaluate(q2_area, filename)
        q2_world_profits[world_name] = values
        q2_risk_rows.append({**summarize("Q2_lambda_025", values, alpha), "test_world": world_name})
    write_csv(
        RESULTS_DIR / "q2_independent_vs_correlated.csv",
        [
            "plan", "test_world", "scenario_count", "mean_profit_yuan", "std_profit_yuan",
            "p05_profit_yuan", "var_05_profit_yuan", "lower_cvar_090_yuan",
            "worst_1pct_mean_yuan", "negative_profit_rate",
        ],
        q2_risk_rows,
    )
    if "q2_independent" in q2_world_profits and "q3_weak_correlation" in q2_world_profits:
        write_csv(
            RESULTS_DIR / "q2_independent_vs_correlated_diff.csv",
            ["comparison", "mean_difference_yuan", "median_difference_yuan", "win_rate", "p05_difference_yuan", "lower_cvar_difference_yuan"],
            [
                paired_stats(q2_world_profits["q3_weak_correlation"], q2_world_profits["q2_independent"], "weak_corr_minus_independent"),
                *(
                    [paired_stats(q2_world_profits["q3_medium_correlation"], q2_world_profits["q2_independent"], "medium_corr_minus_independent")]
                    if "q3_medium_correlation" in q2_world_profits
                    else []
                ),
            ],
        )

    test = {
        key: np.load(SCENARIO_DIR / filename)
        for key, filename in {
            "correlation": "q3_weak_test_2000.npz",
            "elasticity": "q3_weak_elasticity_test_2000.npz",
        }.items()
    }
    areas = {name: load_area(RESULTS_DIR / spec["solution"]) for name, spec in PLANS.items()}
    primary = "elasticity"
    profits = {
        spec["label"]: profits_for_plan(
            areas[name], test[primary], years, plot_by_id, demand_index, parameter_index, price_index, crop_seasons,
        )
        for name, spec in PLANS.items()
    }
    also_corr = {
        spec["label"]: profits_for_plan(
            areas[name], test["correlation"], years, plot_by_id, demand_index, parameter_index, price_index, crop_seasons,
        )
        for name, spec in PLANS.items()
    }

    metric_rows = [summarize(name, values, alpha) for name, values in profits.items()]
    for row in metric_rows:
        row["test_world"] = "correlation_plus_elasticity"
    metric_rows.extend({**summarize(name, values, alpha), "test_world": "correlation_only"} for name, values in also_corr.items())
    write_csv(
        RESULTS_DIR / "q3_comparison_metrics.csv",
        [
            "plan", "test_world", "scenario_count", "mean_profit_yuan", "std_profit_yuan",
            "p05_profit_yuan", "var_05_profit_yuan", "lower_cvar_090_yuan",
            "worst_1pct_mean_yuan", "negative_profit_rate",
        ],
        metric_rows,
    )

    q2_p = profits["Q2_lambda_025"]
    q3a_p = profits["Q3A_correlation_only"]
    q3b_p = profits["Q3B_correlation_elasticity"]
    paired_rows = [
        {"scenario_id": index, "q2_profit_yuan": q2_p[index], "q3a_profit_yuan": q3a_p[index], "q3b_profit_yuan": q3b_p[index],
         "q3a_minus_q2": q3a_p[index] - q2_p[index], "q3b_minus_q2": q3b_p[index] - q2_p[index],
         "q3b_minus_q3a": q3b_p[index] - q3a_p[index]}
        for index in range(len(q2_p))
    ]
    write_csv(
        RESULTS_DIR / "q2_q3_paired_profit.csv",
        ["scenario_id", "q2_profit_yuan", "q3a_profit_yuan", "q3b_profit_yuan", "q3a_minus_q2", "q3b_minus_q2", "q3b_minus_q3a"],
        [{"scenario_id": row["scenario_id"], **{key: round(float(row[key]), 6) for key in row if key != "scenario_id"}} for row in paired_rows],
    )
    comparison_summary = [
        paired_stats(q3a_p, q2_p, "Q3A_minus_Q2"),
        paired_stats(q3b_p, q2_p, "Q3B_minus_Q2"),
        paired_stats(q3b_p, q3a_p, "Q3B_minus_Q3A"),
    ]
    write_csv(
        RESULTS_DIR / "q3_paired_difference_summary.csv",
        ["comparison", "mean_difference_yuan", "median_difference_yuan", "win_rate", "p05_difference_yuan", "lower_cvar_difference_yuan"],
        comparison_summary,
    )

    crop_area: dict[str, dict[int, float]] = {name: defaultdict(float) for name in areas}
    for name, area_by_key in areas.items():
        for (_year, _plot, _season, crop_id), area in area_by_key.items():
            crop_area[name][crop_id] += area
    crop_ids = sorted({crop_id for mapping in crop_area.values() for crop_id in mapping})
    area_change = []
    for crop_id in crop_ids:
        q2_area = crop_area["q2"][crop_id]
        q3a_area = crop_area["q3a"][crop_id]
        q3b_area = crop_area["q3b"][crop_id]
        area_change.append({
            "crop_id": crop_id,
            "crop_name": crop_name[crop_id],
            "q2_total_mu": round(q2_area, 6),
            "q3a_total_mu": round(q3a_area, 6),
            "q3b_total_mu": round(q3b_area, 6),
            "q3a_minus_q2_mu": round(q3a_area - q2_area, 6),
            "q3b_minus_q2_mu": round(q3b_area - q2_area, 6),
        })
    write_csv(
        RESULTS_DIR / "q3_crop_area_change.csv",
        ["crop_id", "crop_name", "q2_total_mu", "q3a_total_mu", "q3b_total_mu", "q3a_minus_q2_mu", "q3b_minus_q2_mu"],
        area_change,
    )
    maybe_plot(profits, area_change)
    print(json.dumps({"primary_test_world": primary, "metrics": metric_rows[:3], "paired": comparison_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
