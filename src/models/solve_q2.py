from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / ".python_packages"))

import highspy  # noqa: E402
import numpy as np  # noqa: E402


CLEAN_DIR = PROJECT_ROOT / "数据" / "清洗后数据"
SCENARIO_DIR = PROJECT_ROOT / "数据" / "情景数据"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_PATH = PROJECT_ROOT / "config" / "q2.json"

CANDIDATES = {
    "mean_value": {
        "scenario_file": "q2_mean.npz",
        "lambda": 0.0,
        "time_limit_key": "mean_value_time_limit_seconds",
        "warm_start": "results/q1_full_waste_solution_long.csv",
    },
    "lambda_0": {
        "scenario_file": "q2_optimization_100.npz",
        "lambda": 0.0,
        "time_limit_key": "saa_time_limit_seconds",
        "warm_start": "results/q2_mean_value_solution_long.csv",
    },
    "lambda_025": {
        "scenario_file": "q2_optimization_100.npz",
        "lambda": 0.25,
        "time_limit_key": "saa_time_limit_seconds",
        "warm_start": "results/q2_lambda_0_solution_long.csv",
    },
    "lambda_050": {
        "scenario_file": "q2_optimization_100.npz",
        "lambda": 0.50,
        "time_limit_key": "saa_time_limit_seconds",
        "warm_start": "results/q2_lambda_025_solution_long.csv",
    },
    "q3a": {
        "scenario_file": "q3_weak_optimization_100.npz",
        "lambda": 0.25,
        "time_limit_key": "saa_time_limit_seconds",
        "warm_start": "results/q2_lambda_025_solution_long.csv",
        "output_prefix": "q3a",
        "model_name": "Q3A_correlation_only",
    },
    "q3b": {
        "scenario_file": "q3_weak_elasticity_optimization_100.npz",
        "lambda": 0.25,
        "time_limit_key": "saa_time_limit_seconds",
        "warm_start": "results/q3a_solution_long.csv",
        "output_prefix": "q3b",
        "model_name": "Q3B_correlation_elasticity",
    },
    "q3a_medium": {
        "scenario_file": "q3_medium_optimization_100.npz",
        "lambda": 0.25,
        "time_limit_key": "saa_time_limit_seconds",
        "warm_start": "results/q2_lambda_025_solution_long.csv",
        "output_prefix": "q3a_medium",
        "model_name": "Q3A_medium_correlation",
    },
    "q3b_medium": {
        "scenario_file": "q3_medium_elasticity_optimization_100.npz",
        "lambda": 0.25,
        "time_limit_key": "saa_time_limit_seconds",
        "warm_start": "results/q3a_medium_solution_long.csv",
        "output_prefix": "q3b_medium",
        "model_name": "Q3B_medium_correlation_elasticity",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def variable_sum(variables: list[Any]) -> Any:
    if not variables:
        return 0.0
    expression = variables[0]
    for variable in variables[1:]:
        expression = expression + variable
    return expression


def weighted_variable_sum(terms: list[tuple[float, Any]]) -> Any:
    if not terms:
        return 0.0
    expression = terms[0][0] * terms[0][1]
    for coefficient, variable in terms[1:]:
        expression = expression + coefficient * variable
    return expression


def empirical_lower_cvar(values: np.ndarray, alpha: float) -> float:
    count = max(1, int(math.ceil((1.0 - alpha) * len(values))))
    return float(np.mean(np.sort(values)[:count]))


def load_start_area(relative_path: str | None) -> dict[tuple[int, str, str, int], float]:
    if not relative_path:
        return {}
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        return {}
    return {
        (int(row["year"]), row["plot_id"], row["season"], int(row["crop_id"])): float(row["area_mu"])
        for row in read_csv(path)
    }


def main(candidate_name: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    candidate = CANDIDATES[candidate_name]
    years = [int(year) for year in config["years"]]
    year_index = {year: index for index, year in enumerate(years)}
    tolerance = float(config["numeric_tolerance"])
    risk_alpha = float(config["risk"]["alpha"])
    risk_lambda = float(candidate["lambda"])

    plots = read_csv(CLEAN_DIR / "plots.csv")
    crops = read_csv(CLEAN_DIR / "crops.csv")
    planting_2023 = read_csv(CLEAN_DIR / "planting_2023.csv")
    parameters = read_csv(CLEAN_DIR / "crop_parameters_2023.csv")
    eligibility_rows = read_csv(CLEAN_DIR / "eligibility.csv")
    metadata = json.loads((SCENARIO_DIR / "q2_scenario_metadata.json").read_text(encoding="utf-8"))
    scenario_data = np.load(SCENARIO_DIR / candidate["scenario_file"])
    demand_scenarios = scenario_data["demand"]
    yield_scenarios = scenario_data["yield"]
    cost_scenarios = scenario_data["cost"]
    price_scenarios = scenario_data["price"]
    scenario_count = int(demand_scenarios.shape[0])

    demand_keys = [(int(item[0]), str(item[1])) for item in metadata["demand_keys"]]
    parameter_keys = [(int(item[0]), str(item[1]), str(item[2])) for item in metadata["parameter_keys"]]
    price_keys = [(int(item[0]), str(item[1])) for item in metadata["price_keys"]]
    demand_key_index = {key: index for index, key in enumerate(demand_keys)}
    parameter_key_index = {key: index for index, key in enumerate(parameter_keys)}
    price_key_index = {key: index for index, key in enumerate(price_keys)}

    plot_by_id = {
        row["plot_id"]: {"land_type": row["land_type"], "area_mu": float(row["area_mu"])}
        for row in plots
    }
    crop_by_id = {
        int(row["crop_id"]): {
            "crop_name": row["crop_name"],
            "crop_type": row["crop_type"],
            "crop_category": row["crop_category"],
            "is_legume": int(row["is_legume"]),
        }
        for row in crops
    }
    legume_crop_ids = {crop_id for crop_id, crop in crop_by_id.items() if crop["is_legume"] == 1}

    eligible_combinations: list[tuple[str, str, int]] = []
    for row in eligibility_rows:
        if int(row["eligible"]) != 1:
            continue
        key = (row["plot_id"], row["season"], int(row["crop_id"]))
        parameter_key = (key[2], plot_by_id[key[0]]["land_type"], key[1])
        if parameter_key not in parameter_key_index:
            raise KeyError(f"适宜组合缺少情景参数键：{parameter_key}")
        eligible_combinations.append(key)
    crop_seasons = sorted({(crop_id, season) for _plot_id, season, crop_id in eligible_combinations})

    model = highspy.Highs()
    solver_config = config["solver"]
    time_limit = float(solver_config[candidate["time_limit_key"]])
    model.setOptionValue("time_limit", time_limit)
    model.setOptionValue("mip_rel_gap", float(solver_config["mip_relative_gap"]))
    model.setOptionValue("random_seed", int(solver_config["random_seed"]))
    if "threads" in solver_config:
        model.setOptionValue("threads", int(solver_config["threads"]))
    if "parallel" in solver_config:
        model.setOptionValue("parallel", str(solver_config["parallel"]))
    prefix = str(candidate.get("output_prefix", f"q2_{candidate_name}"))
    model.setOptionValue("log_file", str(LOGS_DIR / f"{prefix}_solver.log"))
    model.setOptionValue("output_flag", True)

    x: dict[tuple[int, str, str, int], Any] = {}
    y: dict[tuple[int, str, str, int], Any] = {}
    x_by_year_plot_season: dict[tuple[int, str, str], list[Any]] = defaultdict(list)
    x_by_year_crop_season: dict[
        tuple[int, int, str], list[tuple[int, tuple[int, str, str, int], Any]]
    ] = defaultdict(list)
    x_by_year_plot_legume: dict[tuple[int, str], list[Any]] = defaultdict(list)
    y_by_year_crop_season: dict[tuple[int, int, str], list[Any]] = defaultdict(list)
    minimum_area_by_land_type = {
        land_type: float(value)
        for land_type, value in config["management"]["minimum_area_mu_by_land_type"].items()
    }

    for year in years:
        for plot_id, season, crop_id in eligible_combinations:
            plot = plot_by_id[plot_id]
            parameter_index = parameter_key_index[(crop_id, plot["land_type"], season)]
            key = (year, plot_id, season, crop_id)
            x_var = model.addVariable(
                lb=0.0,
                ub=plot["area_mu"],
                type=highspy.HighsVarType.kContinuous,
                name=f"x_{year}_{plot_id}_{season}_{crop_id}",
            )
            y_var = model.addVariable(
                lb=0.0,
                ub=1.0,
                type=highspy.HighsVarType.kInteger,
                name=f"y_{year}_{plot_id}_{season}_{crop_id}",
            )
            x[key] = x_var
            y[key] = y_var
            x_by_year_plot_season[(year, plot_id, season)].append(x_var)
            x_by_year_crop_season[(year, crop_id, season)].append((parameter_index, key, x_var))
            if crop_id in legume_crop_ids:
                x_by_year_plot_legume[(year, plot_id)].append(x_var)
            y_by_year_crop_season[(year, crop_id, season)].append(y_var)
            model.addConstr(x_var <= plot["area_mu"] * y_var, name=f"link_{year}_{plot_id}_{season}_{crop_id}")
            model.addConstr(
                x_var >= minimum_area_by_land_type[plot["land_type"]] * y_var,
                name=f"min_area_{year}_{plot_id}_{season}_{crop_id}",
            )

    # Scenario-specific normal-price sales and total scenario profit.
    normal_sales: dict[tuple[int, int, int, str], Any] = {}
    profit_variables: list[Any] = []
    for scenario in range(scenario_count):
        revenue_terms: list[tuple[float, Any]] = []
        cost_terms: list[tuple[float, Any]] = []
        for year in years:
            yi = year_index[year]
            for crop_id, season in crop_seasons:
                demand_index = demand_key_index.get((crop_id, season))
                demand = 0.0 if demand_index is None else float(demand_scenarios[scenario, yi, demand_index])
                price = float(price_scenarios[scenario, yi, price_key_index[(crop_id, season)]])
                sales_var = model.addVariable(
                    lb=0.0,
                    ub=demand,
                    type=highspy.HighsVarType.kContinuous,
                    name=f"u_{scenario}_{year}_{crop_id}_{season}",
                )
                normal_sales[(scenario, year, crop_id, season)] = sales_var
                production_terms = [
                    (float(yield_scenarios[scenario, yi, parameter_index]), variable)
                    for parameter_index, _area_key, variable in x_by_year_crop_season[(year, crop_id, season)]
                ]
                model.addConstr(
                    sales_var <= weighted_variable_sum(production_terms),
                    name=f"sales_prod_{scenario}_{year}_{crop_id}_{season}",
                )
                revenue_terms.append((price, sales_var))
            for plot_id, season, crop_id in eligible_combinations:
                parameter_index = parameter_key_index[(crop_id, plot_by_id[plot_id]["land_type"], season)]
                cost_terms.append((float(cost_scenarios[scenario, yi, parameter_index]), x[(year, plot_id, season, crop_id)]))
        profit_var = model.addVariable(
            lb=-highspy.kHighsInf,
            ub=highspy.kHighsInf,
            obj=1.0 / scenario_count,
            type=highspy.HighsVarType.kContinuous,
            name=f"profit_{scenario}",
        )
        profit_variables.append(profit_var)
        model.addConstr(
            profit_var == weighted_variable_sum(revenue_terms) - weighted_variable_sum(cost_terms),
            name=f"profit_definition_{scenario}",
        )

    cvar_threshold = None
    cvar_shortfalls: list[Any] = []
    if risk_lambda > 0:
        cvar_threshold = model.addVariable(
            lb=-highspy.kHighsInf,
            ub=highspy.kHighsInf,
            obj=risk_lambda,
            type=highspy.HighsVarType.kContinuous,
            name="lower_tail_profit_threshold",
        )
        tail_weight = risk_lambda / ((1.0 - risk_alpha) * scenario_count)
        for scenario, profit_var in enumerate(profit_variables):
            shortfall = model.addVariable(
                lb=0.0,
                ub=highspy.kHighsInf,
                obj=-tail_weight,
                type=highspy.HighsVarType.kContinuous,
                name=f"lower_tail_shortfall_{scenario}",
            )
            cvar_shortfalls.append(shortfall)
            model.addConstr(shortfall >= cvar_threshold - profit_var, name=f"cvar_shortfall_{scenario}")

    # Shared agricultural and management constraints.
    for (year, plot_id, season), variables in x_by_year_plot_season.items():
        model.addConstr(variable_sum(variables) <= plot_by_id[plot_id]["area_mu"], name=f"area_{year}_{plot_id}_{season}")

    max_plot_count = int(config["management"]["max_plots_per_crop_per_year_season"])
    for (year, crop_id, season), flags in y_by_year_crop_season.items():
        model.addConstr(variable_sum(flags) <= max_plot_count, name=f"max_plots_{year}_{crop_id}_{season}")

    water_mode: dict[tuple[int, str], Any] = {}
    water_plots = [plot_id for plot_id, plot in plot_by_id.items() if plot["land_type"] == "水浇地"]
    for year in years:
        for plot_id in water_plots:
            area = plot_by_id[plot_id]["area_mu"]
            mode = model.addVariable(lb=0.0, ub=1.0, type=highspy.HighsVarType.kInteger, name=f"rice_mode_{year}_{plot_id}")
            water_mode[(year, plot_id)] = mode
            rice_area = x[(year, plot_id, "single", 16)]
            first_area = variable_sum(x_by_year_plot_season[(year, plot_id, "first")])
            second_area = variable_sum(x_by_year_plot_season[(year, plot_id, "second")])
            model.addConstr(rice_area <= area * mode, name=f"water_rice_{year}_{plot_id}")
            model.addConstr(first_area + area * mode <= area, name=f"water_first_{year}_{plot_id}")
            model.addConstr(second_area + area * mode <= area, name=f"water_second_{year}_{plot_id}")
            second_flags = [y[(year, plot_id, "second", crop_id)] for crop_id in (35, 36, 37)]
            model.addConstr(variable_sum(second_flags) <= 1, name=f"water_second_one_{year}_{plot_id}")

    combination_set = set(eligible_combinations)
    for plot_id, season, crop_id in eligible_combinations:
        for first_year, second_year in zip(years, years[1:]):
            model.addConstr(
                y[(first_year, plot_id, season, crop_id)] + y[(second_year, plot_id, season, crop_id)] <= 1,
                name=f"rotation_{first_year}_{plot_id}_{season}_{crop_id}",
            )
    for row in planting_2023:
        key = (row["plot_id"], row["season"], int(row["crop_id"]))
        if key in combination_set:
            model.addConstr(y[(2024, *key)] <= 0, name=f"initial_rotation_{key[0]}_{key[1]}_{key[2]}")

    # 智慧大棚两季作物集合相同，额外约束相邻实际种植槽位不能重茬：
    # 年内 first→second、跨年 second→next first，以及 2023 second→2024 first。
    smart_plots = [plot_id for plot_id, plot in plot_by_id.items() if plot["land_type"] == "智慧大棚"]
    for plot_id in smart_plots:
        for crop_id in range(17, 35):
            for year in years:
                model.addConstr(
                    y[(year, plot_id, "first", crop_id)] + y[(year, plot_id, "second", crop_id)] <= 1,
                    name=f"smart_within_{year}_{plot_id}_{crop_id}",
                )
            for first_year, second_year in zip(years, years[1:]):
                model.addConstr(
                    y[(first_year, plot_id, "second", crop_id)] + y[(second_year, plot_id, "first", crop_id)] <= 1,
                    name=f"smart_cross_{first_year}_{plot_id}_{crop_id}",
                )
    for row in planting_2023:
        plot_id = row["plot_id"]
        if plot_by_id.get(plot_id, {}).get("land_type") != "智慧大棚":
            continue
        if row["season"] != "second":
            continue
        crop_id = int(row["crop_id"])
        if (plot_id, "first", crop_id) in combination_set:
            model.addConstr(
                y[(2024, plot_id, "first", crop_id)] <= 0,
                name=f"smart_initial_cross_2023_{plot_id}_{crop_id}",
            )

    bean_area_2023: dict[str, float] = defaultdict(float)
    for row in planting_2023:
        if int(row["crop_id"]) in legume_crop_ids:
            bean_area_2023[row["plot_id"]] += float(row["area_mu"])
    for plot_id, plot in plot_by_id.items():
        for window_start in range(2023, 2029):
            future_years = [year for year in range(window_start, window_start + 3) if year >= 2024]
            bean_variables = [
                variable for year in future_years for variable in x_by_year_plot_legume[(year, plot_id)]
            ]
            known = bean_area_2023.get(plot_id, 0.0) if window_start == 2023 else 0.0
            model.addConstr(
                variable_sum(bean_variables) >= plot["area_mu"] - known,
                name=f"bean_window_{window_start}_{plot_id}",
            )

    model.setMaximize()

    # Feasible warm start shared across every scenario.
    start_area = load_start_area(candidate.get("warm_start"))
    warm_start_status = None
    if start_area:
        indices: list[int] = []
        values: list[float] = []
        for key, variable in x.items():
            value = start_area.get(key, 0.0)
            indices.extend((variable.index, y[key].index))
            values.extend((value, 1.0 if value > tolerance else 0.0))
        for (year, plot_id), variable in water_mode.items():
            value = 1.0 if start_area.get((year, plot_id, "single", 16), 0.0) > tolerance else 0.0
            indices.append(variable.index)
            values.append(value)

        start_profits = np.zeros(scenario_count, dtype=np.float64)
        for scenario in range(scenario_count):
            scenario_revenue = 0.0
            scenario_cost = 0.0
            for year in years:
                yi = year_index[year]
                for crop_id, season in crop_seasons:
                    production = 0.0
                    for parameter_index, area_key, _variable in x_by_year_crop_season[(year, crop_id, season)]:
                        area_value = start_area.get(area_key, 0.0)
                        production += area_value * float(yield_scenarios[scenario, yi, parameter_index])
                    demand_index = demand_key_index.get((crop_id, season))
                    demand = 0.0 if demand_index is None else float(demand_scenarios[scenario, yi, demand_index])
                    sales = min(production, demand)
                    sales_variable = normal_sales[(scenario, year, crop_id, season)]
                    indices.append(sales_variable.index)
                    values.append(sales)
                    scenario_revenue += sales * float(price_scenarios[scenario, yi, price_key_index[(crop_id, season)]])
                for plot_id, season, crop_id in eligible_combinations:
                    parameter_index = parameter_key_index[(crop_id, plot_by_id[plot_id]["land_type"], season)]
                    scenario_cost += start_area.get((year, plot_id, season, crop_id), 0.0) * float(
                        cost_scenarios[scenario, yi, parameter_index]
                    )
            start_profits[scenario] = scenario_revenue - scenario_cost
            indices.append(profit_variables[scenario].index)
            values.append(float(start_profits[scenario]))
        if risk_lambda > 0 and cvar_threshold is not None:
            threshold = float(np.quantile(start_profits, 1.0 - risk_alpha))
            indices.append(cvar_threshold.index)
            values.append(threshold)
            for scenario, shortfall in enumerate(cvar_shortfalls):
                indices.append(shortfall.index)
                values.append(max(0.0, threshold - float(start_profits[scenario])))
        warm_start_status = str(
            model.setSolution(
                len(indices),
                np.asarray(indices, dtype=np.int32),
                np.asarray(values, dtype=np.float64),
            )
        )

    model.run()
    status = model.getModelStatus()
    status_text = model.modelStatusToString(status)
    info = model.getInfo()
    if info.primal_solution_status != highspy.SolutionStatus.kSolutionStatusFeasible:
        raise RuntimeError(f"Q2 {candidate_name} 未得到可行解，状态={status_text}")

    solution_rows: list[dict[str, Any]] = []
    for key, variable in x.items():
        area = float(model.val(variable))
        if area <= tolerance:
            continue
        year, plot_id, season, crop_id = key
        solution_rows.append({
            "candidate": candidate_name,
            "year": year,
            "season": season,
            "plot_id": plot_id,
            "land_type": plot_by_id[plot_id]["land_type"],
            "crop_id": crop_id,
            "crop_name": crop_by_id[crop_id]["crop_name"],
            "area_mu": round(area, 6),
        })
    solution_rows.sort(key=lambda row: (row["year"], row["plot_id"], row["season"], row["crop_id"]))
    write_csv(
        RESULTS_DIR / f"{prefix}_solution_long.csv",
        ["candidate", "year", "season", "plot_id", "land_type", "crop_id", "crop_name", "area_mu"],
        solution_rows,
    )

    training_profits = np.asarray([float(model.val(variable)) for variable in profit_variables], dtype=np.float64)
    training_rows = [
        {"candidate": candidate_name, "scenario_id": scenario, "profit_yuan": round(float(profit), 6)}
        for scenario, profit in enumerate(training_profits)
    ]
    write_csv(
        RESULTS_DIR / f"{prefix}_training_profit.csv",
        ["candidate", "scenario_id", "profit_yuan"],
        training_rows,
    )
    metrics = {
        "model": candidate.get("model_name", config["model_name"]),
        "candidate": candidate_name,
        "surplus_policy": config["surplus_policy"],
        "scenario_file": candidate["scenario_file"],
        "scenario_count": scenario_count,
        "risk": {
            "alpha": risk_alpha,
            "lambda": risk_lambda,
            "objective_formula": "mean_profit + lambda * lower_tail_profit_CVaR",
        },
        "solver": {
            "name": "HiGHS",
            "version": model.version(),
            "status": status_text,
            "runtime_seconds": model.getRunTime(),
            "mip_gap": info.mip_gap if math.isfinite(info.mip_gap) else None,
            "dual_bound": info.mip_dual_bound if math.isfinite(info.mip_dual_bound) else None,
            "node_count": info.mip_node_count,
            "time_limit_seconds": time_limit,
        },
        "model_size": {
            "area_variables": len(x),
            "binary_planting_variables": len(y),
            "water_mode_variables": len(water_mode),
            "scenario_sales_variables": len(normal_sales),
            "scenario_profit_variables": len(profit_variables),
            "cvar_shortfall_variables": len(cvar_shortfalls),
        },
        "management_constraints": {
            "minimum_area_mu_by_land_type": minimum_area_by_land_type,
            "max_plots_per_crop_per_year_season": max_plot_count,
        },
        "warm_start": {"source": candidate.get("warm_start"), "set_status": warm_start_status},
        "objective": round(model.getObjectiveValue(), 6),
        "training_mean_profit_yuan": round(float(np.mean(training_profits)), 6),
        "training_std_profit_yuan": round(float(np.std(training_profits, ddof=0)), 6),
        "training_p05_profit_yuan": round(float(np.quantile(training_profits, 0.05)), 6),
        "training_lower_cvar_profit_yuan": round(empirical_lower_cvar(training_profits, risk_alpha), 6),
        "positive_area_rows": len(solution_rows),
        "constraint_violations": None,
        "assumption_notes": config["assumption_notes"],
    }
    (RESULTS_DIR / f"{prefix}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", choices=sorted(CANDIDATES))
    args = parser.parse_args()
    main(args.candidate)
