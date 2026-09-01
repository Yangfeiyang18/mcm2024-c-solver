from __future__ import annotations

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


CLEAN_DIR = PROJECT_ROOT / "数据" / "清洗后数据"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_PATH = PROJECT_ROOT / "config" / "q1_baseline.json"


def read_csv(name: str) -> list[dict[str, str]]:
    with (CLEAN_DIR / name).open("r", encoding="utf-8-sig", newline="") as file:
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


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    years: list[int] = config["years"]
    tolerance = float(config["numeric_tolerance"])

    plots = read_csv("plots.csv")
    crops = read_csv("crops.csv")
    planting_2023 = read_csv("planting_2023.csv")
    parameters = read_csv("crop_parameters_2023.csv")
    eligibility_rows = read_csv("eligibility.csv")
    demand_rows = read_csv("demand_2023.csv")

    plot_by_id = {
        row["plot_id"]: {"land_type": row["land_type"], "area_mu": float(row["area_mu"])}
        for row in plots
    }
    crop_by_id = {
        int(row["crop_id"]): {
            "crop_name": row["crop_name"],
            "crop_type": row["crop_type"],
            "is_legume": int(row["is_legume"]),
        }
        for row in crops
    }
    legume_crop_ids = {crop_id for crop_id, crop in crop_by_id.items() if crop["is_legume"] == 1}
    parameter_by_key = {
        (int(row["crop_id"]), row["land_type"], row["season"]): {
            "yield": float(row["yield_jin_per_mu"]),
            "cost": float(row["cost_yuan_per_mu"]),
            "price": float(row["price_mid"]),
            "source": row["parameter_source"],
        }
        for row in parameters
    }
    demand_by_key = {
        (int(row["crop_id"]), row["season"]): float(row["expected_sales_jin"])
        for row in demand_rows
    }

    eligible_combinations: list[tuple[str, str, int]] = []
    for row in eligibility_rows:
        if int(row["eligible"]) != 1:
            continue
        key = (row["plot_id"], row["season"], int(row["crop_id"]))
        plot = plot_by_id[key[0]]
        parameter_key = (key[2], plot["land_type"], key[1])
        if parameter_key not in parameter_by_key:
            raise KeyError(f"适宜组合缺少生产参数：{parameter_key}")
        eligible_combinations.append(key)

    model = highspy.Highs()
    solver_config = config["solver"]
    model.setOptionValue("time_limit", float(solver_config["time_limit_seconds"]))
    model.setOptionValue("mip_rel_gap", float(solver_config["mip_relative_gap"]))
    model.setOptionValue("random_seed", int(solver_config["random_seed"]))
    model.setOptionValue("log_file", str(LOGS_DIR / "q1_baseline_solver.log"))
    model.setOptionValue("output_flag", True)

    x: dict[tuple[int, str, str, int], Any] = {}
    y: dict[tuple[int, str, str, int], Any] = {}
    x_by_year_plot_season: dict[tuple[int, str, str], list[Any]] = defaultdict(list)
    x_by_year_crop_season: dict[tuple[int, int, str], list[tuple[float, Any]]] = defaultdict(list)
    x_by_year_plot_legume: dict[tuple[int, str], list[Any]] = defaultdict(list)

    for year in years:
        for plot_id, season, crop_id in eligible_combinations:
            plot = plot_by_id[plot_id]
            parameter = parameter_by_key[(crop_id, plot["land_type"], season)]
            key = (year, plot_id, season, crop_id)
            x_var = model.addVariable(
                lb=0.0,
                ub=plot["area_mu"],
                obj=-parameter["cost"],
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
            x_by_year_crop_season[(year, crop_id, season)].append((parameter["yield"], x_var))
            if crop_id in legume_crop_ids:
                x_by_year_plot_legume[(year, plot_id)].append(x_var)
            model.addConstr(x_var <= plot["area_mu"] * y_var, name=f"link_{year}_{plot_id}_{season}_{crop_id}")

    # 每个作物-季次的销售价格必须一致；若不一致，不能用一个共享销量变量。
    crop_season_prices: dict[tuple[int, str], float] = {}
    for crop_id, season in sorted({(key[2], key[1]) for key in eligible_combinations}):
        prices = {
            round(parameter_by_key[(crop_id, plot_by_id[plot_id]["land_type"], season)]["price"], 8)
            for plot_id, candidate_season, candidate_crop in eligible_combinations
            if candidate_crop == crop_id and candidate_season == season
        }
        if len(prices) != 1:
            raise ValueError(f"同一作物-季次存在多个价格，需要细分销售变量：{crop_id}/{season}/{prices}")
        crop_season_prices[(crop_id, season)] = prices.pop()

    normal_sales: dict[tuple[int, int, str], Any] = {}
    for year in years:
        for crop_id, season in sorted(crop_season_prices):
            demand = demand_by_key.get((crop_id, season), 0.0)
            sales_var = model.addVariable(
                lb=0.0,
                ub=demand,
                obj=crop_season_prices[(crop_id, season)],
                type=highspy.HighsVarType.kContinuous,
                name=f"u_{year}_{crop_id}_{season}",
            )
            normal_sales[(year, crop_id, season)] = sales_var
            production = weighted_variable_sum(x_by_year_crop_season[(year, crop_id, season)])
            model.addConstr(sales_var <= production, name=f"sales_le_production_{year}_{crop_id}_{season}")

    # 地块每个季次的面积容量。
    for (year, plot_id, season), variables in x_by_year_plot_season.items():
        model.addConstr(variable_sum(variables) <= plot_by_id[plot_id]["area_mu"], name=f"area_{year}_{plot_id}_{season}")

    # 水浇地只能选择单季水稻或两季蔬菜；第二季最多选择一种根菜。
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
            model.addConstr(rice_area <= area * mode, name=f"water_rice_mode_{year}_{plot_id}")
            model.addConstr(first_area + area * mode <= area, name=f"water_first_mode_{year}_{plot_id}")
            model.addConstr(second_area + area * mode <= area, name=f"water_second_mode_{year}_{plot_id}")
            second_flags = [y[(year, plot_id, "second", crop_id)] for crop_id in (35, 36, 37)]
            model.addConstr(variable_sum(second_flags) <= 1, name=f"water_second_one_crop_{year}_{plot_id}")

    # 同一地块、同一季次、同一作物不能连续两年种植。
    combination_set = set(eligible_combinations)
    for plot_id, season, crop_id in eligible_combinations:
        for first_year, second_year in zip(years, years[1:]):
            model.addConstr(
                y[(first_year, plot_id, season, crop_id)] + y[(second_year, plot_id, season, crop_id)] <= 1,
                name=f"rotation_{first_year}_{plot_id}_{season}_{crop_id}",
            )

    # 用2023年记录限制2024年同地块、同季次重茬。
    for row in planting_2023:
        key = (row["plot_id"], row["season"], int(row["crop_id"]))
        if key in combination_set:
            model.addConstr(y[(2024, *key)] <= 0, name=f"initial_rotation_2024_{key[0]}_{key[1]}_{key[2]}")

    # 智慧大棚两季作物集合相同，额外检查年内连续季和跨年连续季。
    smart_plots = [plot_id for plot_id, plot in plot_by_id.items() if plot["land_type"] == "智慧大棚"]
    for plot_id in smart_plots:
        for crop_id in range(17, 35):
            for year in years:
                model.addConstr(
                    y[(year, plot_id, "first", crop_id)] + y[(year, plot_id, "second", crop_id)] <= 1,
                    name=f"smart_within_year_{year}_{plot_id}_{crop_id}",
                )
            for first_year, second_year in zip(years, years[1:]):
                model.addConstr(
                    y[(first_year, plot_id, "second", crop_id)] + y[(second_year, plot_id, "first", crop_id)] <= 1,
                    name=f"smart_cross_year_{first_year}_{plot_id}_{crop_id}",
                )

    # 任意连续三年内，豆类累计种植面积至少覆盖该地块一次。
    bean_area_2023: dict[str, float] = defaultdict(float)
    for row in planting_2023:
        if int(row["crop_id"]) in legume_crop_ids:
            bean_area_2023[row["plot_id"]] += float(row["area_mu"])
    for plot_id, plot in plot_by_id.items():
        for window_start in range(2023, 2029):
            future_years = [year for year in range(window_start, window_start + 3) if year >= 2024]
            bean_variables = [
                variable
                for year in future_years
                for variable in x_by_year_plot_legume[(year, plot_id)]
            ]
            known_bean_area = bean_area_2023.get(plot_id, 0.0) if window_start == 2023 else 0.0
            model.addConstr(
                variable_sum(bean_variables) >= plot["area_mu"] - known_bean_area,
                name=f"bean_window_{window_start}_{plot_id}",
            )

    model.setMaximize()
    model.run()

    status = model.getModelStatus()
    status_text = model.modelStatusToString(status)
    info = model.getInfo()
    has_solution = info.primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible
    if not has_solution:
        raise RuntimeError(f"Q1基线模型没有得到可行解，状态={status_text}")

    solution_rows: list[dict[str, Any]] = []
    yearly_values: dict[int, dict[str, float]] = {
        year: {"revenue": 0.0, "cost": 0.0, "profit": 0.0, "production": 0.0, "normal_sales": 0.0, "surplus": 0.0}
        for year in years
    }
    crop_summary: dict[tuple[int, int, str], dict[str, float]] = defaultdict(
        lambda: {"area_mu": 0.0, "production_jin": 0.0, "normal_sales_jin": 0.0, "surplus_jin": 0.0, "revenue_yuan": 0.0, "cost_yuan": 0.0, "profit_yuan": 0.0}
    )

    for key, variable in x.items():
        year, plot_id, season, crop_id = key
        area_mu = float(model.val(variable))
        if area_mu < tolerance:
            continue
        plot = plot_by_id[plot_id]
        parameter = parameter_by_key[(crop_id, plot["land_type"], season)]
        production = area_mu * parameter["yield"]
        cost = area_mu * parameter["cost"]
        solution_rows.append({
            "scenario": "baseline_waste",
            "year": year,
            "season": season,
            "plot_id": plot_id,
            "land_type": plot["land_type"],
            "crop_id": crop_id,
            "crop_name": crop_by_id[crop_id]["crop_name"],
            "area_mu": round(area_mu, 6),
        })
        summary = crop_summary[(year, crop_id, season)]
        summary["area_mu"] += area_mu
        summary["production_jin"] += production
        summary["cost_yuan"] += cost
        yearly_values[year]["cost"] += cost
        yearly_values[year]["production"] += production

    for key, variable in normal_sales.items():
        year, crop_id, season = key
        normal_sales_value = max(0.0, float(model.val(variable)))
        summary = crop_summary[(year, crop_id, season)]
        production = summary["production_jin"]
        surplus = max(0.0, production - normal_sales_value)
        revenue = normal_sales_value * crop_season_prices[(crop_id, season)]
        summary["normal_sales_jin"] = normal_sales_value
        summary["surplus_jin"] = surplus
        summary["revenue_yuan"] = revenue
        summary["profit_yuan"] = revenue - summary["cost_yuan"]
        yearly_values[year]["normal_sales"] += normal_sales_value
        yearly_values[year]["surplus"] += surplus
        yearly_values[year]["revenue"] += revenue

    for year in years:
        yearly_values[year]["profit"] = yearly_values[year]["revenue"] - yearly_values[year]["cost"]

    solution_rows.sort(key=lambda row: (row["year"], row["plot_id"], row["season"], row["crop_id"]))
    write_csv(
        RESULTS_DIR / "q1_baseline_solution_long.csv",
        ["scenario", "year", "season", "plot_id", "land_type", "crop_id", "crop_name", "area_mu"],
        solution_rows,
    )

    yearly_rows = []
    for year in years:
        values = yearly_values[year]
        yearly_rows.append({
            "scenario": "baseline_waste",
            "year": year,
            "revenue_yuan": round(values["revenue"], 6),
            "cost_yuan": round(values["cost"], 6),
            "profit_yuan": round(values["profit"], 6),
            "production_jin": round(values["production"], 6),
            "normal_sales_jin": round(values["normal_sales"], 6),
            "surplus_jin": round(values["surplus"], 6),
        })
    write_csv(
        RESULTS_DIR / "q1_baseline_yearly_profit.csv",
        ["scenario", "year", "revenue_yuan", "cost_yuan", "profit_yuan", "production_jin", "normal_sales_jin", "surplus_jin"],
        yearly_rows,
    )

    crop_rows = []
    for (year, crop_id, season), values in sorted(crop_summary.items()):
        if values["area_mu"] < tolerance and values["normal_sales_jin"] < tolerance:
            continue
        crop_rows.append({
            "scenario": "baseline_waste",
            "year": year,
            "season": season,
            "crop_id": crop_id,
            "crop_name": crop_by_id[crop_id]["crop_name"],
            **{key: round(value, 6) for key, value in values.items()},
        })
    write_csv(
        RESULTS_DIR / "q1_baseline_crop_summary.csv",
        ["scenario", "year", "season", "crop_id", "crop_name", "area_mu", "production_jin", "normal_sales_jin", "surplus_jin", "revenue_yuan", "cost_yuan", "profit_yuan"],
        crop_rows,
    )

    total_profit = sum(row["profit_yuan"] for row in yearly_rows)
    total_production = sum(row["production_jin"] for row in yearly_rows)
    total_surplus = sum(row["surplus_jin"] for row in yearly_rows)
    metrics = {
        "model": config["model_name"],
        "surplus_policy": "waste",
        "management_constraints": {"minimum_area": False, "maximum_plot_count": False},
        "solver": {
            "name": "HiGHS",
            "version": model.version(),
            "status": status_text,
            "runtime_seconds": model.getRunTime(),
            "mip_gap": info.mip_gap if math.isfinite(info.mip_gap) else None,
            "dual_bound": info.mip_dual_bound if math.isfinite(info.mip_dual_bound) else None,
            "node_count": info.mip_node_count,
            "time_limit_seconds": solver_config["time_limit_seconds"],
        },
        "model_size": {
            "area_variables": len(x),
            "planting_binary_variables": len(y),
            "water_mode_binary_variables": len(water_mode),
            "normal_sales_variables": len(normal_sales),
        },
        "objective": round(model.getObjectiveValue(), 6),
        "total_profit_yuan": round(total_profit, 6),
        "total_production_jin": round(total_production, 6),
        "total_surplus_jin": round(total_surplus, 6),
        "surplus_rate": round(total_surplus / total_production, 10) if total_production > tolerance else 0.0,
        "positive_area_rows": len(solution_rows),
    }
    (RESULTS_DIR / "q1_baseline_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (LOGS_DIR / "q1_baseline_solver_summary.log").write_text(
        "\n".join(
            [
                f"model={config['model_name']}",
                f"solver=HiGHS {model.version()}",
                f"status={status_text}",
                f"runtime_seconds={model.getRunTime()}",
                f"objective={model.getObjectiveValue()}",
                f"mip_gap={info.mip_gap}",
                f"dual_bound={info.mip_dual_bound}",
                f"node_count={info.mip_node_count}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
