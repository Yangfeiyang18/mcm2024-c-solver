from __future__ import annotations

import argparse
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
TOLERANCE = 1e-5


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(candidate: str) -> None:
    prefix = f"q2_{candidate}"
    metrics_path = RESULTS_DIR / f"{prefix}_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    plots = read_csv(CLEAN_DIR / "plots.csv")
    crops = read_csv(CLEAN_DIR / "crops.csv")
    planting_2023 = read_csv(CLEAN_DIR / "planting_2023.csv")
    eligibility = read_csv(CLEAN_DIR / "eligibility.csv")
    solution = read_csv(RESULTS_DIR / f"{prefix}_solution_long.csv")
    reported_profit_rows = read_csv(RESULTS_DIR / f"{prefix}_training_profit.csv")
    metadata = json.loads((SCENARIO_DIR / "q2_scenario_metadata.json").read_text(encoding="utf-8"))
    scenario_data = np.load(SCENARIO_DIR / metrics["scenario_file"])

    years = [int(year) for year in metadata["years"]]
    year_index = {year: index for index, year in enumerate(years)}
    demand_keys = [(int(item[0]), str(item[1])) for item in metadata["demand_keys"]]
    parameter_keys = [(int(item[0]), str(item[1]), str(item[2])) for item in metadata["parameter_keys"]]
    price_keys = [(int(item[0]), str(item[1])) for item in metadata["price_keys"]]
    demand_index = {key: index for index, key in enumerate(demand_keys)}
    parameter_index = {key: index for index, key in enumerate(parameter_keys)}
    price_index = {key: index for index, key in enumerate(price_keys)}
    demand_scenarios = scenario_data["demand"]
    yield_scenarios = scenario_data["yield"]
    cost_scenarios = scenario_data["cost"]
    price_scenarios = scenario_data["price"]

    plot_by_id = {
        row["plot_id"]: {"land_type": row["land_type"], "area": float(row["area_mu"])} for row in plots
    }
    legume_ids = {int(row["crop_id"]) for row in crops if int(row["is_legume"]) == 1}
    eligible_keys = {
        (row["plot_id"], row["season"], int(row["crop_id"]))
        for row in eligibility if int(row["eligible"]) == 1
    }
    area_by_key: dict[tuple[int, str, str, int], float] = defaultdict(float)
    for row in solution:
        area_by_key[(int(row["year"]), row["plot_id"], row["season"], int(row["crop_id"]))] += float(row["area_mu"])

    audit_rows: list[dict[str, Any]] = []

    def record(check: str, violations: list[tuple[str, float]], unit: str, pass_note: str) -> None:
        audit_rows.append({
            "check": check,
            "violation_count": len(violations),
            "max_violation": round(max((value for _, value in violations), default=0.0), 8),
            "unit": unit,
            "status": "PASS" if not violations else "FAIL",
            "details": pass_note if not violations else "; ".join(key for key, _ in violations[:10]),
        })

    record("面积非负", [(str(k), -v) for k, v in area_by_key.items() if v < -TOLERANCE], "亩", "所有面积均非负")
    record(
        "作物适宜性",
        [(str(k), 1.0) for k, v in area_by_key.items() if v > TOLERANCE and (k[1], k[2], k[3]) not in eligible_keys],
        "处", "所有正面积组合均适宜",
    )

    slot_area: dict[tuple[int, str, str], float] = defaultdict(float)
    for (year, plot_id, season, _crop_id), area in area_by_key.items():
        slot_area[(year, plot_id, season)] += area
    record(
        "地块季次面积上限",
        [(str(k), v - plot_by_id[k[1]]["area"]) for k, v in slot_area.items() if v - plot_by_id[k[1]]["area"] > TOLERANCE],
        "亩", "各地块每季面积不超限",
    )

    water_conflicts: list[tuple[str, float]] = []
    water_second_multi: list[tuple[str, float]] = []
    for year in years:
        for plot_id, plot in plot_by_id.items():
            if plot["land_type"] != "水浇地":
                continue
            rice = area_by_key[(year, plot_id, "single", 16)]
            first = sum(v for (y, p, s, _), v in area_by_key.items() if y == year and p == plot_id and s == "first")
            second_ids = [c for (y, p, s, c), v in area_by_key.items() if y == year and p == plot_id and s == "second" and v > TOLERANCE]
            second = sum(area_by_key[(year, plot_id, "second", crop_id)] for crop_id in second_ids)
            if rice > TOLERANCE and first + second > TOLERANCE:
                water_conflicts.append((f"{year}/{plot_id}", first + second))
            if len(second_ids) > 1:
                water_second_multi.append((f"{year}/{plot_id}", float(len(second_ids) - 1)))
    record("水浇地稻菜模式", water_conflicts, "亩", "水稻与两季蔬菜不并存")
    record("水浇地第二季单一作物", water_second_multi, "种", "第二季至多一种根菜")

    rotation: list[tuple[str, float]] = []
    for plot_id, season, crop_id in eligible_keys:
        for year in years[:-1]:
            if area_by_key[(year, plot_id, season, crop_id)] > TOLERANCE and area_by_key[(year + 1, plot_id, season, crop_id)] > TOLERANCE:
                rotation.append((f"{year}-{year + 1}/{plot_id}/{season}/{crop_id}", 1.0))
    record("相邻年份同季重茬", rotation, "处", "无相邻年份同季重茬")

    initial = []
    for row in planting_2023:
        key = (2024, row["plot_id"], row["season"], int(row["crop_id"]))
        if area_by_key[key] > TOLERANCE:
            initial.append((f"2023-2024/{key[1]}/{key[2]}/{key[3]}", area_by_key[key]))
    record("2023-2024初始重茬", initial, "亩", "2024未重复2023同地块同季作物")

    smart_rotation: list[tuple[str, float]] = []
    for plot_id, plot in plot_by_id.items():
        if plot["land_type"] != "智慧大棚":
            continue
        for crop_id in range(17, 35):
            for year in years:
                if area_by_key[(year, plot_id, "first", crop_id)] > TOLERANCE and area_by_key[(year, plot_id, "second", crop_id)] > TOLERANCE:
                    smart_rotation.append((f"within/{year}/{plot_id}/{crop_id}", 1.0))
            for year in years[:-1]:
                if area_by_key[(year, plot_id, "second", crop_id)] > TOLERANCE and area_by_key[(year + 1, plot_id, "first", crop_id)] > TOLERANCE:
                    smart_rotation.append((f"cross/{year}/{plot_id}/{crop_id}", 1.0))
    record("智慧大棚连续季重茬", smart_rotation, "处", "智慧大棚年内和跨年均无连续重茬")

    min_area = {k: float(v) for k, v in metrics["management_constraints"]["minimum_area_mu_by_land_type"].items()}
    minimum_violations = []
    for (year, plot_id, season, crop_id), area in area_by_key.items():
        required = min_area[plot_by_id[plot_id]["land_type"]]
        if area > TOLERANCE and area + TOLERANCE < required:
            minimum_violations.append((f"{year}/{plot_id}/{season}/{crop_id}", required - area))
    record("最小种植面积", minimum_violations, "亩", "所有启用作物达到最小面积")

    max_plots = int(metrics["management_constraints"]["max_plots_per_crop_per_year_season"])
    plot_sets: dict[tuple[int, int, str], set[str]] = defaultdict(set)
    for (year, plot_id, season, crop_id), area in area_by_key.items():
        if area > TOLERANCE:
            plot_sets[(year, crop_id, season)].add(plot_id)
    record(
        "每作物最大地块数",
        [(str(key), float(len(ids) - max_plots)) for key, ids in plot_sets.items() if len(ids) > max_plots],
        "块", "每作物每年每季不超过7块地",
    )

    bean_2023: dict[str, float] = defaultdict(float)
    for row in planting_2023:
        if int(row["crop_id"]) in legume_ids:
            bean_2023[row["plot_id"]] += float(row["area_mu"])
    bean_by_year_plot: dict[tuple[int, str], float] = defaultdict(float)
    for (year, plot_id, _season, crop_id), area in area_by_key.items():
        if crop_id in legume_ids:
            bean_by_year_plot[(year, plot_id)] += area
    bean_shortfall = []
    for plot_id, plot in plot_by_id.items():
        for start in range(2023, 2029):
            total = bean_2023[plot_id] if start == 2023 else 0.0
            total += sum(bean_by_year_plot[(year, plot_id)] for year in range(max(2024, start), start + 3))
            if plot["area"] - total > TOLERANCE:
                bean_shortfall.append((f"{start}-{start + 2}/{plot_id}", plot["area"] - total))
    record("三年豆类覆盖", bean_shortfall, "亩", "每个三年窗口豆类累计面积达标")

    # Independent scenario-profit recalculation from the exported area plan.
    scenario_count = int(demand_scenarios.shape[0])
    recalculated = np.zeros(scenario_count, dtype=np.float64)
    crop_seasons = sorted({(crop_id, season) for _plot_id, season, crop_id in eligible_keys})
    for scenario in range(scenario_count):
        revenue = 0.0
        cost = 0.0
        for year in years:
            yi = year_index[year]
            production: dict[tuple[int, str], float] = defaultdict(float)
            for (candidate_year, plot_id, season, crop_id), area in area_by_key.items():
                if candidate_year != year or area <= TOLERANCE:
                    continue
                pindex = parameter_index[(crop_id, plot_by_id[plot_id]["land_type"], season)]
                production[(crop_id, season)] += area * float(yield_scenarios[scenario, yi, pindex])
                cost += area * float(cost_scenarios[scenario, yi, pindex])
            for crop_id, season in crop_seasons:
                dindex = demand_index.get((crop_id, season))
                demand = 0.0 if dindex is None else float(demand_scenarios[scenario, yi, dindex])
                sold = min(production[(crop_id, season)], demand)
                revenue += sold * float(price_scenarios[scenario, yi, price_index[(crop_id, season)]])
        recalculated[scenario] = revenue - cost

    reported = np.asarray([float(row["profit_yuan"]) for row in reported_profit_rows], dtype=np.float64)
    profit_differences = np.abs(recalculated - reported)
    record(
        "训练情景利润独立重算",
        [(f"scenario_{index}", float(diff)) for index, diff in enumerate(profit_differences) if diff > 1.0],
        "元", f"最大重算差异={float(np.max(profit_differences)):.6f}元",
    )

    failed = [row for row in audit_rows if row["status"] == "FAIL"]
    total_violations = sum(int(row["violation_count"]) for row in audit_rows)
    metrics["constraint_violations"] = {
        "total": total_violations,
        "failed_checks": len(failed),
        "audit_status": "PASS" if not failed else "FAIL",
    }
    metrics["profit_recalculation_max_difference_yuan"] = round(float(np.max(profit_differences)), 6)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(
        RESULTS_DIR / f"{prefix}_constraint_audit.csv",
        ["check", "violation_count", "max_violation", "unit", "status", "details"],
        audit_rows,
    )
    print(json.dumps({
        "candidate": candidate,
        "audit_status": metrics["constraint_violations"]["audit_status"],
        "total_violations": total_violations,
        "profit_recalculation_max_difference_yuan": round(float(np.max(profit_differences)), 6),
    }, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", choices=["mean_value", "lambda_0", "lambda_025", "lambda_050"])
    args = parser.parse_args()
    main(args.candidate)
