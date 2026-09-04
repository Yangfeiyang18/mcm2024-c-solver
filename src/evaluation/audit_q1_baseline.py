from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = PROJECT_ROOT / "数据" / "清洗后数据"
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


def main() -> None:
    plots = read_csv(CLEAN_DIR / "plots.csv")
    crops = read_csv(CLEAN_DIR / "crops.csv")
    planting_2023 = read_csv(CLEAN_DIR / "planting_2023.csv")
    parameters = read_csv(CLEAN_DIR / "crop_parameters_2023.csv")
    eligibility = read_csv(CLEAN_DIR / "eligibility.csv")
    demand_rows = read_csv(CLEAN_DIR / "demand_2023.csv")
    solution = read_csv(RESULTS_DIR / "q1_baseline_solution_long.csv")
    metrics_path = RESULTS_DIR / "q1_baseline_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    plot_by_id = {row["plot_id"]: {"land_type": row["land_type"], "area": float(row["area_mu"])} for row in plots}
    legume_ids = {int(row["crop_id"]) for row in crops if int(row["is_legume"]) == 1}
    eligible_keys = {
        (row["plot_id"], row["season"], int(row["crop_id"]))
        for row in eligibility
        if int(row["eligible"]) == 1
    }
    parameter_by_key = {
        (int(row["crop_id"]), row["land_type"], row["season"]): {
            "yield": float(row["yield_jin_per_mu"]),
            "cost": float(row["cost_yuan_per_mu"]),
            "price": float(row["price_mid"]),
        }
        for row in parameters
    }
    demand = {(int(row["crop_id"]), row["season"]): float(row["expected_sales_jin"]) for row in demand_rows}

    area_by_key: dict[tuple[int, str, str, int], float] = defaultdict(float)
    for row in solution:
        key = (int(row["year"]), row["plot_id"], row["season"], int(row["crop_id"]))
        area_by_key[key] += float(row["area_mu"])

    audit_rows: list[dict[str, Any]] = []

    def record(check: str, violations: list[tuple[str, float]], unit: str, note: str) -> None:
        audit_rows.append({
            "check": check,
            "violation_count": len(violations),
            "max_violation": round(max((value for _, value in violations), default=0.0), 8),
            "unit": unit,
            "status": "PASS" if not violations else "FAIL",
            "details": note if not violations else "; ".join(key for key, _ in violations[:10]),
        })

    negative = [(str(key), -area) for key, area in area_by_key.items() if area < -TOLERANCE]
    record("面积非负", negative, "亩", "所有面积均非负")

    ineligible = [(str(key), 1.0) for key, area in area_by_key.items() if area > TOLERANCE and (key[1], key[2], key[3]) not in eligible_keys]
    record("作物适宜性", ineligible, "处", "所有正面积组合均在适宜性矩阵中")

    slot_area: dict[tuple[int, str, str], float] = defaultdict(float)
    for (year, plot_id, season, _crop_id), area in area_by_key.items():
        slot_area[(year, plot_id, season)] += area
    area_excess = []
    for key, total in slot_area.items():
        excess = total - plot_by_id[key[1]]["area"]
        if excess > TOLERANCE:
            area_excess.append((str(key), excess))
    record("地块季次面积上限", area_excess, "亩", "各地块每季面积未超过地块面积")

    water_conflicts = []
    water_second_multi = []
    for year in range(2024, 2031):
        for plot_id, plot in plot_by_id.items():
            if plot["land_type"] != "水浇地":
                continue
            rice = area_by_key[(year, plot_id, "single", 16)]
            first = sum(area for (candidate_year, candidate_plot, season, _), area in area_by_key.items() if candidate_year == year and candidate_plot == plot_id and season == "first")
            second_crop_ids = [crop_id for (candidate_year, candidate_plot, season, crop_id), area in area_by_key.items() if candidate_year == year and candidate_plot == plot_id and season == "second" and area > TOLERANCE]
            second = sum(area_by_key[(year, plot_id, "second", crop_id)] for crop_id in second_crop_ids)
            if rice > TOLERANCE and first + second > TOLERANCE:
                water_conflicts.append((f"{year}/{plot_id}", first + second))
            if len(second_crop_ids) > 1:
                water_second_multi.append((f"{year}/{plot_id}", float(len(second_crop_ids) - 1)))
    record("水浇地稻菜模式", water_conflicts, "亩", "未出现水稻与两季蔬菜并存")
    record("水浇地第二季单一作物", water_second_multi, "种", "第二季至多选择一种根菜")

    rotation = []
    for plot_id, season, crop_id in eligible_keys:
        for year in range(2024, 2030):
            if area_by_key[(year, plot_id, season, crop_id)] > TOLERANCE and area_by_key[(year + 1, plot_id, season, crop_id)] > TOLERANCE:
                rotation.append((f"{year}-{year + 1}/{plot_id}/{season}/{crop_id}", 1.0))
    record("相邻年份同季重茬", rotation, "处", "同地块同季作物未连续两年出现")

    initial_rotation = []
    for row in planting_2023:
        key = (2024, row["plot_id"], row["season"], int(row["crop_id"]))
        if area_by_key[key] > TOLERANCE:
            initial_rotation.append((f"2023-2024/{row['plot_id']}/{row['season']}/{row['crop_id']}", area_by_key[key]))
    record("2023-2024初始重茬", initial_rotation, "亩", "2024未重复2023同地块同季作物")

    smart_rotation = []
    for plot_id, plot in plot_by_id.items():
        if plot["land_type"] != "智慧大棚":
            continue
        for crop_id in range(17, 35):
            for year in range(2024, 2031):
                if area_by_key[(year, plot_id, "first", crop_id)] > TOLERANCE and area_by_key[(year, plot_id, "second", crop_id)] > TOLERANCE:
                    smart_rotation.append((f"within/{year}/{plot_id}/{crop_id}", 1.0))
            for year in range(2024, 2030):
                if area_by_key[(year, plot_id, "second", crop_id)] > TOLERANCE and area_by_key[(year + 1, plot_id, "first", crop_id)] > TOLERANCE:
                    smart_rotation.append((f"cross/{year}-{year + 1}/{plot_id}/{crop_id}", 1.0))
    for row in planting_2023:
        plot_id = row["plot_id"]
        if plot_by_id.get(plot_id, {}).get("land_type") != "智慧大棚":
            continue
        if row["season"] != "second":
            continue
        crop_id = int(row["crop_id"])
        area = area_by_key[(2024, plot_id, "first", crop_id)]
        if area > TOLERANCE:
            smart_rotation.append((f"initial_cross/2023-2024/{plot_id}/{crop_id}", area))
    record("智慧大棚连续季重茬", smart_rotation, "处", "年内、跨年及2023S2→2024S1均无同作物")

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
            shortfall = plot["area"] - total
            if shortfall > TOLERANCE:
                bean_shortfall.append((f"{start}-{start + 2}/{plot_id}", shortfall))
    record("三年豆类覆盖", bean_shortfall, "亩", "每个连续三年窗口累计豆类面积不少于地块面积")

    # 完全脱离求解器，按面积、产量、价格、成本重新核算利润。
    production: dict[tuple[int, int, str], float] = defaultdict(float)
    cost_by_year: dict[int, float] = defaultdict(float)
    for (year, plot_id, season, crop_id), area in area_by_key.items():
        plot = plot_by_id[plot_id]
        parameter = parameter_by_key[(crop_id, plot["land_type"], season)]
        production[(year, crop_id, season)] += area * parameter["yield"]
        cost_by_year[year] += area * parameter["cost"]
    revenue_by_year: dict[int, float] = defaultdict(float)
    for (year, crop_id, season), quantity in production.items():
        price = next(
            parameter["price"]
            for (candidate_crop, _land_type, candidate_season), parameter in parameter_by_key.items()
            if candidate_crop == crop_id and candidate_season == season
        )
        revenue_by_year[year] += min(quantity, demand.get((crop_id, season), 0.0)) * price
    recalculated_profit = sum(revenue_by_year[year] - cost_by_year[year] for year in range(2024, 2031))
    profit_difference = abs(recalculated_profit - float(metrics["total_profit_yuan"]))
    profit_violations = [("recalculated_profit", profit_difference)] if profit_difference > 0.1 else []
    record("利润独立重算", profit_violations, "元", f"独立重算利润={recalculated_profit:.6f}元")

    failed_checks = [row for row in audit_rows if row["status"] == "FAIL"]
    total_violations = sum(int(row["violation_count"]) for row in audit_rows)
    metrics["constraint_violations"] = {
        "total": total_violations,
        "failed_checks": len(failed_checks),
        "audit_status": "PASS" if not failed_checks else "FAIL",
    }
    metrics["recalculated_profit_yuan"] = round(recalculated_profit, 6)
    metrics["profit_recalculation_difference_yuan"] = round(profit_difference, 6)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_csv(
        RESULTS_DIR / "q1_baseline_constraint_audit.csv",
        ["check", "violation_count", "max_violation", "unit", "status", "details"],
        audit_rows,
    )
    print(json.dumps({
        "audit_status": metrics["constraint_violations"]["audit_status"],
        "total_violations": total_violations,
        "recalculated_profit_yuan": round(recalculated_profit, 6),
        "reported_profit_yuan": metrics["total_profit_yuan"],
        "difference_yuan": round(profit_difference, 6),
    }, ensure_ascii=False, indent=2))
    if failed_checks:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
