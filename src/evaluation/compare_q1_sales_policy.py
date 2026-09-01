from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_metrics(prefix: str) -> dict[str, Any]:
    return json.loads((RESULTS_DIR / f"{prefix}_metrics.json").read_text(encoding="utf-8"))


def write_metric_comparison(waste: dict[str, Any], discount: dict[str, Any]) -> None:
    definitions = {
        "total_profit_yuan": "七年总利润",
        "total_production_jin": "七年总产量",
        "total_surplus_jin": "超过预期销量的总产量",
        "surplus_rate": "超产量占总产量比例",
        "positive_area_rows": "非零种植记录数",
        "solver_gap": "求解器最优性差距",
        "runtime_seconds": "求解用时",
        "constraint_violations": "独立审计发现的约束违反数",
    }
    values = {
        "total_profit_yuan": (waste["total_profit_yuan"], discount["total_profit_yuan"]),
        "total_production_jin": (waste["total_production_jin"], discount["total_production_jin"]),
        "total_surplus_jin": (waste["total_surplus_jin"], discount["total_surplus_jin"]),
        "surplus_rate": (waste["surplus_rate"], discount["surplus_rate"]),
        "positive_area_rows": (waste["positive_area_rows"], discount["positive_area_rows"]),
        "solver_gap": (waste["solver"]["mip_gap"], discount["solver"]["mip_gap"]),
        "runtime_seconds": (waste["solver"]["runtime_seconds"], discount["solver"]["runtime_seconds"]),
        "constraint_violations": (
            waste["constraint_violations"]["total"],
            discount["constraint_violations"]["total"],
        ),
    }
    rows = []
    for metric, (waste_value, discount_value) in values.items():
        change = float(discount_value) - float(waste_value)
        change_pct = change / float(waste_value) if abs(float(waste_value)) > 1e-12 else None
        rows.append({
            "metric": metric,
            "full_waste": round(float(waste_value), 10),
            "full_discount": round(float(discount_value), 10),
            "change": round(change, 10),
            "change_pct": "" if change_pct is None else round(change_pct, 10),
            "explanation": definitions[metric],
        })
    with (RESULTS_DIR / "q1_waste_vs_discount.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["metric", "full_waste", "full_discount", "change", "change_pct", "explanation"],
        )
        writer.writeheader()
        writer.writerows(rows)


def aggregate_crops(prefix: str) -> dict[tuple[int, str], dict[str, float | str]]:
    totals: dict[tuple[int, str], dict[str, float | str]] = defaultdict(
        lambda: {
            "area_mu": 0.0,
            "production_jin": 0.0,
            "normal_sales_jin": 0.0,
            "surplus_jin": 0.0,
            "revenue_yuan": 0.0,
            "cost_yuan": 0.0,
            "profit_yuan": 0.0,
        }
    )
    for row in read_csv(RESULTS_DIR / f"{prefix}_crop_summary.csv"):
        key = (int(row["crop_id"]), row["crop_name"])
        for field in (
            "area_mu",
            "production_jin",
            "normal_sales_jin",
            "surplus_jin",
            "revenue_yuan",
            "cost_yuan",
            "profit_yuan",
        ):
            totals[key][field] = float(totals[key][field]) + float(row[field])
    return totals


def write_crop_comparison() -> list[dict[str, Any]]:
    waste = aggregate_crops("q1_full_waste")
    discount = aggregate_crops("q1_full_discount")
    rows: list[dict[str, Any]] = []
    for crop_id, crop_name in sorted(set(waste) | set(discount)):
        waste_values = waste.get((crop_id, crop_name), {})
        discount_values = discount.get((crop_id, crop_name), {})
        row: dict[str, Any] = {"crop_id": crop_id, "crop_name": crop_name}
        for field in ("area_mu", "production_jin", "surplus_jin", "profit_yuan"):
            waste_value = float(waste_values.get(field, 0.0))
            discount_value = float(discount_values.get(field, 0.0))
            row[f"waste_{field}"] = round(waste_value, 6)
            row[f"discount_{field}"] = round(discount_value, 6)
            row[f"change_{field}"] = round(discount_value - waste_value, 6)
        rows.append(row)
    rows.sort(key=lambda row: abs(row["change_profit_yuan"]), reverse=True)
    output = RESULTS_DIR / "q1_waste_vs_discount_by_crop.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    waste = read_metrics("q1_full_waste")
    discount = read_metrics("q1_full_discount")
    write_metric_comparison(waste, discount)
    crop_rows = write_crop_comparison()
    summary = {
        "profit_increase_yuan": round(discount["total_profit_yuan"] - waste["total_profit_yuan"], 6),
        "profit_increase_rate": round(
            (discount["total_profit_yuan"] - waste["total_profit_yuan"]) / waste["total_profit_yuan"], 10
        ),
        "production_increase_jin": round(discount["total_production_jin"] - waste["total_production_jin"], 6),
        "surplus_increase_jin": round(discount["total_surplus_jin"] - waste["total_surplus_jin"], 6),
        "surplus_rate_change_percentage_points": round(
            100 * (discount["surplus_rate"] - waste["surplus_rate"]), 6
        ),
        "top_profit_change_crops": crop_rows[:8],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
