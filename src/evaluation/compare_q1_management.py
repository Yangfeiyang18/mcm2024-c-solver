from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
MINIMUM_AREA = {
    "平旱地": 5.0,
    "梯田": 5.0,
    "山坡地": 5.0,
    "水浇地": 5.0,
    "普通大棚": 0.3,
    "智慧大棚": 0.3,
}
MAX_PLOTS = 7
TOLERANCE = 1e-5


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def summarize(prefix: str) -> dict[str, float]:
    rows = read_csv(RESULTS_DIR / f"{prefix}_solution_long.csv")
    metrics = json.loads((RESULTS_DIR / f"{prefix}_metrics.json").read_text(encoding="utf-8"))
    plot_sets: dict[tuple[int, int, str], set[str]] = defaultdict(set)
    total_area = 0.0
    minimum_area_violations = 0
    minimum_area_shortfall = 0.0
    for row in rows:
        area = float(row["area_mu"])
        total_area += area
        plot_sets[(int(row["year"]), int(row["crop_id"]), row["season"])].add(row["plot_id"])
        required = MINIMUM_AREA[row["land_type"]]
        if area > TOLERANCE and area + TOLERANCE < required:
            minimum_area_violations += 1
            minimum_area_shortfall += required - area
    plot_counts = [len(plot_ids) for plot_ids in plot_sets.values()]
    return {
        "total_profit_yuan": float(metrics["total_profit_yuan"]),
        "solver_gap": float(metrics["solver"]["mip_gap"]),
        "positive_area_rows": float(len(rows)),
        "total_planted_area_mu": total_area,
        "average_area_per_positive_row_mu": total_area / len(rows),
        "surplus_rate": float(metrics["surplus_rate"]),
        "total_surplus_jin": float(metrics["total_surplus_jin"]),
        "active_crop_year_season_groups": float(len(plot_counts)),
        "average_plots_per_active_group": sum(plot_counts) / len(plot_counts),
        "maximum_plots_in_one_group": float(max(plot_counts)),
        "groups_over_7_plots": float(sum(count > MAX_PLOTS for count in plot_counts)),
        "minimum_area_violation_rows": float(minimum_area_violations),
        "minimum_area_total_shortfall_mu": minimum_area_shortfall,
        "constraint_violations": float(metrics["constraint_violations"]["total"]),
    }


def main() -> None:
    baseline = summarize("q1_baseline")
    full = summarize("q1_full_waste")
    explanations = {
        "total_profit_yuan": "加入管理约束后的利润代价",
        "solver_gap": "求解器尚未证明的相对差距",
        "positive_area_rows": "实际非零的地块-季次-作物记录数",
        "total_planted_area_mu": "七年所有季次种植面积合计",
        "average_area_per_positive_row_mu": "每条种植记录的平均面积",
        "surplus_rate": "总超产量占总产量比例",
        "total_surplus_jin": "超过预期销量的总产量",
        "active_crop_year_season_groups": "实际出现的作物-年份-季次组合数",
        "average_plots_per_active_group": "每个作物-年份-季次平均使用地块数",
        "maximum_plots_in_one_group": "单个作物-年份-季次使用地块数最大值",
        "groups_over_7_plots": "超过7块地的组合数量",
        "minimum_area_violation_rows": "低于配置最小面积的记录数量",
        "minimum_area_total_shortfall_mu": "这些小面积记录距最低要求的合计差额",
        "constraint_violations": "独立审计发现的正式模型约束违反数",
    }
    rows: list[dict[str, Any]] = []
    for metric in baseline:
        base = baseline[metric]
        current = full[metric]
        change = current - base
        change_pct = change / base if abs(base) > 1e-12 else None
        rows.append({
            "metric": metric,
            "baseline": round(base, 10),
            "full_waste": round(current, 10),
            "change": round(change, 10),
            "change_pct": "" if change_pct is None else round(change_pct, 10),
            "explanation": explanations[metric],
        })
    output = RESULTS_DIR / "q1_baseline_vs_full_waste.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "baseline", "full_waste", "change", "change_pct", "explanation"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({row["metric"]: row for row in rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
