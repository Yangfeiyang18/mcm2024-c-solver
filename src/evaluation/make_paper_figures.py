from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from official_io import FIGURES_DIR, LOCKED, PROJECT_ROOT, RESULTS_DIR, read_csv, write_csv, write_json


def setup_font() -> None:
    candidates = [
        PROJECT_ROOT / "figures" / "fonts" / "SimHei.ttf",
        PROJECT_ROOT / "figures" / "fonts" / "NotoSansSC-Regular.otf",
        Path("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            name = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def savefig(name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def plot_q1_profit() -> None:
    waste = read_csv(RESULTS_DIR / "q1_full_waste_yearly_profit.csv")
    discount = read_csv(RESULTS_DIR / "q1_full_discount_yearly_profit.csv")
    years = [int(row["year"]) for row in waste]
    waste_wan = [float(row["profit_yuan"]) / 10000 for row in waste]
    discount_wan = [float(row["profit_yuan"]) / 10000 for row in discount]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    axes[0].bar(["滞销浪费", "超额半价"], [LOCKED["q1_waste_profit_wan"], LOCKED["q1_discount_profit_wan"]], color=["#4C78A8", "#F58518"])
    axes[0].set_ylabel("七年总利润（万元）")
    axes[0].set_title("问题1：两种销售策略总利润")
    for index, value in enumerate([LOCKED["q1_waste_profit_wan"], LOCKED["q1_discount_profit_wan"]]):
        axes[0].text(index, value + 40, f"{value:.2f}", ha="center", va="bottom")
    axes[1].plot(years, waste_wan, marker="o", label="滞销浪费")
    axes[1].plot(years, discount_wan, marker="o", label="超额半价")
    axes[1].set_xlabel("年份")
    axes[1].set_ylabel("年利润（万元）")
    axes[1].set_title("问题1：分年利润")
    axes[1].legend()
    savefig("q1_profit_comparison.png")


def plot_q1_surplus() -> None:
    waste: dict[str, float] = defaultdict(float)
    discount: dict[str, float] = defaultdict(float)
    for row in read_csv(RESULTS_DIR / "q1_waste_vs_discount_by_crop.csv"):
        waste[row["crop_name"]] += float(row["waste_surplus_jin"])
        discount[row["crop_name"]] += float(row["discount_surplus_jin"])
    names = sorted(set(waste) | set(discount), key=lambda name: discount[name] - waste[name], reverse=True)[:12]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    index = range(len(names))
    ax.bar([i - 0.18 for i in index], [waste[name] / 10000 for name in names], width=0.36, label="滞销浪费")
    ax.bar([i + 0.18 for i in index], [discount[name] / 10000 for name in names], width=0.36, label="超额半价")
    ax.set_xticks(list(index), names, rotation=35, ha="right")
    ax.set_ylabel("七年超额产量（万斤）")
    ax.set_title("问题1：主要作物超额产量对比")
    ax.legend()
    savefig("q1_surplus_by_crop.png")


def plot_q2_table() -> None:
    rows = [
        ["口径", "平均利润（万元）", "标准差（万元）", "下尾CVaR（万元）"],
        ["100训练情景", "4091.90", "18.44", "4060.86"],
        ["2000独立测试", "4092.17", "17.81", "4060.84"],
        ["2000弱相关测试", "4087.15", "19.73", "4051.93"],
        ["2000中相关测试", "4085.93", "23.22", "4044.93"],
    ]
    fig, ax = plt.subplots(figsize=(9.6, 2.8))
    ax.axis("off")
    table = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 1.55)
    ax.set_title("问题2锁定方案（λ=0.25）风险指标")
    savefig("q2_risk_table.png")
    write_csv(
        RESULTS_DIR / "final" / "q2_locked_risk_table.csv",
        ["setting", "mean_wan", "std_wan", "cvar_wan", "note"],
        [
            {"setting": "train_100", "mean_wan": 4091.90, "std_wan": 18.44, "cvar_wan": 4060.86, "note": "优化情景，选λ用"},
            {"setting": "test_2000_independent", "mean_wan": 4092.17, "std_wan": 17.81, "cvar_wan": 4060.84, "note": "问题2正式样本外"},
            {"setting": "test_2000_weak_corr", "mean_wan": 4087.15, "std_wan": 19.73, "cvar_wan": 4051.93, "note": "同一Q2方案，问题3弱相关世界"},
            {"setting": "test_2000_medium_corr", "mean_wan": 4085.93, "std_wan": 23.22, "cvar_wan": 4044.93, "note": "同一Q2方案，问题3中相关世界"},
        ],
    )


def plot_q3_risk() -> None:
    labels = ["独立", "弱相关", "中相关"]
    mean = [4092.17, 4087.15, 4085.93]
    std = [17.81, 19.73, 23.22]
    cvar = [4060.84, 4051.93, 4044.93]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0))
    series = [
        (axes[0], mean, "平均利润（万元）", "#4C78A8"),
        (axes[1], std, "标准差（万元）", "#F58518"),
        (axes[2], cvar, "下尾CVaR（万元）", "#54A24B"),
    ]
    for ax, values, title, color in series:
        ax.plot(labels, values, marker="o", color=color, linewidth=2)
        ax.set_title(title)
        for x, y in zip(labels, values):
            ax.text(x, y, f"{y:.2f}", ha="center", va="bottom")
    fig.suptitle("同一Q2方案在2000情景下的风险递进")
    savefig("q3_risk_comparison.png")


def plot_area_structure() -> None:
    totals: dict[str, float] = defaultdict(float)
    for row in read_csv(RESULTS_DIR / "q2_lambda_025_solution_long.csv"):
        totals[row["crop_name"]] += float(row["area_mu"])
    names = sorted(totals, key=totals.get, reverse=True)[:15]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(names, [totals[name] for name in names], color="#4C78A8")
    ax.set_ylabel("七年累计种植面积（亩）")
    ax.set_title("问题2正式方案：主要作物面积结构")
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    savefig("q2_crop_area_structure.png")


def write_q3_test2000_correlation() -> None:
    wanted = {
        "q3_weak_test_2000": "weak",
        "q3_medium_test_2000": "medium",
    }
    keep = {"wheat_corn_demand_corr", "mean_yield_price_corr", "mean_cost_corr_first8"}
    rows = []
    for row in read_csv(RESULTS_DIR / "q3_marginal_correlation_audit.csv"):
        if row["scenario_set"] in wanted and row["check"] in keep:
            rows.append({
                "sample": "test_2000",
                "strength": wanted[row["scenario_set"]],
                "check": row["check"],
                "value": float(row["value"]),
                "note": row["note"] + "；论文只用2000测试情景，不用100训练情景",
            })
    write_csv(
        RESULTS_DIR / "final" / "q3_test2000_empirical_correlation.csv",
        ["sample", "strength", "check", "value", "note"],
        rows,
    )
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    checks = ["wheat_corn_demand_corr", "mean_yield_price_corr", "mean_cost_corr_first8"]
    labels = ["小麦–玉米需求", "产量–价格", "成本共同波动"]
    weak = {row["check"]: row["value"] for row in rows if row["strength"] == "weak"}
    medium = {row["check"]: row["value"] for row in rows if row["strength"] == "medium"}
    index = range(len(checks))
    ax.bar([i - 0.18 for i in index], [weak[name] for name in checks], width=0.36, label="弱相关 2000测试")
    ax.bar([i + 0.18 for i in index], [medium[name] for name in checks], width=0.36, label="中相关 2000测试")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(list(index), labels)
    ax.set_ylabel("经验相关系数")
    ax.set_title("问题3：2000测试情景经验相关（不是因子权重0.30/0.60）")
    ax.legend()
    savefig("q3_test2000_correlation.png")


def main() -> None:
    setup_font()
    plot_q1_profit()
    plot_q1_surplus()
    plot_q2_table()
    plot_q3_risk()
    plot_area_structure()
    write_q3_test2000_correlation()
    write_json(
        RESULTS_DIR / "final" / "paper_numbers.json",
        {
            "source_version": "2026-09-04 locked closeout",
            "use_only_from": ["results/final", "outputs/final", "figures/final"],
            "numbers": {
                "q1_waste_profit_wan": LOCKED["q1_waste_profit_wan"],
                "q1_discount_profit_wan": LOCKED["q1_discount_profit_wan"],
                "q1_waste_gap_pct": 1.04,
                "q1_discount_gap_pct": 0.77,
                "q2_lambda": 0.25,
                "q2_train_mean_wan": 4091.90,
                "q2_train_cvar_wan": 4060.86,
                "q2_gap_pct": 2.51,
                "q3_test2000_mean_wan": {"independent": 4092.17, "weak": 4087.15, "medium": 4085.93},
                "q3_test2000_std_wan": {"independent": 17.81, "weak": 19.73, "medium": 23.22},
                "q3_test2000_cvar_wan": {"independent": 4060.84, "weak": 4051.93, "medium": 4044.93},
                "q3_test2000_wheat_corn_demand_corr": {"weak": 0.082407, "medium": 0.371002},
                "q3_test2000_yield_price_corr": {"weak": -0.046752, "medium": -0.208393},
            },
            "do_not_use": {
                "old_q1_waste_wan": 4075.77,
                "old_q1_discount_wan": 5838.10,
                "q3_train100_wheat_corn_corr": {"weak": 0.159475, "medium": 0.361927},
                "factor_weight_is_not_correlation": {"weak": 0.30, "medium": 0.60},
            },
        },
    )
    print(json.dumps({"figures": sorted(path.name for path in FIGURES_DIR.glob("*.png"))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
