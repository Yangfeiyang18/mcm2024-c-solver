from __future__ import annotations

import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_q2_scenarios import base_arrays, build_key_data  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = PROJECT_ROOT / "数据" / "情景数据"
RESULTS_DIR = PROJECT_ROOT / "results"
Q2_CONFIG_PATH = PROJECT_ROOT / "config" / "q2.json"
Q3_CONFIG_PATH = PROJECT_ROOT / "config" / "q3.json"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


_ERF = np.vectorize(math.erf, otypes=[float])


def phi(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _ERF(np.asarray(z, dtype=np.float64) / math.sqrt(2.0)))


def uniform_ppf(u: np.ndarray, low: float, high: float) -> np.ndarray:
    return low + (high - low) * np.clip(u, 0.0, 1.0)


def triangular_ppf(u: np.ndarray, left: float, mode: float, right: float) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    fc = (mode - left) / (right - left)
    out = np.empty_like(u)
    lower = u < fc
    out[lower] = left + np.sqrt(u[lower] * (right - left) * (mode - left))
    out[~lower] = right - np.sqrt((1.0 - u[~lower]) * (right - left) * (right - mode))
    return out


def combine_latent(
    common: np.ndarray,
    idiosyncratic: np.ndarray,
    common_weight: float,
    idio_weight: float,
) -> np.ndarray:
    return common_weight * common + idio_weight * idiosyncratic


def generate_correlated_set(
    count: int,
    rng: np.random.Generator,
    years: list[int],
    demand_keys: list[tuple[int, str]],
    parameter_keys: list[tuple[int, str, str]],
    price_keys: list[tuple[int, str]],
    categories: dict[int, str],
    bases: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    q2_uncertainty: dict[str, Any],
    q3_config: dict[str, Any],
) -> dict[str, np.ndarray]:
    base_demand, base_yield, base_cost, base_price = bases
    year_count = len(years)
    group_names = list(q3_config["crop_groups"])
    group_index = {name: index for index, name in enumerate(group_names)}
    crop_group: dict[int, int] = {}
    for name, crop_ids in q3_config["crop_groups"].items():
        for crop_id in crop_ids:
            crop_group[int(crop_id)] = group_index[name]
    missing = sorted({crop_id for crop_id, _season in demand_keys} - set(crop_group))
    if missing:
        raise KeyError(f"作物未分配到问题3类群：{missing}")

    strength = q3_config["correlation_strength"]
    common_w = float(q3_config["factor_weights"][strength]["common"])
    idio_w = float(q3_config["factor_weights"][strength]["idiosyncratic"])
    price_loads = q3_config["price_factor_loadings"]
    price_scale = float(np.sqrt(
        price_loads["market"] ** 2 + price_loads["cost_passthrough"] ** 2 + price_loads["climate"] ** 2
    ))
    morel_id = int(q3_config["morel_crop_id"])

    wheat_corn_bounds = q2_uncertainty["wheat_corn_demand_annual_growth_uniform"]
    other_demand_bounds = q2_uncertainty["other_crop_demand_relative_to_2023_uniform"]
    yield_bounds = q2_uncertainty["yield_relative_to_2023_uniform"]
    cost_left, cost_mode, cost_right = q2_uncertainty["cost_annual_growth_triangular"]
    grain_bounds = q2_uncertainty["grain_price_relative_to_2023_uniform"]
    veg_left, veg_mode, veg_right = q2_uncertainty["vegetable_price_annual_growth_triangular"]
    fungus_bounds = q2_uncertainty["ordinary_fungus_price_annual_decline_uniform"]
    morel_decline = float(q2_uncertainty["morel_price_annual_decline_fixed"])

    market = rng.standard_normal((count, year_count, len(group_names)))
    climate = rng.standard_normal((count, year_count, len(group_names)))
    cost_factor = rng.standard_normal((count, year_count))

    demand = np.empty((count, year_count, len(demand_keys)), dtype=np.float64)
    yields = np.empty((count, year_count, len(parameter_keys)), dtype=np.float64)
    costs = np.empty((count, year_count, len(parameter_keys)), dtype=np.float64)
    prices = np.empty((count, year_count, len(price_keys)), dtype=np.float64)

    for key_index, (crop_id, _season) in enumerate(demand_keys):
        group = crop_group[crop_id]
        latent = combine_latent(
            market[:, :, group],
            rng.standard_normal((count, year_count)),
            common_w,
            idio_w,
        )
        uniforms = phi(latent)
        if crop_id in (6, 7):
            running = np.full(count, base_demand[key_index], dtype=np.float64)
            for year_index in range(year_count):
                growth = uniform_ppf(uniforms[:, year_index], *wheat_corn_bounds)
                running = running * (1.0 + growth)
                demand[:, year_index, key_index] = running
        else:
            shocks = uniform_ppf(uniforms, *other_demand_bounds)
            demand[:, :, key_index] = base_demand[key_index] * (1.0 + shocks)

    crop_yield_noise: dict[int, np.ndarray] = {}
    for crop_id in {crop_id for crop_id, _land, _season in parameter_keys}:
        crop_yield_noise[crop_id] = rng.standard_normal((count, year_count))
    for key_index, (crop_id, _land_type, _season) in enumerate(parameter_keys):
        group = crop_group[crop_id]
        latent = combine_latent(
            climate[:, :, group],
            crop_yield_noise[crop_id],
            common_w,
            idio_w,
        )
        shocks = uniform_ppf(phi(latent), *yield_bounds)
        yields[:, :, key_index] = base_yield[key_index] * (1.0 + shocks)

    running_cost = np.broadcast_to(base_cost, (count, len(parameter_keys))).copy()
    cost_noise = rng.standard_normal((count, year_count, len(parameter_keys)))
    for year_index in range(year_count):
        latent = combine_latent(
            np.repeat(cost_factor[:, year_index][:, None], len(parameter_keys), axis=1),
            cost_noise[:, year_index, :],
            common_w,
            idio_w,
        )
        growth = triangular_ppf(phi(latent), cost_left, cost_mode, cost_right)
        running_cost *= 1.0 + growth
        costs[:, year_index, :] = running_cost

    for key_index, (crop_id, _season) in enumerate(price_keys):
        if crop_id == morel_id:
            for year_index in range(year_count):
                prices[:, year_index, key_index] = base_price[key_index] * (1.0 - morel_decline) ** (year_index + 1)
            continue
        group = crop_group[crop_id]
        common = (
            price_loads["market"] * market[:, :, group]
            + price_loads["cost_passthrough"] * cost_factor
            + price_loads["climate"] * climate[:, :, group]
        ) / price_scale
        latent = combine_latent(common, rng.standard_normal((count, year_count)), common_w, idio_w)
        uniforms = phi(latent)
        category = categories[crop_id]
        if category == "grain":
            shocks = uniform_ppf(uniforms, *grain_bounds)
            prices[:, :, key_index] = base_price[key_index] * (1.0 + shocks)
        elif category == "vegetable":
            running = np.full(count, base_price[key_index], dtype=np.float64)
            for year_index in range(year_count):
                growth = triangular_ppf(uniforms[:, year_index], veg_left, veg_mode, veg_right)
                running *= 1.0 + growth
                prices[:, year_index, key_index] = running
        else:
            running = np.full(count, base_price[key_index], dtype=np.float64)
            for year_index in range(year_count):
                decline = uniform_ppf(uniforms[:, year_index], *fungus_bounds)
                running *= 1.0 - decline
                prices[:, year_index, key_index] = running

    return {"demand": demand, "yield": yields, "cost": costs, "price": prices}


def crop_relative_price_change(
    prices: np.ndarray,
    price_keys: list[tuple[int, str]],
    base_price: np.ndarray,
) -> dict[int, np.ndarray]:
    by_crop: dict[int, list[np.ndarray]] = {}
    for index, (crop_id, _season) in enumerate(price_keys):
        relative = prices[:, :, index] / base_price[index] - 1.0
        by_crop.setdefault(crop_id, []).append(relative)
    return {crop_id: np.mean(np.stack(values, axis=0), axis=0) for crop_id, values in by_crop.items()}


def elasticity_matrix(q3_config: dict[str, Any]) -> dict[int, dict[int, float]]:
    matrix: dict[int, dict[int, float]] = {}
    substitute = float(q3_config["substitute_elasticity"])
    for group_name in q3_config["substitute_groups"]:
        crop_ids = [int(crop_id) for crop_id in q3_config["crop_groups"][group_name]]
        for crop_i in crop_ids:
            for crop_j in crop_ids:
                if crop_i == crop_j:
                    continue
                matrix.setdefault(crop_i, {})[crop_j] = substitute
    for pair in q3_config.get("extra_substitute_pairs", []):
        left, right = (int(value) for value in pair["crop_ids"])
        elasticity = (
            substitute
            if pair.get("use_main_substitute_elasticity", True)
            else float(pair.get("elasticity", substitute))
        )
        matrix.setdefault(left, {})[right] = elasticity
        matrix.setdefault(right, {})[left] = elasticity
    for pair in q3_config.get("complement_pairs", []):
        left, right = (int(value) for value in pair["crop_ids"])
        elasticity = float(pair["elasticity"])
        matrix.setdefault(left, {})[right] = elasticity
        matrix.setdefault(right, {})[left] = elasticity
    return {crop_i: {crop_j: value for crop_j, value in others.items() if abs(value) > 1e-12} for crop_i, others in matrix.items()}


def config_with_strength(q3_config: dict[str, Any], strength: str) -> dict[str, Any]:
    config = copy.deepcopy(q3_config)
    config["correlation_strength"] = strength
    return config


def config_with_elasticity(q3_config: dict[str, Any], substitute: float) -> dict[str, Any]:
    config = copy.deepcopy(q3_config)
    config["substitute_elasticity"] = float(substitute)
    if abs(substitute) < 1e-12:
        for pair in config.get("complement_pairs", []):
            pair["elasticity"] = 0.0
        for pair in config.get("extra_substitute_pairs", []):
            pair["elasticity"] = 0.0
            pair["use_main_substitute_elasticity"] = False
    return config


def apply_elasticity(
    arrays: dict[str, np.ndarray],
    demand_keys: list[tuple[int, str]],
    price_keys: list[tuple[int, str]],
    base_price: np.ndarray,
    q3_config: dict[str, Any],
) -> dict[str, np.ndarray]:
    clip_low, clip_high = q3_config["demand_adjustment_clip"]
    matrix = elasticity_matrix(q3_config)
    relative_price = crop_relative_price_change(arrays["price"], price_keys, base_price)
    adjusted = arrays["demand"].copy()
    for demand_index, (crop_id, _season) in enumerate(demand_keys):
        if crop_id not in matrix:
            continue
        adjustment = np.ones(adjusted.shape[:2], dtype=np.float64)
        for other_id, elasticity in matrix[crop_id].items():
            if other_id not in relative_price:
                continue
            adjustment = adjustment + elasticity * relative_price[other_id]
        adjustment = np.clip(adjustment, clip_low, clip_high)
        adjusted[:, :, demand_index] = arrays["demand"][:, :, demand_index] * adjustment
    return {**arrays, "demand": adjusted}


def flatten_year(values: np.ndarray, year_index: int) -> np.ndarray:
    return values[:, year_index, :].reshape(-1)


def summarize_set(name: str, arrays: dict[str, np.ndarray], years: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parameter_name, values in arrays.items():
        for year_index, year in enumerate(years):
            flattened = flatten_year(values, year_index)
            rows.append({
                "scenario_set": name,
                "parameter": parameter_name,
                "year": year,
                "minimum": round(float(np.min(flattened)), 8),
                "p05": round(float(np.quantile(flattened, 0.05)), 8),
                "mean": round(float(np.mean(flattened)), 8),
                "p95": round(float(np.quantile(flattened, 0.95)), 8),
                "maximum": round(float(np.max(flattened)), 8),
            })
    return rows


def mean_offdiag_corr(matrix: np.ndarray) -> float:
    if matrix.shape[0] < 2:
        return float("nan")
    corr = np.corrcoef(matrix)
    n = corr.shape[0]
    return float((np.sum(corr) - n) / (n * (n - 1)))


def correlation_audit_rows(
    arrays: dict[str, np.ndarray],
    demand_keys: list[tuple[int, str]],
    parameter_keys: list[tuple[int, str, str]],
    price_keys: list[tuple[int, str]],
    q3_config: dict[str, Any],
    set_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    year0 = 0
    group_demand: dict[str, list[np.ndarray]] = {name: [] for name in q3_config["crop_groups"]}
    crop_to_group = {
        int(crop_id): group_name
        for group_name, crop_ids in q3_config["crop_groups"].items()
        for crop_id in crop_ids
    }
    for index, (crop_id, _season) in enumerate(demand_keys):
        group_demand[crop_to_group[crop_id]].append(arrays["demand"][:, year0, index])
    for group_name, series in group_demand.items():
        if len(series) < 2:
            continue
        stacked = np.vstack(series)
        rows.append({
            "scenario_set": set_name,
            "check": f"within_group_demand_corr:{group_name}",
            "value": round(mean_offdiag_corr(stacked), 6),
            "expected_direction": "positive",
            "note": "公共因子权重不是该经验相关系数",
        })

    price_by_crop: dict[int, np.ndarray] = {}
    for index, (crop_id, _season) in enumerate(price_keys):
        price_by_crop.setdefault(crop_id, arrays["price"][:, year0, index])
    yield_by_crop: dict[int, np.ndarray] = {}
    for index, (crop_id, _land, _season) in enumerate(parameter_keys):
        yield_by_crop.setdefault(crop_id, arrays["yield"][:, year0, index])
    yield_price = []
    for crop_id, yield_series in yield_by_crop.items():
        if crop_id == int(q3_config["morel_crop_id"]) or crop_id not in price_by_crop:
            continue
        yield_price.append(float(np.corrcoef(yield_series, price_by_crop[crop_id])[0, 1]))
    rows.append({
        "scenario_set": set_name,
        "check": "mean_yield_price_corr",
        "value": round(float(np.nanmean(yield_price)), 6),
        "expected_direction": "negative",
        "note": "丰产压价机制",
    })

    cost_series = [arrays["cost"][:, year0, index] for index in range(min(8, arrays["cost"].shape[2]))]
    rows.append({
        "scenario_set": set_name,
        "check": "mean_cost_corr_first8",
        "value": round(mean_offdiag_corr(np.vstack(cost_series)), 6),
        "expected_direction": "positive",
        "note": "共同投入成本因子",
    })
    wheat = next(index for index, key in enumerate(demand_keys) if key[0] == 6)
    corn = next(index for index, key in enumerate(demand_keys) if key[0] == 7)
    rows.append({
        "scenario_set": set_name,
        "check": "wheat_corn_demand_corr",
        "value": round(float(np.corrcoef(arrays["demand"][:, year0, wheat], arrays["demand"][:, year0, corn])[0, 1]), 6),
        "expected_direction": "positive",
        "note": "主要粮食替代组共同市场因子",
    })
    return rows


def main() -> None:
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    q2 = json.loads(Q2_CONFIG_PATH.read_text(encoding="utf-8"))
    q3 = json.loads(Q3_CONFIG_PATH.read_text(encoding="utf-8"))
    years = [int(year) for year in q2["years"]]
    demand_keys, parameter_keys, price_keys, categories = build_key_data()
    bases = base_arrays(demand_keys, parameter_keys, price_keys)
    streams = np.random.SeedSequence(int(q3["random_seed"])).spawn(8)
    counts = q2["scenario_counts"]
    weak_cfg = config_with_strength(q3, "weak")
    medium_cfg = config_with_strength(q3, "medium")
    optimization = generate_correlated_set(
        int(counts["optimization"]), np.random.default_rng(streams[4]), years,
        demand_keys, parameter_keys, price_keys, categories, bases, q2["uncertainty"], weak_cfg,
    )
    test = generate_correlated_set(
        int(counts["out_of_sample_test"]), np.random.default_rng(streams[5]), years,
        demand_keys, parameter_keys, price_keys, categories, bases, q2["uncertainty"], weak_cfg,
    )
    medium_optimization = generate_correlated_set(
        int(counts["optimization"]), np.random.default_rng(streams[6]), years,
        demand_keys, parameter_keys, price_keys, categories, bases, q2["uncertainty"], medium_cfg,
    )
    medium_test = generate_correlated_set(
        int(counts["out_of_sample_test"]), np.random.default_rng(streams[7]), years,
        demand_keys, parameter_keys, price_keys, categories, bases, q2["uncertainty"], medium_cfg,
    )
    zero_cfg = config_with_elasticity(q3, 0.0)
    opt_zero = apply_elasticity(optimization, demand_keys, price_keys, bases[3], zero_cfg)
    epsilon0_max_diff = float(np.max(np.abs(opt_zero["demand"] - optimization["demand"])))
    (RESULTS_DIR / "q3_epsilon0_check.json").write_text(
        json.dumps({
            "check": "Q3B_epsilon_0_demand_equals_Q3A",
            "max_abs_demand_difference": epsilon0_max_diff,
            "status": "PASS" if epsilon0_max_diff < 1e-10 else "FAIL",
        }, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    if epsilon0_max_diff >= 1e-10:
        raise RuntimeError(f"ε=0 未还原 Q3-A 需求，最大差={epsilon0_max_diff}")

    small_cfg = config_with_elasticity(q3, float(q3["substitute_elasticity"]))
    sets = {
        "q3_weak_optimization_100": optimization,
        "q3_weak_test_2000": test,
        "q3_weak_elasticity_optimization_100": apply_elasticity(optimization, demand_keys, price_keys, bases[3], small_cfg),
        "q3_weak_elasticity_test_2000": apply_elasticity(test, demand_keys, price_keys, bases[3], small_cfg),
        "q3_medium_optimization_100": medium_optimization,
        "q3_medium_test_2000": medium_test,
        "q3_medium_elasticity_optimization_100": apply_elasticity(medium_optimization, demand_keys, price_keys, bases[3], small_cfg),
        "q3_medium_elasticity_test_2000": apply_elasticity(medium_test, demand_keys, price_keys, bases[3], small_cfg),
    }
    summary_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []
    for name, arrays in sets.items():
        np.savez_compressed(SCENARIO_DIR / f"{name}.npz", **arrays)
        summary_rows.extend(summarize_set(name, arrays, years))
        corr_rows.extend(correlation_audit_rows(arrays, demand_keys, parameter_keys, price_keys, q3, name))

    assumption = {
        "model_layers": ["Q3-A correlation only", "Q3-B correlation plus substitution/complementarity"],
        "correlation_strengths": ["weak", "medium"],
        "factor_weights": q3["factor_weights"],
        "note_factor_weight_is_not_correlation": True,
        "crop_groups": q3["crop_groups"],
        "substitute_elasticity": q3["substitute_elasticity"],
        "extra_substitute_pairs": q3.get("extra_substitute_pairs", []),
        "complement_pairs": q3["complement_pairs"],
        "demand_adjustment_clip": q3["demand_adjustment_clip"],
        "epsilon0_check": {
            "max_abs_demand_difference": epsilon0_max_diff,
            "status": "PASS",
        },
        "random_stream_spawn_keys": {
            "q3_weak_optimization_100": list(streams[4].spawn_key),
            "q3_weak_test_2000": list(streams[5].spawn_key),
            "q3_medium_optimization_100": list(streams[6].spawn_key),
            "q3_medium_test_2000": list(streams[7].spawn_key),
        },
        "assumption_notes": q3["assumption_notes"],
    }
    (RESULTS_DIR / "q3_assumption_config.json").write_text(
        json.dumps(assumption, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(
        RESULTS_DIR / "q3_scenario_distribution_audit.csv",
        ["scenario_set", "parameter", "year", "minimum", "p05", "mean", "p95", "maximum"],
        summary_rows,
    )
    write_csv(
        RESULTS_DIR / "q3_marginal_correlation_audit.csv",
        ["scenario_set", "check", "value", "expected_direction", "note"],
        corr_rows,
    )
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        group_names = list(q3["crop_groups"])
        crop_to_group = {
            int(crop_id): name
            for name, crop_ids in q3["crop_groups"].items()
            for crop_id in crop_ids
        }
        series = []
        labels = []
        for name in group_names:
            members = [
                optimization["demand"][:, 0, index]
                for index, (crop_id, _season) in enumerate(demand_keys)
                if crop_to_group[crop_id] == name
            ]
            if not members:
                continue
            series.append(np.mean(np.vstack(members), axis=0))
            labels.append(name)
        corr = np.corrcoef(np.vstack(series))
        figures = PROJECT_ROOT / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        fig.colorbar(image, ax=ax, fraction=0.046)
        ax.set_title("Q3-A 2024 类群平均需求相关性（弱相关）")
        fig.tight_layout()
        fig.savefig(figures / "q3_correlation_heatmap.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass
    print(json.dumps({name: {key: list(value.shape) for key, value in arrays.items()} for name, arrays in sets.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
