from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = PROJECT_ROOT / "数据" / "清洗后数据"
SCENARIO_DIR = PROJECT_ROOT / "数据" / "情景数据"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIG_PATH = PROJECT_ROOT / "config" / "q2.json"


def read_csv(name: str) -> list[dict[str, str]]:
    with (CLEAN_DIR / name).open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_key_data() -> tuple[list[tuple[int, str]], list[tuple[int, str, str]], list[tuple[int, str]], dict[int, str]]:
    demand_rows = read_csv("demand_2023.csv")
    parameter_rows = read_csv("crop_parameters_2023.csv")
    crop_rows = read_csv("crops.csv")
    demand_keys = sorted((int(row["crop_id"]), row["season"]) for row in demand_rows)
    parameter_keys = sorted((int(row["crop_id"]), row["land_type"], row["season"]) for row in parameter_rows)
    price_keys = sorted({(crop_id, season) for crop_id, _land_type, season in parameter_keys})
    categories = {int(row["crop_id"]): row["crop_category"] for row in crop_rows}
    return demand_keys, parameter_keys, price_keys, categories


def base_arrays(
    demand_keys: list[tuple[int, str]],
    parameter_keys: list[tuple[int, str, str]],
    price_keys: list[tuple[int, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    demand_map = {
        (int(row["crop_id"]), row["season"]): float(row["expected_sales_jin"])
        for row in read_csv("demand_2023.csv")
    }
    parameter_map = {
        (int(row["crop_id"]), row["land_type"], row["season"]): (
            float(row["yield_jin_per_mu"]),
            float(row["cost_yuan_per_mu"]),
            float(row["price_mid"]),
        )
        for row in read_csv("crop_parameters_2023.csv")
    }
    price_map: dict[tuple[int, str], float] = {}
    for crop_id, land_type, season in parameter_keys:
        key = (crop_id, season)
        price = parameter_map[(crop_id, land_type, season)][2]
        if key in price_map and abs(price_map[key] - price) > 1e-8:
            raise ValueError(f"同一作物季次存在不同基准价格：{key}")
        price_map[key] = price
    return (
        np.asarray([demand_map[key] for key in demand_keys], dtype=np.float64),
        np.asarray([parameter_map[key][0] for key in parameter_keys], dtype=np.float64),
        np.asarray([parameter_map[key][1] for key in parameter_keys], dtype=np.float64),
        np.asarray([price_map[key] for key in price_keys], dtype=np.float64),
    )


def generate_random_set(
    count: int,
    rng: np.random.Generator,
    years: list[int],
    demand_keys: list[tuple[int, str]],
    parameter_keys: list[tuple[int, str, str]],
    price_keys: list[tuple[int, str]],
    categories: dict[int, str],
    bases: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    uncertainty: dict[str, Any],
) -> dict[str, np.ndarray]:
    base_demand, base_yield, base_cost, base_price = bases
    year_count = len(years)
    demand = np.empty((count, year_count, len(demand_keys)), dtype=np.float64)
    yields = np.empty((count, year_count, len(parameter_keys)), dtype=np.float64)
    costs = np.empty((count, year_count, len(parameter_keys)), dtype=np.float64)
    prices = np.empty((count, year_count, len(price_keys)), dtype=np.float64)

    wheat_corn_bounds = uncertainty["wheat_corn_demand_annual_growth_uniform"]
    other_demand_bounds = uncertainty["other_crop_demand_relative_to_2023_uniform"]
    yield_bounds = uncertainty["yield_relative_to_2023_uniform"]
    cost_left, cost_mode, cost_right = uncertainty["cost_annual_growth_triangular"]
    grain_bounds = uncertainty["grain_price_relative_to_2023_uniform"]
    veg_left, veg_mode, veg_right = uncertainty["vegetable_price_annual_growth_triangular"]
    fungus_bounds = uncertainty["ordinary_fungus_price_annual_decline_uniform"]
    morel_decline = float(uncertainty["morel_price_annual_decline_fixed"])

    for key_index, (crop_id, _season) in enumerate(demand_keys):
        if crop_id in (6, 7):
            running = np.full(count, base_demand[key_index], dtype=np.float64)
            for year_index in range(year_count):
                running = running * (1.0 + rng.uniform(*wheat_corn_bounds, size=count))
                demand[:, year_index, key_index] = running
        else:
            shocks = rng.uniform(*other_demand_bounds, size=(count, year_count))
            demand[:, :, key_index] = base_demand[key_index] * (1.0 + shocks)

    yield_shocks = rng.uniform(*yield_bounds, size=yields.shape)
    yields[:] = base_yield[None, None, :] * (1.0 + yield_shocks)

    running_cost = np.broadcast_to(base_cost, (count, len(parameter_keys))).copy()
    for year_index in range(year_count):
        growth = rng.triangular(cost_left, cost_mode, cost_right, size=running_cost.shape)
        running_cost *= 1.0 + growth
        costs[:, year_index, :] = running_cost

    for key_index, (crop_id, _season) in enumerate(price_keys):
        category = categories[crop_id]
        if category == "grain":
            shocks = rng.uniform(*grain_bounds, size=(count, year_count))
            prices[:, :, key_index] = base_price[key_index] * (1.0 + shocks)
        elif category == "vegetable":
            running = np.full(count, base_price[key_index], dtype=np.float64)
            for year_index in range(year_count):
                growth = rng.triangular(veg_left, veg_mode, veg_right, size=count)
                running *= 1.0 + growth
                prices[:, year_index, key_index] = running
        elif crop_id == 41:
            for year_index in range(year_count):
                prices[:, year_index, key_index] = base_price[key_index] * (1.0 - morel_decline) ** (year_index + 1)
        else:
            running = np.full(count, base_price[key_index], dtype=np.float64)
            for year_index in range(year_count):
                decline = rng.uniform(*fungus_bounds, size=count)
                running *= 1.0 - decline
                prices[:, year_index, key_index] = running
    return {"demand": demand, "yield": yields, "cost": costs, "price": prices}


def generate_mean_set(
    years: list[int],
    demand_keys: list[tuple[int, str]],
    parameter_keys: list[tuple[int, str, str]],
    price_keys: list[tuple[int, str]],
    categories: dict[int, str],
    bases: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, np.ndarray]:
    base_demand, base_yield, base_cost, base_price = bases
    year_count = len(years)
    demand = np.empty((1, year_count, len(demand_keys)), dtype=np.float64)
    yields = np.broadcast_to(base_yield, (1, year_count, len(parameter_keys))).copy()
    costs = np.empty((1, year_count, len(parameter_keys)), dtype=np.float64)
    prices = np.empty((1, year_count, len(price_keys)), dtype=np.float64)
    for index, (crop_id, _season) in enumerate(demand_keys):
        for year_index in range(year_count):
            demand[0, year_index, index] = (
                base_demand[index] * 1.075 ** (year_index + 1) if crop_id in (6, 7) else base_demand[index]
            )
    for year_index in range(year_count):
        costs[0, year_index, :] = base_cost * 1.05 ** (year_index + 1)
    for index, (crop_id, _season) in enumerate(price_keys):
        for year_index in range(year_count):
            category = categories[crop_id]
            if category == "grain":
                factor = 1.0
            elif category == "vegetable":
                factor = 1.05 ** (year_index + 1)
            elif crop_id == 41:
                factor = 0.95 ** (year_index + 1)
            else:
                factor = 0.97 ** (year_index + 1)
            prices[0, year_index, index] = base_price[index] * factor
    return {"demand": demand, "yield": yields, "cost": costs, "price": prices}


def save_set(name: str, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(SCENARIO_DIR / f"q2_{name}.npz", **arrays)


def summarize_set(name: str, arrays: dict[str, np.ndarray], years: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parameter_name, values in arrays.items():
        for year_index, year in enumerate(years):
            flattened = values[:, year_index, :].reshape(-1)
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


def main() -> None:
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    years = [int(year) for year in config["years"]]
    demand_keys, parameter_keys, price_keys, categories = build_key_data()
    bases = base_arrays(demand_keys, parameter_keys, price_keys)
    seed_sequence = np.random.SeedSequence(int(config["random_seed"]))
    optimization_seed, test_seed, convergence_small_seed, convergence_large_seed = seed_sequence.spawn(4)
    counts = config["scenario_counts"]
    sets = {
        "mean": generate_mean_set(years, demand_keys, parameter_keys, price_keys, categories, bases),
        "optimization_100": generate_random_set(
            int(counts["optimization"]), np.random.default_rng(optimization_seed), years,
            demand_keys, parameter_keys, price_keys, categories, bases, config["uncertainty"],
        ),
        "test_2000": generate_random_set(
            int(counts["out_of_sample_test"]), np.random.default_rng(test_seed), years,
            demand_keys, parameter_keys, price_keys, categories, bases, config["uncertainty"],
        ),
        "optimization_50": generate_random_set(
            int(counts["convergence_small"]), np.random.default_rng(convergence_small_seed), years,
            demand_keys, parameter_keys, price_keys, categories, bases, config["uncertainty"],
        ),
        "optimization_200": generate_random_set(
            int(counts["convergence_large"]), np.random.default_rng(convergence_large_seed), years,
            demand_keys, parameter_keys, price_keys, categories, bases, config["uncertainty"],
        ),
    }
    summary_rows: list[dict[str, Any]] = []
    for name, arrays in sets.items():
        save_set(name, arrays)
        summary_rows.extend(summarize_set(name, arrays, years))
    metadata = {
        "seed_entropy": int(config["random_seed"]),
        "random_stream_spawn_keys": {
            "optimization_100": list(optimization_seed.spawn_key),
            "test_2000": list(test_seed.spawn_key),
            "optimization_50": list(convergence_small_seed.spawn_key),
            "optimization_200": list(convergence_large_seed.spawn_key),
        },
        "years": years,
        "demand_keys": [[crop_id, season] for crop_id, season in demand_keys],
        "parameter_keys": [[crop_id, land_type, season] for crop_id, land_type, season in parameter_keys],
        "price_keys": [[crop_id, season] for crop_id, season in price_keys],
        "scenario_shapes": {
            name: {key: list(value.shape) for key, value in arrays.items()} for name, arrays in sets.items()
        },
        "assumption_notes": config["assumption_notes"],
    }
    (SCENARIO_DIR / "q2_scenario_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(
        RESULTS_DIR / "q2_scenario_distribution_audit.csv",
        ["scenario_set", "parameter", "year", "minimum", "p05", "mean", "p95", "maximum"],
        summary_rows,
    )
    print(json.dumps(metadata["scenario_shapes"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
