from __future__ import annotations

import shutil

import compare_q1_sales_policy
import fill_official_results
import make_paper_figures
from official_io import FINAL_OUTPUTS_DIR, FINAL_RESULTS_DIR, PROJECT_ROOT, RESULTS_DIR


COPY_FILES = [
    "q1_full_waste_solution_long.csv",
    "q1_full_waste_metrics.json",
    "q1_full_waste_yearly_profit.csv",
    "q1_full_waste_crop_summary.csv",
    "q1_full_waste_constraint_audit.csv",
    "q1_full_discount_solution_long.csv",
    "q1_full_discount_metrics.json",
    "q1_full_discount_yearly_profit.csv",
    "q1_full_discount_crop_summary.csv",
    "q1_full_discount_constraint_audit.csv",
    "q1_waste_vs_discount.csv",
    "q1_waste_vs_discount_by_crop.csv",
    "q2_lambda_025_solution_long.csv",
    "q2_lambda_025_metrics.json",
    "q2_lambda_025_constraint_audit.csv",
    "q2_lambda_025_training_profit.csv",
    "q2_independent_vs_correlated.csv",
    "q2_independent_vs_correlated_diff.csv",
    "q3_comparison_metrics.csv",
    "q3_marginal_correlation_audit.csv",
    "q3_assumption_config.json",
    "q3_epsilon0_check.json",
    "q3a_metrics.json",
    "q3a_medium_metrics.json",
    "q3a_constraint_audit.csv",
    "q3a_medium_constraint_audit.csv",
]


def copy_final_results() -> None:
    FINAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in COPY_FILES:
        source = RESULTS_DIR / name
        if source.exists():
            shutil.copy2(source, FINAL_RESULTS_DIR / name)


def main() -> None:
    compare_q1_sales_policy.main()
    fill_official_results.main()
    make_paper_figures.main()
    copy_final_results()
    print(f"final results -> {FINAL_RESULTS_DIR.relative_to(PROJECT_ROOT)}")
    print(f"final outputs -> {FINAL_OUTPUTS_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
