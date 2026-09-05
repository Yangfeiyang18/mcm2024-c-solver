from __future__ import annotations

import json
import shutil
import subprocess
import sys

from official_io import (
    FINAL_OUTPUTS_DIR,
    FINAL_RESULTS_DIR,
    LOCKED,
    OFFICIAL_BOOKS,
    PROJECT_ROOT,
    RESULTS_DIR,
    close_enough,
    load_area_by_key,
    read_csv,
    verify_excel_against_csv,
    write_json,
)


def run_audit(script: str, *args: str) -> None:
    command = [sys.executable, str(PROJECT_ROOT / "src" / "evaluation" / script), *args]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)


def audit_all_pass(path: Path) -> bool:
    rows = read_csv(path)
    return all(row.get("status") == "PASS" for row in rows) and rows


def check_official_excel() -> list[dict]:
    reports = []
    for book_name, csv_path in OFFICIAL_BOOKS.items():
        report = verify_excel_against_csv(FINAL_OUTPUTS_DIR / book_name, csv_path)
        reports.append(report)
        if report["status"] != "PASS":
            raise RuntimeError(f"{book_name} 与 {csv_path.name} 不一致")
    return reports


def check_locked_metrics() -> list[str]:
    failures: list[str] = []
    waste = json.loads((RESULTS_DIR / "q1_full_waste_metrics.json").read_text(encoding="utf-8"))
    discount = json.loads((RESULTS_DIR / "q1_full_discount_metrics.json").read_text(encoding="utf-8"))
    q2 = json.loads((RESULTS_DIR / "q2_lambda_025_metrics.json").read_text(encoding="utf-8"))
    pairs = [
        (waste["total_profit_yuan"], LOCKED["q1_waste_profit_yuan"], "Q1浪费利润"),
        (discount["total_profit_yuan"], LOCKED["q1_discount_profit_yuan"], "Q1半价利润"),
        (waste["solver"]["mip_gap"], LOCKED["q1_waste_gap"], "Q1浪费gap"),
        (discount["solver"]["mip_gap"], LOCKED["q1_discount_gap"], "Q1半价gap"),
        (q2["training_mean_profit_yuan"], LOCKED["q2_training_mean_yuan"], "Q2训练均值"),
        (q2["training_lower_cvar_profit_yuan"], LOCKED["q2_training_cvar_yuan"], "Q2训练CVaR"),
        (q2["solver"]["mip_gap"], LOCKED["q2_gap"], "Q2 gap"),
    ]
    for actual, expected, name in pairs:
        if not close_enough(actual, expected):
            failures.append(f"{name} 对不上锁定值：{actual} vs {expected}")
    if waste["constraint_violations"]["audit_status"] != "PASS":
        failures.append("Q1浪费审计不是 PASS")
    if discount["constraint_violations"]["audit_status"] != "PASS":
        failures.append("Q1半价审计不是 PASS")
    if q2["constraint_violations"]["audit_status"] != "PASS":
        failures.append("Q2审计不是 PASS")
    if close_enough(waste["total_profit_yuan"], LOCKED["obsolete_q1_waste_yuan"], abs_tol=100):
        failures.append("Q1浪费仍是旧数字 4075.77万")
    if close_enough(discount["total_profit_yuan"], LOCKED["obsolete_q1_discount_yuan"], abs_tol=100):
        failures.append("Q1半价仍是旧数字 5838.10万")
    return failures


def check_comparison_table() -> list[str]:
    failures: list[str] = []
    rows = {row["metric"]: row for row in read_csv(RESULTS_DIR / "q1_waste_vs_discount.csv")}
    waste = float(rows["total_profit_yuan"]["full_waste"])
    discount = float(rows["total_profit_yuan"]["full_discount"])
    if not close_enough(waste, LOCKED["q1_waste_profit_yuan"]):
        failures.append(f"比较表浪费利润仍是旧值 {waste}")
    if not close_enough(discount, LOCKED["q1_discount_profit_yuan"]):
        failures.append(f"比较表半价利润仍是旧值 {discount}")
    return failures


def check_q2_q3_same_plan() -> list[str]:
    failures: list[str] = []
    q2 = load_area_by_key(RESULTS_DIR / "q2_lambda_025_solution_long.csv")
    for name in ("q3a_solution_long.csv", "q3a_medium_solution_long.csv"):
        other = load_area_by_key(RESULTS_DIR / name)
        keys = set(q2) | set(other)
        if any(abs(q2.get(key, 0.0) - other.get(key, 0.0)) > 1e-6 for key in keys):
            failures.append(f"{name} 与 Q2 正式方案不同")
    return failures


def check_final_copies() -> list[str]:
    required = [
        FINAL_RESULTS_DIR / "paper_numbers.json",
        FINAL_RESULTS_DIR / "q1_full_waste_solution_long.csv",
        FINAL_RESULTS_DIR / "q1_full_discount_solution_long.csv",
        FINAL_RESULTS_DIR / "q2_lambda_025_solution_long.csv",
        FINAL_RESULTS_DIR / "q1_waste_vs_discount.csv",
        FINAL_RESULTS_DIR / "q3_test2000_empirical_correlation.csv",
        FINAL_OUTPUTS_DIR / "result1_1.xlsx",
        FINAL_OUTPUTS_DIR / "result1_2.xlsx",
        FINAL_OUTPUTS_DIR / "result2.xlsx",
        PROJECT_ROOT / "结果版本说明.md",
    ]
    return [f"缺少 {path}" for path in required if not path.exists()]


def main() -> None:
    run_audit("audit_q1_full_waste.py")
    run_audit("audit_q1_full_discount.py")
    run_audit("audit_q2.py", "lambda_025")
    FINAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in (
        "q1_full_waste_constraint_audit.csv",
        "q1_full_discount_constraint_audit.csv",
        "q2_lambda_025_constraint_audit.csv",
    ):
        source = RESULTS_DIR / name
        if source.exists():
            shutil.copy2(source, FINAL_RESULTS_DIR / name)
    excel_reports = check_official_excel()
    failures = []
    failures.extend(check_locked_metrics())
    failures.extend(check_comparison_table())
    failures.extend(check_q2_q3_same_plan())
    failures.extend(check_final_copies())
    for name in (
        "q1_full_waste_constraint_audit.csv",
        "q1_full_discount_constraint_audit.csv",
        "q2_lambda_025_constraint_audit.csv",
    ):
        if not audit_all_pass(RESULTS_DIR / name):
            failures.append(f"{name} 存在未通过检查")
    report = {
        "status": "PASS" if not failures else "FAIL",
        "excel": excel_reports,
        "failures": failures,
    }
    write_json(FINAL_RESULTS_DIR / "consistency_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit("最终一致性检查未通过")


if __name__ == "__main__":
    main()
