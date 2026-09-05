from __future__ import annotations

import json

from official_io import (
    FINAL_OUTPUTS_DIR,
    OFFICIAL_BOOKS,
    TEMPLATE_DIR,
    fill_official_workbook,
    write_json,
)


def main() -> None:
    reports = []
    for book_name, solution_path in OFFICIAL_BOOKS.items():
        report = fill_official_workbook(
            TEMPLATE_DIR / book_name,
            solution_path,
            FINAL_OUTPUTS_DIR / book_name,
        )
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["status"] != "PASS":
            raise SystemExit(f"{book_name} 与 CSV 不一致")
    write_json(FINAL_OUTPUTS_DIR / "official_fill_report.json", {"books": reports})


if __name__ == "__main__":
    main()
