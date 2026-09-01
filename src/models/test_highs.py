from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / ".python_packages"))

import highspy  # noqa: E402


def main() -> None:
    model = highspy.Highs()
    model.setOptionValue("output_flag", False)

    x = model.addVariable(lb=0, ub=10, type=highspy.HighsVarType.kInteger, name="x")
    y = model.addVariable(lb=0, ub=10, type=highspy.HighsVarType.kInteger, name="y")
    model.addConstr(2 * x + y <= 4, name="capacity_1")
    model.addConstr(x + 2 * y <= 4, name="capacity_2")
    model.maximize(3 * x + 2 * y)

    status = model.getModelStatus()
    solution = model.getSolution()
    objective = model.getObjectiveValue()
    result = {
        "solver": "HiGHS",
        "version": model.version(),
        "status": model.modelStatusToString(status),
        "x": solution.col_value[0],
        "y": solution.col_value[1],
        "objective": objective,
    }
    print(result)

    if result["status"] != "Optimal":
        raise RuntimeError(f"HiGHS 未返回最优解：{result}")
    if abs(objective - 6.0) > 1e-8:
        raise RuntimeError(f"测试模型目标值应为6，实际为{objective}")


if __name__ == "__main__":
    main()
