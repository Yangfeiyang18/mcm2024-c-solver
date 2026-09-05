# MCM 2024 Problem C — Crop Planting Solver

2024 年全国大学生数学建模竞赛 **C 题「农作物的种植策略」** 代码仓库。  
由交接包整理为可 `git clone` 后直接配置环境运行的标准仓库。

## 仓库结构

```text
mcm2024-c-solver/
├─ README.md                 # 本说明
├─ requirements.txt          # Python 依赖（推荐）
├─ requirements-solver.txt   # 原交接包求解器钉死版本
├─ config/                   # Q1/Q2 求解配置
├─ src/
│  ├─ data/                  # 清洗 / 情景生成
│  ├─ models/                # Q1/Q2 求解脚本
│  └─ evaluation/            # 审计与对比
├─ 数据/
│  ├─ 原始数据/              # 题面附件
│  ├─ 清洗后数据/            # 已审计，默认可直接用
│  └─ 情景数据/              # Q2 情景 npz
├─ 建模/                     # 建模口径与交接文档
├─ results/                  # 已有中间结果（CSV/JSON）
├─ outputs/                  # 官方模板填写结果
└─ logs/                     # 求解与清洗日志
```

> 代码用脚本自身位置定位项目根目录，并依赖中文路径 `数据/...`。  
> **不要改这些中文目录名**，否则求解脚本会找不到数据。

## 主机快速开始（Linux / Windows）

```bash
git clone <本仓库 URL> mcm2024-c-solver
cd mcm2024-c-solver

python -m venv .venv
# Windows:
#   .\.venv\Scripts\Activate.ps1
# Linux/macOS:
#   source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 冒烟测试 HiGHS
python src/models/test_highs.py
```

期望输出中含 `"status": "Optimal"` 且 `objective` 约为 `6.0`。

## 常用命令

均在**仓库根目录**执行：

```bash
# 问题1 基线（校验用，非最终）
python src/models/solve_q1_baseline.py

# 问题1 完整：超额浪费 / 半价
python src/models/solve_q1_full_waste.py
python src/models/solve_q1_full_discount.py

# 问题1 审计与对比
python src/evaluation/audit_q1_full_waste.py
python src/evaluation/audit_q1_full_discount.py
python src/evaluation/compare_q1_sales_policy.py

# 问题2（情景已生成时可直接求解；耗时长）
python src/models/solve_q2.py
python src/evaluation/audit_q2.py
```

重新生成 Q2 情景（一般不必）：

```bash
python src/data/generate_q2_scenarios.py
```

数据清洗脚本为 Node：`src/data/clean_data.mjs`。新机器**默认不要重跑清洗**，直接用 `数据/清洗后数据/`。

## 环境说明

| 项 | 说明 |
|----|------|
| Python | 建议 3.10+ |
| 求解器 | `highspy==1.15.1`（HiGHS） |
| 数值 | `numpy` |
| 可选本地包目录 | 旧代码会 `sys.path` 插入 `.python_packages/`；**新环境用 venv 即可**，不必复制旧设备该目录 |

## 进度摘要

三问核心模型已收工。论文和官方表只从定稿目录取数，见 [`结果版本说明.md`](结果版本说明.md)。

- Q1 浪费 / 半价：4076.85 / 5846.51 万元；官方表 `outputs/final/result1_1.xlsx`、`result1_2.xlsx`
- Q2 锁定 λ=0.25；官方表 `outputs/final/result2.xlsx`
- Q3：种植结构与 Q2 相同；2000 情景上风险随相关档位递进
- 收尾复现：`python src/evaluation/prepare_final_bundle.py` 然后 `python src/evaluation/check_final_consistency.py`

旧交接文档仍可用于环境和早期口径；**最终数字以 `结果版本说明.md` 和 `建模/求解收工交接_20260904.md` 为准。**

## 文档入口

- `结果版本说明.md` — 论文取数目录
- `建模/求解收工交接_20260904.md` — 收工口径与不能写清单
- `建模/交接文档_新设备_建模与代码进度_20260901.md` — 早期交接
- `练习题三人分工流程.md`

## License / 用途

仅供本队备赛与主机复现使用；题面附件版权归竞赛组委会。