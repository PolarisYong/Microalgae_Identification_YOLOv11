# 微藻实验数据批处理 — 同批次分组

把一批 chamber（腔室）的生长曲线拟合结果按 μmax 聚成"同批次"数据组，并输出 Excel 分组工作簿、标记后的源工作簿、汇总 PPT 与 μmax 总览。

本功能是 `data_analysis` 项目下的一个模块，目录为 `batch_grouping/`。只使用纯 Python，不依赖 Node.js、Microsoft Excel 或 PowerPoint 的 COM 自动化。

## 目录结构

```
batch_grouping/
├── run_full_batch.py           # 入口①：在源码顶部填路径，或走 CLI
├── batch_run_folders.py        # 入口②：批处理某根目录下的多个子文件夹
├── data_analysis_batch/        # 核心包
│   ├── grouping.py             #   分组算法（CV + Jarque-Bera + Pareto DP）
│   ├── excel_io.py             #   工作簿读写、标记、合成图嵌入
│   ├── ppt_io.py               #   PPT 生成 + 合成生长曲线图
│   ├── validation.py           #   输出校验
│   ├── pipeline.py             #   流程编排
│   └── models.py               #   数据模型
├── data/
│   ├── source_data/            #   原始数据 + 参考输出.xlsx
│   └── processed_data/         #   处理结果
├── test_output_custom_group_B/ # 历史测试输出
├── _smoke_test_compact.py      # 分组算法回归测试
├── _smoke_test_compose.py      # 合成图函数回归测试
└── _verify_composite_embed.py  # 真实输出的嵌入图核对
```

## 安装环境

推荐 Python 3.11。本模块与项目内其他模块共用同一个环境，依赖清单统一放在仓库根目录 `requirements-data-analysis.txt`：

```powershell
conda create -n lpr_dp python=3.11 pip -y -c defaults --override-channels
conda activate lpr_dp
python -m pip install -r "E:\pythonProject\Microalgae_Identification_YOLOv11\requirements-data-analysis.txt"
```

本模块实际用到的依赖为 `openpyxl`、`python-pptx`、`pillow`。
分组算法（含 Jarque-Bera 检验）为纯 Python 实现，不依赖 numpy / scipy / statsmodels。

## 用法一：`run_full_batch.py`（单/多个指定文件夹）

在源码顶部填写输入与输出，两个列表按位置一一对应：

```python
input_data = "L30‑N20‑IC0.5%"
output_data = input_data
output_data_label = "18"

TARGET_BASE_DIRS: list[str] = [
    rf"F:\Workbuddy_Workspace\Peipei_Workspace\实验结果汇总\{input_data}",
]
TARGET_OUTPUT_DIRS: list[str] = [
    rf"F:\Workbuddy_Workspace\Peipei_Workspace\实验结果汇总\{output_data}\{output_data_label}",
]
```

输出位置填空字符串 `""` 时，结果写回对应输入文件夹。

分组数量规则同样在顶部：

```python
TARGET_MAX_GROUP_SIZE: int = 18
TARGET_MIN_GROUP_SIZE: int = 12
```

- `TARGET_MAX_GROUP_SIZE`：完整组大小上限，受每页 `2×9` PPT 版式限制，最大 18。
- `TARGET_MIN_GROUP_SIZE`：末组允许的最小大小，必须满足 `2 <= min < max <= 18`。
- `TARGET_MAX_MEAN_SPREAD_RATIO`：可选，组间均值极差上限（相对总体均值）。设为 `None` 时"组数优先"；设成如 `0.3` 时"组间差距小"会压过组数。

命令行方式（不需要改源码）：

```powershell
python "E:\pythonProject\Microalgae_Identification_YOLOv11\data_analysis\batch_grouping\run_full_batch.py" `
  --base-dir "F:\...\L30‑N20‑IC0.5%" `
  --output-dir "F:\...\L30‑N20‑IC0.5%\18"
```

## 用法二：`batch_run_folders.py`（整目录批处理）

一次处理某根目录下的**所有**子文件夹：

```powershell
python "E:\pythonProject\Microalgae_Identification_YOLOv11\data_analysis\batch_grouping\batch_run_folders.py" `
  --root "F:\Workbuddy_Workspace\Peipei_Workspace\实验结果汇总"
```

只处理指定的几个文件夹（`--folder` 可重复）：

```powershell
python "...\batch_run_folders.py" --root "F:\...\实验结果汇总" `
  --folder "L30-N20-IC0.5%" --folder "L210-N300-IC10%"
```

> 文件夹名容错：实验目录名里常用的是 Unicode 连字符 `‑`（U+2011），
> 手写 ASCII `-` 也能匹配上，不必照抄。

先看会处理什么、不写任何文件：

```powershell
python "...\batch_run_folders.py" --root "F:\...\实验结果汇总" --list
```

主要参数：

| 参数 | 说明 |
|---|---|
| `--root` | 根目录，处理其下所有直接子文件夹 |
| `--folder` | 只处理指定子文件夹，可重复；支持裸名或绝对路径 |
| `--reference` | 参考工作簿；默认按 `<root>/参考输出.xlsx` → `<folder>/参考输出.xlsx` 顺序自动找 |
| `--output-label` | 输出子目录名。默认 `18_batch_<时间戳>`（每轮独立）；传 `18` 用固定名 |
| `--report` | 汇总 JSON 报告路径 |
| `--max-group-size` / `--min-group-size` | 分组大小，默认 18 / 12 |
| `--max-mean-spread-ratio` | 组间均值极差上限（相对值） |
| `--ndjson-policy` | `archive` / `delete` / `keep`，默认 archive |
| `--allow-cleanup` | 允许管线先删旧输出再写（默认关闭，见下） |
| `--list` / `--dry-run` | 只列出待处理文件夹 |

`--allow-cleanup` 说明：本功能默认会先清理自己上一轮的输出。在受限沙箱
（如 WorkBuddy 内置终端）里，大批量删除可能被安全策略拦截，因此
`batch_run_folders.py` 默认关闭该清理、直接覆盖写入。在你自己的普通终端或
PyCharm 里跑时，可以加上 `--allow-cleanup` 恢复原生行为。

## 数据要求

每个目标文件夹需要：

- 包含原始 `.xlsx` 文件（内含 `汇总结果` 页签）。
- 参考工作簿 `参考输出.xlsx` 位于：`--reference` 显式指定 > 根目录 > 目标文件夹内 > 根目录的父目录。

## 分组规则

在同一培养条件的数据内部，划分出尽可能多的数据组：

1. **组内**：`CV < 15%` 且 `Jarque-Bera p > 0.05`（同质 + 近似正态）。
2. **组间**：各组 μmax 均值之差尽量小。
3. **形状**：`k` 个恰为 `TARGET_MAX_GROUP_SIZE` 的满组 + 至多一个末组（`TARGET_MIN_GROUP_SIZE .. max-1`）。

算法为按 μmax 排序后的右到左 Pareto 前沿动态规划，允许跳过离群行，
因此边界离群值不会污染任何一组。

## 主要输出

| 文件 | 说明 |
|---|---|
| `selected__groups_<组号>_<数量>.xlsx` | 每组一份；`selected_<n>` 页签含成员数据 + 数据下方**一张全组合成的生长曲线图** |
| `<源文件名>_marked.xlsx` | 源工作簿副本，入选 chamber 标红 |
| `selected_groups_growth_curves_2x9.pptx` | 每组一页的 2×9 版式 PPT |
| `selected_groups_umax_summary.xlsx` | 每组一行，含该组平均 μmax |
| `_batch_group_work/partition_results.json` | 分组明细 |
| `_batch_group_work/chart_manifest.json` | 图表清单 |
| `_batch_group_work/run_report.json` | 完整运行报告 |
| `_batch_group_work/ppt_charts/group_*/` | 各腔室 PNG + 该组 `composite.png` |

根目录已有的 `.ndjson` 默认移动到 `_batch_group_work/logs`。

## 回归测试

```powershell
python "E:\pythonProject\Microalgae_Identification_YOLOv11\data_analysis\batch_grouping\_smoke_test_compact.py"    # 分组算法
python "E:\pythonProject\Microalgae_Identification_YOLOv11\data_analysis\batch_grouping\_smoke_test_compose.py"    # 合成图函数
python "E:\pythonProject\Microalgae_Identification_YOLOv11\data_analysis\batch_grouping\_verify_composite_embed.py" # 嵌入图核对
```

`_verify_composite_embed.py` 不带参数时自动挑选 `data/processed_data` 下最近一次运行输出。

## PyCharm 配置

1. Python Interpreter 选择 `LPR_dp` Conda 环境。
2. Script path 选 `...\batch_grouping\run_full_batch.py` 或 `batch_run_folders.py`。
3. Working directory 选 `...\data_analysis\batch_grouping`。
