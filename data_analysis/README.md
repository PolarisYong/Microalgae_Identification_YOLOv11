# data_analysis —— 微藻实验数据分析

> 本目录原为独立项目 `E:/pythonProject/data_analysis`，2026-10 已并入
> `Microalgae_Identification_YOLOv11` 仓库，作为「数据分析」模块。
> 上游的识别产物由 [`../microalgae_identification/`](../microalgae_identification/) 产出。

微藻实验数据处理模块集合。每个功能是一个自成语境的子文件夹，含自己的代码、数据与文档。

## 功能模块

| 目录 | 功能 | 入口 |
|---|---|---|
| `exp_batch_aggregator/` | **实验工况汇总**：依据工况标记表（批次→流道→数据图标签格式），把各批次各流道的实验结果按相同工况归集到同一文件夹，文件名加批次日期前缀 | `exp_cond_summarizer.py` |
| `batch_grouping/` | **同批次分组**：把同一工况下各 chamber 的 μmax 聚成同质数据组，输出分组 Excel、标记源文件、汇总 PPT | `run_full_batch.py`（单/多文件夹）<br>`batch_run_folders.py`（整目录批处理） |
| `ccf_chlorella/` | **CCF 响应面分析**：小球藻三因子（光照/氨氮/无机碳）CCF 设计、RSM 建模与讲解报告生成 | `ccf_from_table.py` |
| `simdata_pipeline/` | **模拟数据构建**：把 batch_grouping 的数据组回溯到原始腔室数据并重新打包（含合并拟合与可视化图） | `build_20_groups.py`（CCF 选定的 20 组）<br>`link_group_assets.py`（回链分组文件与曲线页）<br>`build_unused_groups.py`（导出未使用的 26 组） |
| `kinetics/` | **生长动力学（并行实现）**：点级 QC、固定 N0 拟合、Bootstrap 置信区间，输出独立的一套结果 | `06_growth_kinetics_pipeline.py` |
| `exploratory/` | **探索性分析**：μmax 抽取、条件编码拆解、论文配图 | `97_get_3data.py`、`98_extract_miumax_data.py`、`paper_figures/` |

各模块的详细用法见其自身目录下的 `README.md`。

## 环境

各模块**共用同一个 Conda 环境**，依赖统一到仓库根的 `requirements-data-analysis.txt`：

```powershell
conda create -n lpr_dp python=3.11 pip -y -c defaults --override-channels
conda activate lpr_dp
python -m pip install -r "E:\pythonProject\Microalgae_Identification_YOLOv11\requirements-data-analysis.txt"
```

已验证环境：Python 3.11.15（Conda 环境 `lpr_dp`）。

| 依赖 | 被哪些模块使用 |
|---|---|
| `numpy`、`scipy` | `ccf_chlorella`（RSM 建模）、`simdata_pipeline`（合并拟合） |
| `matplotlib` | `ccf_chlorella`、`simdata_pipeline`（嵌入的可视化图） |
| `pandas`、`xlrd` | `exp_batch_aggregator` |
| `openpyxl` | 三个模块共用 |
| `pillow`、`python-pptx` | `batch_grouping` |

`batch_grouping` 的分组算法（含 Jarque-Bera 检验）为纯 Python 实现，不依赖科学计算栈。

## 典型数据流

三个模块按实验分析的自然顺序衔接，前一模块的输出目录即后一模块的输入目录：

```
原始实验数据（按日期/流道组织）
        │
        │  exp_batch_aggregator —— 按「实验工况」横向归集
        ▼
实验结果汇总/<工况标签>/<YYYYMMDD>_CH<n>_可视化结果_<N>.xlsx
        │
        │  batch_grouping —— 在单一工况内部，按 μmax 纵向同质分组
        ▼
分组 Excel + 标记工作簿 + 2×9 汇总 PPT + μmax 总览
        │
        │  ccf_chlorella —— 跨工况的响应面建模与寻优
        ▼
RSM 模型 / 驻点 / 交互作用图 / 讲解报告（HTML）
```

各模块可独立运行，不强制依赖上游输出：若要跳过某一步，按各模块 README 中的「输入结构」手工准备目录即可。

## 约定

- 新增功能时，在 `data_analysis/` 下新建一个功能文件夹，把该功能的**入口脚本、核心包、数据目录、README** 全部放进去，保持模块自包含。
- **依赖统一在仓库根** `requirements-data-analysis.txt` 管理，各模块不再单独维护依赖清单；新增第三方依赖时直接追加，并在注释中标明使用方。
- 模块级说明只放在本文件；功能细节写进各模块自己的 `README.md`。
- 涉及文件复制/删除的脚本一律提供 `--dry-run`（或 `--list`）预览，且默认只复制、不移动不删除源文件。

## 数据存放约定

- 大体积数据工作区（`batch_grouping/data/`、`模拟数据处理_part1/`、`_backup/`、`simdata_pipeline/_logs/`）
  **已列入 `.gitignore`**：本地目录结构保留，但不进入版本库。
- 各模块的输入数据放在自己的 `data/` 目录下，脚本一律从 `data/` 读取，**不直接引用实验数据盘（F:）的路径**——数据盘路径不可控。
- 需要的数据从数据盘**复制**进来，数据盘侧保持只读，不在其上做删除或整理。
- 若某份数据同时被多个模块使用，在各模块 `data/` 下各存一份，并在 README 注明来源。
- `simdata_pipeline/config.py` 的 `PROJECT_ROOT` 由文件位置自动推导，整仓搬家后无需手改；
  需要指向别处时改这一行即可。

## 待办

- `batch_grouping` 的 `--root` 默认路径与 `exp_batch_aggregator` 的输出目录尚未统一到一处，跨模块串联时需手工指定。
- `simdata_pipeline` 依赖 `ccf_chlorella/data/μmax拟合-20 Groups.xlsx` 作为映射表，
  该跨模块引用已由 `config.py` 的 `CCF_MAPPING_XLSX` 显式配置（相对 `PROJECT_ROOT`）。
