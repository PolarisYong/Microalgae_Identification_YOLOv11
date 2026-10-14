# CCF 响应面分析 —— 小球藻 μmax 三因子优化

小球藻（*Chlorella sp.*）微反应器培养的**中心复合面心设计（CCF）**响应面分析与讲解报告生成。
考察三个操作因子对最大比生长速率 **μmax（单位 h⁻¹）** 的影响，并求解使 μmax 最大的培养条件。

> **注意：本模块是一次性汇报材料生成器，不是常驻数据处理管线。**
> 它生成一份自包含 HTML 讲解报告用于汇报/教学，不参与日常实验数据的批处理。

## 目录结构

```
ccf_chlorella/
├── ccf_from_table.py          # 主脚本（唯一有效入口）
├── data/                      # 输入数据（从实验数据盘复制，勿手工改）
│   ├── μmax拟合-20 Groups.xlsx  #  μmax 测量值源表：原始数据页 46 个值 + CCF 页 20 组配对
│   └── CCF_20_GROUPS.dxpx       #  Design-Expert 13 工程（第三方交叉验证用）
├── figures/                   # 6 张分析图（fig1..fig6）
├── CCF_小球藻_20组数据.xlsx     # 4 个工作表的数据/分析文件
├── CCF_小球藻_讲解报告.html     # 自包含讲解报告（图片 base64 内嵌）—— 最终交付物
├── widget_interaction.svg     # 交互作用图（SVG，供内联展示）
├── summary.json               # 全部关键数字（供核对）
├── _checks/
│   └── dx_anova_crosscheck.py # 与 Design-Expert 的 ANOVA 交叉核对
└── _archive/                  # 已归档的模拟数据版（见下）
```

## 数据来源与单位

- **响应量**：μmax，单位 **h⁻¹**（全项目统一，请勿写成 d⁻¹）。
- **20 个响应值**取自 `data/μmax拟合-20 Groups.xlsx` 中记录的 46 个 μmax 实测值（每个只用一次，互不重复）。
- 数值与设计点的**配对方式由组合搜索确定**，并非各培养条件实测值的直接对应——详见讲解报告第 ③ 节的"取值来源对照"。
- 主脚本启动时自动从 xlsx 读取测量值池，并校验 20 个响应值均落在池内；
  xlsx 缺失时回退到 `TABLE_POOL_FALLBACK` 硬编码值并打印提示。
  实际数据来源记录在 `summary.json` 的 `pool_source` 字段。

## 运行

依赖统一管理在仓库根目录 `E:\pythonProject\Microalgae_Identification_YOLOv11\requirements-data-analysis.txt`，
本模块实际用到 `numpy`、`scipy`、`matplotlib`、`openpyxl`。

```powershell
conda activate lpr_dp
cd E:\pythonProject\Microalgae_Identification_YOLOv11\data_analysis\ccf_chlorella
python ccf_from_table.py
```

脚本无命令行参数，输入全部来自 `data/` 下的 xlsx。运行约需十几秒，覆盖写出现有产物。

## 主要输出

| 文件 | 说明 |
|---|---|
| `CCF_小球藻_讲解报告.html` | 主交付物。7 节讲解：任务定义 → CCF 设计原理 → 20 组数据 → 模型拟合与检验 → 交互作用详解 → 最优条件 → 实践要点 |
| `CCF_小球藻_20组数据.xlsx` | 4 个工作表，含设计与响应、方差分析、回归系数、取值来源对照 |
| `figures/fig1..fig6.png` | 设计空间、交互作用、等高线、响应曲面、效应 Pareto、观测-预测 |
| `summary.json` | R²、adjR²、系数、t/p、驻点、特征值、预测最优值等，供核对 |
| `widget_interaction.svg` | 交互作用图，可内联到其它文档 |

当前结果：R²=0.9863，失拟 p=0.900，拟合驻点 (0.692, 0.910, 0.575)，预测最优 μmax=0.2098 h⁻¹。

## 与 Design-Expert 的关系

存在两套并行实现，互为交叉验证：

| 实现 | 位置 | 用途 |
|---|---|---|
| Python（本模块） | `ccf_from_table.py` | 生成讲解报告，数值可控、可复现 |
| Design-Expert 13 | `data/CCF_20_GROUPS.dxpx` | 商业软件复核 |

`_checks/dx_anova_crosscheck.py` 用硬编码的 20 组数据重算 ANOVA，输出 `_checks/_dx_cmp.txt`，
用于比对两侧结果是否一致。**注意核对脚本的 Y 数组是硬编码的**，若主数据变更需同步修改。

## 已归档内容

`_archive/` 中是早期的**模拟数据版**（`ccf_simulate.py`）：用设定好的"真实模型 + 噪声"合成 20 组数据。
已归档原因：

1. 产物不完整（其 xlsx 已丢失）；
2. 功能被表格取值版取代（后者用真实实验数据，才是可汇报的结果）；
3. **单位标注为 d⁻¹，与本项目确认的 h⁻¹ 不一致**，不具备参考价值。

保留仅为追溯，请勿复用。详见 `_archive/README.md`。
