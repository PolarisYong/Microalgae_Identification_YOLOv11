# simdata_pipeline —— CCF「20 组模拟数据」构建

把 ccf_chlorella 选定的 20 组响应曲面数据，**回溯到原始腔室级数据**并重新打包。

## 做了什么

```
CCF 映射表（μmax拟合-20 Groups.xlsx 的 CCF 页，20 行）
   │
   ├─ 前三列 光照/氨氮/无机碳（编码 -1/0/1）──→ 输出文件夹名「L30‑N20‑IC10%」
   │
   └─ 第五列「原始数据来源」──→ 来源工况 + Response 序号
                                 │
                                 ▼
        数据汇总分析结果/<来源工况>/18_batch_*/selected__groups_<i>_<n>.xlsx
                                 │      （batch_grouping **已选好**的腔室，本脚本不重新筛选）
                                 ▼
        μmax_revision_原始数据/<日期>/数据汇总/03_可视化结果/96小时/CH<k>_可视化结果_96.xlsx
                                 │      页签 CH<k>_CB<j>
                                 ▼
        20组模拟数据/<自变量组合>/<自变量组合>_可视化结果.xlsx
```

## 两个关键约定（容易踩坑）

**1. 自变量组合 ≠ 数据来源条件。**
CCF 表的 μmax 是跨条件挑选后重新配对的，所以每行的「光照/氨氮/无机碳」与「原始数据来源」经常不一致。
**文件夹名按自变量组合取，数据按「原始数据来源」取。**

**2. 「原始数据来源」的解码方式。**
形如 `-11-1 res1`、`000 第四行 res1`、`00-1 res3`：
- 编码部分用正则 `-?1|0` 切分 → `-11-1` → (光强 −1, 氨氮 +1, CO2 −1)
- `第X行`：中心点 (0,0,0) 在原始汇总表中有多行，用它区分
- `resN`：原始表的第 N 个 Response 列

**Response N 的 μmax 均值 ≡ batch_grouping 第 N 组的 μmax 均值**（已用样本量 n=12/17 核对一致），
因此脚本**优先按 μmax 均值匹配分组**，匹配失败才退回 `resN → 第 N 组`。实测 20 行全部由 μmax 命中。

## 目录结构

```
simdata_pipeline/
├── config.py              # ★ 所有路径与参数集中在这里，换机器只改这个文件
├── build_20_groups.py     # 入口①：构建 20 组模拟数据（原始腔室数据 + 合并拟合 + 可视化图）
├── link_group_assets.py   # 入口②：把每个子文件夹回链到分组 xlsx 与生长曲线 pptx 页
├── build_unused_groups.py # 入口③：导出未被 CCF 选中的剩余数据组 → 未使用数据/
├── build_second_round.py  # 入口④：未参与分组的腔室做二次分组 → 所有未参与数据分组腔室数据/
├── relabel_group.py       # 入口⑤：把一个数据组的所有标签改写成指定条件（工作簿/PPT/文件名）
├── build_english_version.py # 入口⑥：20 组 + 最优解整体转英文版 → Group21_English_Version/
├── link_selected_summary.py # 入口⑦：selected 页并入可视化工作簿的培养条件区 + 删多余页签
├── _verify_selected.py    # 入口⑦ 的校验脚本（逐格比对源区域 / 图片锚点 / 页签数）
├── core/
│   ├── group_index.py     # 分组索引 + 腔室集合匹配 + summary 指标读取
│   ├── unused_scan.py     # 入口④：扫描标红行 → 未参与分组的腔室
│   ├── screening.py       # 入口④：按微藻生长规律筛查异常腔室
│   ├── second_round.py    # 入口④：汇总/剔除/数据来源三份文档的写出
│   ├── group_assets.py    # 入口④：复用 batch_grouping 算法与 PPT 版式生成分组产物
│   ├── relabel.py         # 入口⑤：读回腔室数据 + 重建 PPT（换图 / 改标签）
│   ├── pptx_tools.py      # 从 pptx 抽取指定页另存（link 用）
│   ├── logging_setup.py   # 日志（控制台 + 文件）
│   ├── conditions.py      # 工况编码 ↔ 目录名 互转
│   ├── ccf_table.py       # 读 CCF 映射表 + 解析「原始数据来源」
│   ├── groups.py          # 读 batch_grouping 已选分组 + 定位分组
│   ├── rawdata.py         # 聚合文件名 → 原始数据路径，读取腔室页签 + 目标数量/总面积列
│   ├── fitting.py         # 合并拟合（修正 Logistic，Y=ln(Nrel) 尺度）
│   ├── plotting.py        # 可视化图（移植自 08_processed_batch.py，生成 PNG 并插图）
│   ├── writer.py          # 生成输出工作簿
│   ├── en_labels.py       # 入口⑥：条件映射 + 全英文标签 + 参数键归一化
│   ├── en_plotting.py     # 入口⑥：英文图表（Y=ln(Nrel) 尺度）
│   ├── en_writer.py       # 入口⑥：英文工作簿（汇总结果 + CB-n 页）
│   ├── en_richtext.py     # 入口⑥：模板 run 级富文本（斜体/上下标）spec 构建
│   ├── en_pptx.py         # 入口⑥：英文 PPT（图片替换 + 标签配对 + 符号格式）
│   ├── en_combo.py        # 入口⑥：selected 页组合生长曲线图重绘
│   ├── en_selected.py     # 入口⑥：selected 表头中→英
│   └── selected_summary.py # 入口⑦：把「汇总结果」培养条件区并入 selected_N 页
└── _logs/                 # 每次运行的日志
```

## 运行

依赖 `openpyxl` + `numpy` + `scipy` + `matplotlib`（均已含在项目根 `requirements.txt`）。
`numpy`/`scipy` 用于合并拟合，`matplotlib` 用于生成嵌入图。

```powershell
cd E:\pythonProject\Microalgae_Identification_YOLOv11\data_analysis\simdata_pipeline
python build_20_groups.py --dry-run    # 先核对映射，不写文件（强烈建议先跑这个）
python build_20_groups.py              # 正式写出（含可视化图）
python build_20_groups.py --no-plots   # 只写数据表，不嵌图（更快、体积更小）
```

可选参数：`--ccf-xlsx` `--agg-dir` `--raw-dir` `--out-dir` `--hours` `--dry-run` `--no-plots`

### 入口② `link_group_assets.py`（数据组回链）

```powershell
python link_group_assets.py --dry-run              # 只核对，不复制不建 pptx
python link_group_assets.py                        # 正式执行（默认不覆盖）
python link_group_assets.py --on-exist backup      # 已存在先备份再写入
python link_group_assets.py --on-exist overwrite   # 直接覆盖
```

### 入口③ `build_unused_groups.py`（导出未使用数据组）

「数据汇总分析结果」下共有 46 个 `selected__groups_*.xlsx`，CCF 只用了 20 个。
本脚本把剩下 26 个导出到 `未使用数据/`，每个子文件夹的结构与「20组模拟数据」完全一致。

```powershell
python build_unused_groups.py --dry-run    # 只核对（不建目录、不写文件）
python build_unused_groups.py              # 正式导出
python build_unused_groups.py --no-plots   # 不嵌图
```

- **已使用判定**：用生成表格「数据来源」页签的腔室集合，反查命中的分组文件（复用入口②的匹配逻辑）
- **子文件夹命名**：读分组文件 summary 页签 `mu_max_mean_h^-1`，去掉前导「0.」取前 4 位
  （0.0657777… → `0657`）；重名自动加 `_02`/`_03`
- **幂等**：再次运行时，已存在的子文件夹会被反查识别并跳过；中途失败留下的空文件夹会被复用而非另建
- 源目录全程只读，只做读取与复制

### 入口④ `build_second_round.py`（未参与分组腔室的二次分组）

前三个入口都建立在「batch_grouping **已选好**的腔室」之上。本入口处理剩下的部分：
`数据汇总分析结果/<工况>/18_batch_*/<源>_marked.xlsx` 的「汇总结果」页签里，**标红行 = 该腔室
已被选进某个数据组**；把未被标红的腔室挑出来，筛查后重新分组。

```powershell
python build_second_round.py --dry-run        # 只核对，不写任何文件（强烈建议先跑）
python build_second_round.py                  # 正式写出
python build_second_round.py --no-plots       # 不嵌可视化图（快很多）
python build_second_round.py --per-condition  # 改为「每个自变量文件夹各自分组」
python build_second_round.py --force          # 已存在的数据组文件夹重算覆盖
```

五步流程与产物：

| 步骤 | 做什么 | 产物 |
|---|---|---|
| ① 扫描 | 遍历工况文件夹 → 用标红行剔除已使用腔室 | —— |
| ② 汇总 | 每个未使用腔室一个页签，H/I 列标注实际来源 | `未参与数据分组腔室数据.xlsx`（每份 ≤50 页签，超出拆 `_part2`…） |
| ③ 筛查 | 按微藻生长规律剔除异常腔室，逐条给出中文原因 | `剔除腔室数据.xlsx`（同样拆分 + 原因写进 H4/I4） |
| ④ 分组 | 剩余腔室按 batch_grouping 的方式分组 | `<μmax 编码>/` ×3 个文件 |
| ⑤ 归档 | 逐条记录每组成员的实际来源 | `文档数据来源.xlsx` |

```
所有未参与数据分组腔室数据/
├── 未参与数据分组腔室数据.xlsx      ← 全部未标红腔室（每份 50 个页签）
├── 未参与数据分组腔室数据_part2.xlsx …
├── 剔除腔室数据.xlsx                ← 生长规律异常的腔室
├── 文档数据来源.xlsx                ← 数据组成员 / 未参与分组腔室 / 剔除腔室 三张清单
├── _work/                           ← 中间产物（腔室曲线 PNG），可删
└── <μmax 编码>/                     ← 0452 / 0717 / …（与「未使用数据」同款命名）
        ├── <μmax 编码>_可视化结果.xlsx
        ├── selected__groups_<i>_<n>.xlsx
        └── <μmax 编码>.pptx
```

**几个实现上的取舍**

- **来源标注不破坏原始结构**：来源写在 H/I 两列（自变量文件夹 / 源文件 / 原始页签 / 剔除原因），
  A:F 列仍是原始 6 列数据，页签可被后续程序按原格式读取。页签名本身也带来源前缀
  （`CH1_CB18` → 重名时退化成 `20260504_CH1_CH1_CB18` → 再加工况前缀 → `_02`）。
- **筛查阈值不是拍脑袋定的**：用「已被 batch_grouping 接受（标红）的 749 个腔室」的实际分布反推，
  取在已接受组边界稍外侧——已接受组 `fold` 最低 4.05、`end_peak` 的 1% 分位 0.436、
  `jumps` 中位数 2。因此只会剔除「比专家从未接受过的还差」的腔室。
- **R² ≤ 0.9 不算生长异常**：那是 batch_grouping 自己的严格候选条件（`N0 > 0 且 R² > 0.9`），
  不达标的腔室仍完整保留在「未参与数据分组腔室数据.xlsx」里，只是不参与分组，数量单独统计。
  想把它们一并写进剔除文件，把 `config.SCREEN_DROP_POOR_FIT` 改成 `True`。
- **默认把所有未使用腔室视为同一自变量下的数据**（单一池），这是需求原话；
  加 `--per-condition` 可改成每个工况各自分组。
- **PPT 的腔室曲线是现场绘制的**：batch_grouping 的 `extract_growth_charts` 依赖源工作簿里
  已嵌入的 300 dpi 图片，但这些源文件的 drawing 关联用了绝对路径，openpyxl 读不出 `_images`，
  因此改用本项目的 matplotlib 绘图（与可视化工作簿里的图同款）。

### 入口⑤ `relabel_group.py`（按新条件改写一个数据组的全部标签）

把一个数据组文件夹里所有「条件标识」与「原始腔室溯源」统一改写成指定条件，
用于「把这组数据视作某个条件的结果」的场景。

```powershell
python relabel_group.py --condition "L182‑N287‑IC461" \
                        --note "光照=182 μmol/m²/s，氨氮=287 mg/L，无机碳=461 mg/L"
python relabel_group.py --dry-run          # 只打印计划与校验，不写文件
```

默认目标：`模拟数据处理_part1/20组模拟数据+1组最优模拟/2208`。

| 改写对象 | 改前 | 改后 |
|---|---|---|
| 工作簿内嵌图标题 | `L120-N300-IC5.25%_CB1 Cell growth curve fitting` | `L182-N287-IC461_CB1 Cell growth curve fitting` |
| 汇总页合并拟合图 | `(merged data, 12 chambers, L120-N300-IC5.25%)` | `(merged data, 12 chambers, L182-N287-IC461)` |
| 腔室页签名 / 汇总表腔室名称 | `CH1_CB1` … `CH1_CB12` | `CB1` … `CB12` |
| 腔室页 A 列图片名 | `CH11_CB1_H1.tif` | `CB1_H1.tif` |
| 「数据来源」页签 | —— | **整页删除** |
| PPT 图内标题 / 腔室标签 | `CH11-CB1` | `CB1` |
| 文件名 | `2208_可视化结果.xlsx` / `L120‑N300‑IC5.25%.pptx` | `<条件>_可视化结果.xlsx` / `<条件>.pptx` |

**为什么必须重绘而不是改字**：工作簿内嵌图和 PPT 配图都是 **PNG 位图**，图标题是画在图上的
像素，没有任何「改文字」的办法。所以流程是：从原工作簿腔室页读回时间序列（数据一字不改）
→ 交给 `writer.build_workbook` 重出一份工作簿 → 用 `relabel.rebuild_pptx` 把 PPT 里 12 张图
**原地换二进制**（图片部件的 `_blob`，位置/尺寸/版式完全不动）并改写标签。

**安全与校验**：

- 改写前整份备份到 `<项目根>/_backup/<文件夹>_原版_<时间戳>/`
- 改写后自动校验：页签名、图数（`腔室数×4+3`）、「汇总」页与原版**逐格比对**（只允许腔室名称列变化）、
  全表扫描是否还残留 `CH<k>_CB<j>` 之类的原始溯源
- 失败或校验不通过会明确报错并保留备份，不会静默产出半成品
- 文件夹名不改写；`selected__groups_*.xlsx` 原样保留（它仍是分组索引，里面的
  `源文件名称` 列天然带原始出处）

### 入口⑥ `build_english_version.py`（英文版 21 组 → Group21_English_Version/）

把「20组模拟数据+1组最优模拟」里每组的三个文件整体转为英文版，输出到
`模拟数据处理_part1/Group21_English_Version/<同名子文件夹>/`。依据：
`可视化结果参考模板.xlsx`（页签结构与英文表头）与 `图表文字内容修改意见.pptx`
（图表文字与公式口径）。

```powershell
python build_english_version.py            # 全量（已存在的输出跳过）
python build_english_version.py --force    # 覆盖重算
python build_english_version.py --dry-run  # 只打印计划
python build_english_version.py --only "L30‑N20‑IC0.5%"   # 只转一个文件夹
```

**转换口径**（均已与用户确认）：

| 项 | 口径 |
|---|---|
| 工作簿结构 | 「汇总结果」页（8 列腔室表 + 培养条件块 + Pooled fitting results 11 行 + 3 张合并图）+ `CB-1..CB-n` 腔室页（A:F 数据 + 10 行参数 + 4 张图），无「数据来源」页 |
| 生长曲线 | **全部改 Y = ln(Nrel) 尺度**：散点 ln(实测)，曲线按修改意见公式用已存参数解析生成，**不重新拟合**；纵轴 `Log relative cell number, Y` |
| 参数框 | `λ` 带单位 h；拟合优度写作 `R²_Y`；Krel 仍报告原值（不取对数） |
| 派生参数 | **Tlag = λ（可为负）、Tprolif = t90 − λ**（与修改意见严格一致，与旧版钳位口径不同） |
| 图表标题 | `Growth curve / A_tot,rel / A_mean,rel (merged data, N chambers, L30-N20-DIC75)`；腔室图 `Growth curve (CB-1, <cond>)` 等；横轴 `Cultivation time, t (h)`（t 斜体，mathtext `$t$`）；纵轴 `Log relative cell number, Y`（Y 斜体，`$Y$`） |
| 下标格式 | 图内 `Atot,rel`/`Amean,rel` 一律 mathtext 下标 `$A_{\mathrm{tot,rel}}$` / `$A_{\mathrm{mean,rel}}$`（标题与纵轴标签） |
| 字体 | 图表文字 Times New Roman；数学符号（μmax、Krel、R²_Y、下标）用 stix 数学字体（`prepare()` 里 `mathtext.fontset="stix"`）；工作簿单元格与 PPT 文本框也统一 Times New Roman（对齐模板，原为 Calibri/Arial/微软雅黑） |
| 单元格符号格式 | `core/en_richtext.py`：按模板 run 级复刻富文本——变量字母斜体（μ/K/N/T/t/λ/R/Y/I/C/F/η）、描述性下标正体（max/rel/lag/prolif/plat/d/90/DIC/0）、单位指数上标（h⁻¹/m⁻²/s⁻¹/L⁻¹）、R² 与 NH₄⁺ 用 Unicode 字符。覆盖：汇总表头（粗体 sz11、行高 44）、培养条件块（非粗体 sz10）、Pooled 参数名、CB 页 E 列参数列、selected 表头（粗体 sz12） |
| PPT 符号格式 | `Mean μmax = … h⁻¹` → μ 斜体 + max 下标 + -1 上标（rPr baseline）；`Group i (n = N)` 的 n 斜体；run 保留原粗体/字号 |
| 条件标签 | 文件夹名 `L120‑N160‑IC5.25%` → `L120-N160-DIC320`（IC0.5%/5.25%/10% ↔ DIC75/320/565，编码 ∓1/0/±1；2208 → `L182-N287-DIC461`） |
| 腔室页细节 | 页签 `CB-1..CB-n`；A 列图片名 `CB{k}_H{i}.tif`；状态 `成功→Success` |
| PPT | 页眉「数据组 i（n = N）→ Group i (n = N)」；标签 → `CB-<k>`（经数据来源映射，几何配对）；配图按原像素尺寸重绘替换（Y 尺度英文） |
| selected | `selected_`/`strict_` 页表头中→英，数据原样（源文件名/腔室名属原始溯源，保留）；`selected_N` 页的组合生长曲线图整图重绘（`core/en_combo.py`）：子图标题 `Growth curve (CB-<k>, <cond>)` 不保留原始腔室名称，Y 尺度 + Times 字体，按 selected 行序经 `src_to_index` 定位腔室（μmax 逐位校验），按原图像素尺寸（6848/4562×1144）原位替换 `xl/media/image1.png`，锚点不动 |
| 文件名 | `<cond>_visualization_results[_NN].xlsx` / `<cond>[_NN].pptx` / selected 原名；2208 的 `数据溯源.xlsx` 原样复制 |

**多组文件夹的三方匹配**（如 L120‑N160‑IC5.25% 有 6 组）：xlsx ↔ selected ↔ pptx
按**原始腔室集合**匹配（数据来源页 / selected 页腔室名称列 / PPT 标签三方集合相等），
缺项跳过并在日志中明确列出。pptx 文件名沿用源数据条件、不代表所属组，只作参考。

**注意**：源腔室页 E 列参数名里 μmax 写作 ASCII `uMAX`（汇总页为 `μmax`），
`en_labels.map_params` 已做归一化兼容。

### 入口⑦ `link_selected_summary.py`（selected 页并入培养条件区）

**只处理英文版** `Group21_English_Version/`（中文版不动）。对每个
`selected__groups_X_Y.xlsx`：

1. 按**原始腔室集合**找到同文件夹对应的 `<标签>_visualization_results<suffix>.xlsx`
   （与入口⑥同一套匹配；已处理过的 selected 页腔室列被覆盖，会自动回退读源目录
   同名文件的「腔室名称」列，并把配对结果记到
   `_logs/selected_summary_pairs.json`，便于 `--force` 重跑）；
2. 把可视化工作簿「汇总结果」页 `A1:H<培养条件取值行>`（8 列腔室汇总表 +
   空行 + `Cultivation conditions (0, -1, 0)` 标题 + 三列条件表头 + 三列条件
   取值）逐格复制进 `selected_Y` 页，**替换**原来的 13 列腔室表；
   `core/selected_summary.py` 照搬值/字体/对齐/边框/填充/数字格式与富文本
   （斜体变量、上下标），列宽 A:H 与行高 1 取源页；
3. 该页嵌的组合生长曲线图 `OneCellAnchor` 下移到条件块末行之下第 2 行
   （末行 R → Excel 第 R+2 行），原图所在 212 高的行高随图搬过去、旧行高复位，
   保证图不压住表格；
4. 删除 `selected_Y` 以外的页签（`strict_M` / `summary`），只留一个页签。

```powershell
python link_selected_summary.py --dry-run   # 只打印计划（不写文件）
python link_selected_summary.py             # 执行（已处理过的跳过，幂等）
python link_selected_summary.py --force     # 强制重做
python link_selected_summary.py --only "L120‑N20‑IC5.25%"
```

产物：原地覆盖 20 个 `selected__groups_*.xlsx`（15 个文件夹）；改前整份备份到
`_backup/selected_cond_<时间戳>/<文件夹>/`。「最优解验证工况」文件夹无 selected 文件，
不涉及。校验脚本：`_verify_selected.py`（逐格比对源区域、图片锚点、页签数、残留）。

## 输出结构

```
模拟数据处理_part1/20组模拟数据/
├── L30‑N20‑IC0.5%/L30‑N20‑IC0.5%_可视化结果.xlsx
├── ...（15 个工况文件夹）
└── L120‑N160‑IC5.25%/                       ← 中心点，6 个重复
        L120‑N160‑IC5.25%_可视化结果_01.xlsx
        ..._02 / _03 / _04 / _05 / _06.xlsx
```

每个 xlsx 内含三类页签：

| 页签 | 内容 |
|---|---|
| `汇总` | **三张表 + 3 张图**，布局对齐参考实现 `08_processed_batch.py` 的「汇总结果」（详见下节） |
| `CH1_CB1` … `CH1_CBn` | 原始腔室页签，原样搬运（图片序列 + 参数块）**+ 4 张图**。腔室统一重命名为 `CH1_CB<序号>`，与原始 CH 编号无关 |
| `数据来源` | 两列【腔室名】【原始数据来源】，记录该腔室取自哪个 excel 的哪个页签 |

### 可视化图（共 7 张/工作簿）

参考实现是把 matplotlib 生成的 PNG 用 `ws.add_image` **嵌进单元格**（不是 Excel 原生图表对象），
本管线照此实现，锚点与显示尺寸逐项对齐参考文件的实际输出：

| 位置 | 锚点 | 尺寸 | 图 |
|---|---|---|---|
| 腔室页 | `H2` | 600×400 | `<工况>_CB<n> Cell growth curve fitting`：蓝点实测 + 红拟合曲线 + 参数框 |
| 腔室页 | `H24` | 600×400 | `<工况>_CB<n> data vs merged fitting curve`：该腔室实测 + 黄合并拟合曲线 |
| 腔室页 | `R2` | 600×400 | `<工况>_CB<n> - Relative total cell area variation`（绿） |
| 腔室页 | `R24` | 600×400 | `<工况>_CB<n> - Relative average cell area variation`（紫，散点+折线） |
| 汇总页 | 与「合并拟合结果」同一行，`C` 列起 | 700×500 | `Growth curve (merged data, N chambers, 工况)`：全部腔室实测 + 黄合并曲线 |
| 汇总页 | 同排第 2 张（自动算列） | 700×500 | `Relative total area (merged data, N chambers, 工况)`（绿） |
| 汇总页 | 同排第 3 张（自动算列） | 700×500 | `Relative average cell area (merged data, N chambers, 工况)`（紫） |

> 汇总页 3 张图的**顶边对齐「合并拟合结果」标题所在行**，列位置由 `writer._merged_anchors`
> 按实际列宽累加计算：每张图占 `PLOT_MERGED_SIZE[0]` 像素，之后向右跳过若干整列直到
> 累计宽度 ≥ 图宽 + `PLOT_MERGED_GAP_PX`，因此任何列宽下都不会重叠（18 腔室时落在 `C` / `I` / `O` 列）。
> 图区域未设宽度的列会被统一设为 `PLOT_MERGED_BAND_WIDTH`（默认 8.43 太窄会把图挤在一起）。

**取值口径**（与参考实现一致）：

| 图 | 数据来源 |
|---|---|
| 生长曲线类 | 腔室页 `目标数量` 列（相对数量 Nrel），t = 0…96 h（97 点） |
| 相对总细胞面积 | 腔室页 `总面积(μm²)` 列，源数据已按 0 h 归一化，直接使用 |
| 相对平均细胞面积 | `总面积 ÷ 目标数量` 逐点计算。**已核对**：等于源工作簿「数据汇总」页签的「相对平均细胞面积」列（源文件存为 2 位小数，本管线保留精确商，差异 ≤0.005） |
| 面积类图的时间范围 | 前 96 个点（t = 0…95），对齐参考实现的 `valid_num=96`，见 `config.PLOT_AREA_POINTS` |

**可调项**（`config.py`）：`PLOT_ENABLED`、`PLOT_DPI`（当前 150）、`PLOT_CHAMBER_SIZE`、
`PLOT_MERGED_SIZE`、`PLOT_CHAMBER_ANCHORS`、`PLOT_MERGED_START_COLUMN`、`PLOT_MERGED_GAP_PX`、
`PLOT_MERGED_ROW_OFFSET`、`PLOT_MERGED_BAND_WIDTH`。
图内文字全为英文（标题/图例/坐标轴），不依赖中文字体。

> 体积提示：dpi 只影响 PNG 分辨率，**不影响 Excel 中的显示尺寸**（显示尺寸恒为 600×400 / 700×500）。
> dpi=150 时每个工作簿约 8–10 MB，20 组合计约 170 MB；dpi=300 约 15–17 MB / 组，合计约 300 MB。
> 屏幕查看 150 足够，需要排版印刷时再改回 300。

### 「汇总」页签的三张表

| 表 | 内容 | 出现条件 |
|---|---|---|
| ① 腔室级拟合汇总 | 每个腔室一行，12 个字段（μmax / Krel / Nrel(0) / λ / R² / T_d / F / η / 滞后期 / 对数期 / 稳定期 / t90），取自各腔室页签的参数块 | 总是 |
| ② 未参与拟合数据 | 数据不完整（点数 < 5、含非正值或非数值）的腔室，前置红色分隔行 | 仅当存在这类腔室时；正常数据不出现 |
| ③ 合并拟合结果 | 11 个字段：参与拟合的页签数量、总数据点数量、μmax、Krel、Nrel(0)、λ、R²、滞后期/对数期/稳定期时长、t90 | 总是 |

**表③ 的计算口径**（与参考实现一致）：

- 模型为 Zwietering 修正 Logistic，拟合在 **Y = ln(Nrel)** 尺度：
  `Y(t) = A / (1 + exp(4μ/A·(λ − t) + 2))`，`Nrel(t) = exp(Y(t))`，`Krel = exp(A)`
- **R² 同样在 ln(Nrel) 尺度上计算**，不可与 Nrel 尺度的 R² 混用
- `Nrel(0)` 是拟合曲线在 t=0 处的值（非固定为 1）
- `t90` = 曲线达到平台倍数 90% 的时刻（阈值见 `config.MERGE_STABLE_FRACTION`）

**与参考实现唯一的差别**：不做腔室筛选。参考实现会按 R²≥0.8 做后向剔除 + beam search + 回看修复，
挑一个「最优腔室子集」；本管线按需求跳过该步，**所有数据可用的腔室直接沿时间轴拼接后一次性拟合**。
因此合并 R² 可能低于参考实现（参考实现靠筛选把 R² 抬到 0.8 以上），这是预期结果、不是计算错误。

已用源文件 `20260504/…/CH1_可视化结果_96.xlsx` 的 21 个腔室做过回归验证，本模块复算结果与
参考实现逐项一致：μmax=0.1655、Krel=176.33、Nrel(0)=1.17、λ=11.6、R²=0.8996、
滞后期 11.6 / 对数期 32.79 / 稳定期 51.61、t90=44.39、数据点 2037。

腔室级单拟合亦已核对：单个腔室复算出的 μmax / Krel / λ / R² 与该腔室参数块中 08 脚本写入的值一致
（例：CH1_CB1 → 0.1696 / 128.92 / 9.42 / 0.9860）。

## 入口②：link_group_assets.py（回链分组产物）

`build_20_groups.py` 只搬了原始腔室数据，没带上 batch_grouping 的分组清单与生长曲线页。
本脚本补齐：把每个子文件夹匹配到它的 `selected__groups_*.xlsx`，并把 pptx 里对应页抽出来。

```powershell
python link_group_assets.py                    # 默认：目标已存在则跳过，绝不覆盖
python link_group_assets.py --dry-run          # 只核对，不复制不建 pptx
python link_group_assets.py --on-exist backup      # 已存在先备份成 *_backup_<时间戳>
python link_group_assets.py --on-exist overwrite   # 直接覆盖
python link_group_assets.py --agg-dir ... --out-dir ... --batch-dir 18_batch_20260915
```

**匹配依据**：输出工作簿「数据来源」页签里记录每个腔室取自哪个原始文件/页签，
归一化成 `(日期_文件名, 原腔室名)` 集合，与分组文件 `selected_<n>` 页签的
`(源文件名称, 腔室名称)` 做**集合相等**比对。用集合而非逐行顺序，是因为输出页签
已重排成 `CH1_CB1…CBn`。

两侧键的写法不同，这是最容易踩空的地方：

| 侧 | 键的样子 |
|---|---|
| 分组文件 | `20260531_CH1_可视化结果_96.xlsx` + `CH1_CB5` |
| 输出「数据来源」 | `20260531\数据汇总\…\CH4_可视化结果_96.xlsx｜页签：CH4_CB1` → 归一化为 `20260531_CH4_可视化结果_96.xlsx` + `CH4_CB1` |

**页序**：`selected_groups_growth_curves_2x9.pptx` 第 i 页 ≡ `selected__groups_<i>_<n>.xlsx`（页标题「数据组 i（n = …）」，脚本会解析标题做交叉校验，不符记 WARNING）。
抽取方式是**以源 pptx 为底稿只删不需要的页**，比复制 shape 保真（图、表、版式全保留）。

**⚠️ 子文件夹名通常 ≠ 来源工况**（μmax 是跨条件挑选后重新配对的），所以脚本全库搜索，
新建的 pptx 以**来源工况**命名（如 `L120‑N20‑IC5.25%.pptx` 落在 `L120‑N160‑IC0.5%` 文件夹里）。

本次运行结果：**20 个子文件夹全部唯一命中，0 未匹配**，复制 20 个 xlsx + 新建 20 个 pptx；
`L120‑N160‑IC5.25%` 子文件夹因含 6 组中心点重复，收到 6 个 xlsx + 6 个 pptx。
源目录（数据汇总分析结果）全程只读，已用 `find -newermt` 复核无文件被改动。

## 异常情况处理

脚本遇到以下情况**只记 WARNING 并继续**，不会中断：

- 原始文件缺失 / 页签缺失 → 跳过该腔室，计数后提示
- 腔室数 ≠ 18 → 提示实际数量（batch_grouping 的末组可能是 12 个，属正常）
- 来源工况不存在 / 分组定位失败 → 跳过该组

本次运行：**成功 20 组，6 处告警**，均为"该组实际为 12 个腔室"（对应 `selected__groups_*_12.xlsx`），非数据缺陷。
20 组的合并拟合全部成功，无腔室被计入「未参与拟合数据」。
