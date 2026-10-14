# Microalgae_Identification_YOLOv11

微藻显微图像的**实例分割识别**与**实验数据分析**一体化仓库。

- **微藻识别**：从原始显微图按流道（CH）分拣，到 YOLO 分割识别，再到按流道/腔室（CB）汇总、标准化、生长曲线拟合，形成一条可批量复现的流水线。
- **数据分析**：在识别产物之上做工况归集、同批次同质分组、响应面建模与模拟数据构建。

两个模块解耦：识别模块不依赖数据分析模块；数据分析模块读取识别产出的 Excel，可独立运行。

---

## 0. 快速开始（Ubuntu 22.04 + conda）

全新机器从零到跑完识别流程，六步：

```bash
# ① 系统依赖（OpenCV 运行库 + 中文字体）
sudo apt update && sudo apt install -y build-essential git wget \
     libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 fonts-wqy-zenhei fonts-noto-cjk

# ② Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
"$HOME/miniconda3/bin/conda" init bash
source ~/.bashrc          # 或关闭重开终端，让 conda 命令生效

# ③ 克隆（完整历史 1.5 GB；只想跑流程用 --depth 1，约 240 MB）
git clone --depth 1 git@github.com:PolarisYong/Microalgae_Identification_YOLOv11.git
cd Microalgae_Identification_YOLOv11

# ④ 环境  ← --extra-index-url 不能省
conda create -n microalgae python=3.11 pip -y && conda activate microalgae
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126

# ⑤ 配置批次（默认 photo_root 是旧服务器路径，必须改）
cp configs/pipeline.example.yaml configs/batch_20261001.yaml
python microalgae_identification/run_pipeline.py --config configs/batch_20261001.yaml --dry-run

# ⑥ 跑
python microalgae_identification/run_pipeline.py --config configs/batch_20261001.yaml
```

三点必须注意：

- `--extra-index-url https://download.pytorch.org/whl/cu126` **不能省**：`torch==2.7.1+cu126`、
  `torchvision==0.22.1+cu126` 是 CUDA 专用轮子，PyPI 上没有，否则 pip 报
  `No matching distribution found`。也不需要另装 CUDA Toolkit，驱动 ≥ 525 即可。
- 换机器 / 换批次**只改配置里的 `photo_root`**，其它路径不用动。
- 默认权重 `runs/segment/train4/weights/best.pt` 与腔室模板
  `microalgae_identification/detection/chamber_template.png` 均已入库，克隆后免下载。

> 完整版（SSH 密钥配置、环境校验、阶段级续跑、排错表、平台差异清单）见
> [`docs/ubuntu_setup.md`](docs/ubuntu_setup.md)。

---

## 1. 目录结构

```
Microalgae_Identification_YOLOv11/
├── configs/                              ★ 批次配置（换数据只改这里）
│   ├── pipeline.example.yaml                 配置模板与字段说明
│   └── batch_20261001.yaml                   20261001 批次实例
│
├── microalgae_identification/            ★ 模块一：微藻识别
│   ├── config.py                             配置加载（YAML → 环境变量 → 命令行）
│   ├── run_pipeline.py                       一键执行入口
│   ├── preprocessing/                        ① 原始图分拣与归集
│   │   ├── 01_select_ch.py                      周目录重排、补 (001) 后缀
│   │   └── 02_move_photoes.py                   HEIDSTAR.COM.x.y → CH{z}
│   ├── detection/                            ② 识别
│   │   ├── detect_batch.py                      YOLO 分割 + 腔室掩膜过滤（核心）
│   │   └── chamber_template.png                腔室模板图（必须与脚本同目录）
│   └── postprocessing/                       ③④⑤ 汇总 → 标准化 → 可视化
│       ├── 00_merge_excel_summary.py
│       ├── 01_standardize_relative_cell_number.py
│       └── 08_growth_curve_batch.py
│
├── data_analysis/                        ★ 模块二：数据分析
│   ├── exp_batch_aggregator/                 按实验工况横向归集各批次结果
│   ├── batch_grouping/                       单工况内按 μmax 做同质分组
│   ├── ccf_chlorella/                        CCF 响应面建模与寻优
│   ├── simdata_pipeline/                     模拟数据构建与回链
│   ├── kinetics/                             生长动力学并行实现（Bootstrap 置信区间）
│   └── exploratory/                          探索性分析与论文配图
│
├── training/                             训练与论文配图（非主流程）
│   ├── train_model.py                         YOLO 分割模型训练
│   ├── dataset_label/                         LabelMe ↔ YOLO 标注互转、数据集划分
│   ├── figures/                               训练曲线与论文配图脚本
│   └── dump_conf_curves.py                    置信度扫描
│
├── legacy/                               历史版本与一次性脚本（已归档，不参与主流程）
│   ├── detection/                            microalgae_dense_hybrid_01~03
│   ├── processed_versions/                   02 / 05 / 07_processed_batch.py
│   └── research/                             实验脚本与一次性数据处理脚本
│
├── docs/                                 双通道混合推理说明 + 迁移映射表 + Ubuntu 部署指南
├── runs/                                 训练产出（含 detect_batch.py 默认权重）
├── dataset/                              训练数据集（已在 .gitignore 中排除内容）
├── requirements.txt                      识别 + 数据分析合并依赖
└── requirements-data-analysis.txt        数据分析模块最小依赖（可选单独建环境）
```

---

## 2. 环境依赖

| 场景 | Python | 依赖文件 | 关键包 |
|---|---|---|---|
| 识别主流程 | 3.11+（Linux + CUDA 实测） | `requirements.txt` | `torch==2.7.1+cu126`、`ultralytics==8.4.60`、`opencv-python`、`pandas`、`openpyxl`、`matplotlib`、`scipy`、`tqdm`、PyYAML |
| 数据分析 | 3.11（3.11.15 验证） | `requirements-data-analysis.txt` | `numpy`、`scipy`、`matplotlib`、`pandas`、`openpyxl`、`xlrd`、`pillow`、`python-pptx` |

```bash
# 识别环境（Linux 服务器，需 CUDA）
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126

# 数据分析环境（可与上面共用，也可单独建）
conda create -n lpr_dp python=3.11 pip -y
conda activate lpr_dp
pip install -r requirements-data-analysis.txt
```

### 2.1 依赖要点（跨平台）

- **`--extra-index-url` 不能省。** `torch==2.7.1+cu126`、`torchvision==0.22.1+cu126` 是 CUDA 12.6 专用轮子，PyPI 上没有，必须走 PyTorch 官方索引，否则 pip 报 `No matching distribution found`。
- **不需要在系统里另装 CUDA Toolkit。** cu126 的 CUDA 运行库随 pip 包一起安装，只要 NVIDIA 驱动 ≥ 525 即可（`nvidia-smi` 确认）。
- **默认权重已入库。** `runs/segment/train4/weights/best.pt`（约 22 MB）随仓库一起克隆，无需另外下载。
- **没有 GPU 也能跑**（自动回退 CPU），但全图 + 分块双通道推理会很慢。
- **克隆体积偏大**是因为历史提交里曾放进过大数据集产物；只要最新代码用 `--depth 1` 即可，代价是看不到提交历史。
- Ubuntu 22.04 的完整部署步骤见 [§0 快速开始](#0-快速开始ubuntu-2204--conda) 与 [`docs/ubuntu_setup.md`](docs/ubuntu_setup.md)。

> 字体：`detect_batch.py` 的标注文字优先使用文泉驿正黑 / PingFang / DejaVuSans，
> 全部缺失时回退到 PIL 默认字体（不影响识别数量与面积，仅影响叠加图上的中文显示）。
> Ubuntu 上装 `fonts-wqy-zenhei` 即可命中第一种。

---

## 3. 识别主流程

### 3.1 流程总览

```
原始显微图（周目录 2026-Week-N）
   │  ① 01_select_ch.py          周目录重排，子目录补 (001) 序号后缀
   │  ① 02_move_photoes.py       HEIDSTAR.COM.x.y → CH{z}，图重命名 CH{z}_CB{c}_H{h}.tif
   ▼
<photo_root>/CH1 … CH12/IMG001xNNN/
   │  ② detect_batch.py          YOLO 分割；全图 + 分块双通道推理；腔室掩膜过滤
   ▼
CH{z}/IMG001xNNN/results/CH{z}_CB{c}.xlsx    （含「汇总统计」页：图片名称/处理状态/目标数量/总面积）
   │  ③ 00_merge_excel_summary.py     按流道合并 → 数据汇总/01_原始数据/CH{n}.xlsx
   │  ④ 01_standardize_..._number.py  以首行归一化 → 02_标准化数据/CH{n}_标准化.xlsx
   │  ⑤ 08_growth_curve_batch.py      修正 Logistic 拟合 → 03_可视化结果/<N>小时/CH{n}_可视化结果_<N>.xlsx
   ▼
按流道/腔室的生长曲线与动力学参数（μmax、Krel、Nrel(0)、滞后期/对数期/稳定期时长…）
```

### 3.2 一次配置，跑完整条流程

**第 1 步：写批次配置。** 复制模板并填写日期、数据根目录、流道与工况映射：

```bash
cp configs/pipeline.example.yaml configs/batch_20261001.yaml
```

```yaml
batch:
  date: "20261001"
  photo_root: "/home/smart/github_projects/photoes/{date}"   # {date} 会被上面日期替换
model:
  weights: ""                    # 留空 = runs/segment/train4/weights/best.pt
  chamber_template: ""           # 留空 = 识别脚本同目录下的 chamber_template.png
detection:
  channels: 12                   # CH1…CH12；也可写 [1,2,3] 或 "1-12"
summary:
  valid_hours: "96"
  channel_conditions:            # ⑤ 步需要：流道 → 实验工况
    1: "N287"
    2: "N325"
    # …
```

**第 2 步：一键执行。**

```bash
python microalgae_identification/run_pipeline.py --config configs/batch_20261001.yaml
```

常用变体：

```bash
# 先检查配置解析出的路径是否正确，不实际执行
python microalgae_identification/run_pipeline.py --config configs/batch_20261001.yaml --dry-run

# 图片已手工分拣好，跳过前两步
python microalgae_identification/run_pipeline.py --config configs/batch_20261001.yaml --from detect

# 只重跑汇总（例如识别结果有更新）
python microalgae_identification/run_pipeline.py --config configs/batch_20261001.yaml --only merge

# 临时换批次，不写配置文件
python microalgae_identification/run_pipeline.py --date 20261015 --hours 96 --channels 1-12
```

失败即中断，修复后可用 `--from <阶段>` 续跑。阶段名：`sort` `move` `detect` `merge` `standardize` `visualize`。

### 3.3 单独运行某一步

每个脚本都可独立运行，并共享同一套配置参数：

```bash
python microalgae_identification/preprocessing/01_select_ch.py   --config configs/batch_20261001.yaml
python microalgae_identification/preprocessing/02_move_photoes.py --config configs/batch_20261001.yaml
python microalgae_identification/detection/detect_batch.py       --config configs/batch_20261001.yaml
python microalgae_identification/postprocessing/00_merge_excel_summary.py             --config configs/batch_20261001.yaml
python microalgae_identification/postprocessing/01_standardize_relative_cell_number.py --config configs/batch_20261001.yaml
python microalgae_identification/postprocessing/08_growth_curve_batch.py               --config configs/batch_20261001.yaml
```

参数优先级：**命令行 > YAML 配置 > 内置默认值**。内置默认值与重构前脚本中的硬编码逐字一致，
因此不带任何参数运行时，行为与重构前完全相同。

| 参数 | 作用 |
|---|---|
| `--config` | 指定批次 YAML 文件 |
| `--date` | 批次日期，同时决定 `photo_root` 中的 `{date}` |
| `--photo-root` | 批次根目录（含 CH1…CHn） |
| `--model` | 模型权重 `.pt` 路径 |
| `--chamber-template` | 腔室模板图路径 |
| `--channels` | 流道列表：`1-12` / `1,3,5` / `12` |
| `--hours` | 培养时长（小时），决定 03_可视化结果 下的目录名与文件名后缀 |

### 3.4 产物命名约定（历史批次兼容，不可更改）

| 阶段 | 产物 | 说明 |
|---|---|---|
| ① | `CH{z}_CB{c}_H{h}.tif` | `z`=流道，`c`=腔室，`h`=时序序号 |
| ② | `CH{z}/IMG001x{c}/results/CH{z}_CB{c}.xlsx` | 每腔室一本工作簿，含「汇总统计」页 |
| ② | `results/R_<原图名>` | 原图 / 掩膜 / 带编号掩膜三联图 |
| ② | `results/shared_bgfit_*.png` | 腔室模板配准调试图 |
| ③ | `数据汇总/01_原始数据/CH{n}.xlsx` | 页签名 = 腔室标识 `CH{n}_CB{m}` |
| ④ | `数据汇总/02_标准化数据/CH{n}_标准化.xlsx` | 新增「相对平均细胞面积」列与「数据汇总」页 |
| ⑤ | `数据汇总/03_可视化结果/<N>小时/CH{n}_可视化结果_<N>.xlsx` | 腔室级拟合 + 合并拟合 + 内嵌图 |

> ③ 步会跳过「目标数量最大值 ≤ 3」或「存在 0 值」的腔室，这是原有筛选规则，重构未改动。

---

## 4. 数据分析模块

详见 [`data_analysis/README.md`](data_analysis/README.md)。四个子模块按实验分析的自然顺序衔接，
前一模块的输出目录即后一模块的输入目录：

```
原始实验数据（按日期/流道组织）
   │  exp_batch_aggregator   按实验工况横向归集
   ▼
实验结果汇总/<工况标签>/<YYYYMMDD>_CH<n>_可视化结果_<N>.xlsx
   │  batch_grouping         单工况内按 μmax 纵向同质分组
   ▼
分组 Excel + 标记工作簿 + 2×9 汇总 PPT + μmax 总览
   │  ccf_chlorella          跨工况响应面建模与寻优
   │  simdata_pipeline       回溯到原始腔室数据重新打包
   ▼
RSM 模型 / 驻点 / 交互作用图 / 20 组模拟数据
```

> 大数据工作区（`batch_grouping/data/`、`模拟数据处理_part1/` 等）已列入 `.gitignore`，
> 不进入版本库，但本地目录结构保持不变。`simdata_pipeline/config.py` 的根路径由文件位置自动推导。

---

## 5. 归档说明（legacy/）

以下脚本已移出主目录，仅作历史追溯，不参与当前流程：

- `legacy/detection/` —— `microalgae_dense_hybrid_01~03.py`，`detect_batch.py` 的前身
- `legacy/processed_versions/` —— `02/05/07_processed_batch.py`，`08_growth_curve_batch.py` 的算法迭代快照，各自绑定历史批次的工况映射
- `legacy/research/` —— 训练推理实验脚本、一次性改名/提取脚本、测试残留

已删除：`test_model.py` ~ `test_model_16.py`、`microalgae_dense_hybrid_00.py`。

完整的「旧路径 → 新路径」对照见 [`docs/migration_map.md`](docs/migration_map.md)。

---

## 6. 常见问答

**换了机器 / 换了数据盘要改哪里？**
只改 `configs/<批次>.yaml` 的 `photo_root`，其余路径全部由它派生。

**识别结果变了？**
先确认 `chamber_template.png` 与 `detect_batch.py` 在同一目录（或用 `model.chamber_template` 显式指路）。
腔室掩膜过滤依赖这张模板图，缺失会使过滤失效、结果偏移。

**某个流道被跳过了？**
`08_growth_curve_batch.py` 只处理在 `summary.channel_conditions` 中登记过的流道，
未登记的会打印「未配置对应实验参数」并跳过。
