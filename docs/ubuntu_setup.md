# 在 Ubuntu 22.04 上从零部署（conda 环境管理）

本文面向**一台全新的 Ubuntu 22.04 机器**，从装 conda 到跑完整条微藻识别流程。
识别主流程（模块一）是重点；数据分析模块（模块二）见第 8 节，可选。

---

## 0. 前置条件

| 项目 | 要求 | 说明 |
|---|---|---|
| 操作系统 | Ubuntu 22.04 LTS | glibc 2.35，满足 `manylinux_2_28` 轮子要求 |
| GPU（推荐） | NVIDIA 显卡 + 驱动 ≥ 525 | 只需驱动，**不需要**在系统里另装 CUDA Toolkit |
| 无 GPU | 可以跑 | 自动回退 CPU，但全图 + 分块双通道推理很慢 |
| 磁盘 | ≥ 5 GB | 完整克隆约 1.5 GB，conda 环境约 3 GB |
| 网络 | 能访问 GitHub 与 PyPI | 需要装约 1.5 GB 的 pip 包 |

---

## 1. 系统依赖

OpenCV（`opencv-python`，非 headless 版）在 Linux 上需要图形运行库，中文字体用于识别叠加图上的标注：

```bash
sudo apt update
sudo apt install -y build-essential git wget curl \
     libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
     fonts-wqy-zenhei fonts-noto-cjk
```

---

## 2. 安装 Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
"$HOME/miniconda3/bin/conda" init bash
exec bash          # 或关闭重开终端，让 conda 命令生效
conda --version    # 验证
```

> 如果用的是 `zsh`，把上面两处 `bash` 换成 `zsh`。

---

## 3. 克隆仓库

仓库历史里曾经提交过大数据集产物，所以**完整克隆的 pack 约 1.5 GB**（工作区本身只有约 240 MB）。

```bash
# 方式 A：只要跑流程（推荐，约 240 MB，无历史）
git clone --depth 1 git@github.com:PolarisYong/Microalgae_Identification_YOLOv11.git

# 方式 B：完整历史（约 1.5 GB，可 git log / 回溯重构前后对比）
git clone git@github.com:PolarisYong/Microalgae_Identification_YOLOv11.git

cd Microalgae_Identification_YOLOv11
```

走 HTTPS 则用：
```bash
git clone --depth 1 https://github.com/PolarisYong/Microalgae_Identification_YOLOv11.git
```

**SSH 方式需要先配好密钥**（否则 `Permission denied (publickey)`）：
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub      # 复制内容到 GitHub → Settings → SSH and GPG keys
ssh -T git@github.com          # 见到 "Hi xxx!" 即成功
```

> 用 `--depth 1` 克隆后若想补齐完整历史：`git fetch --unshallow`。

**克隆后确认这几样东西在**（它们都已入库，不需要额外下载）：
```bash
ls microalgae_identification/detection/chamber_template.png   # 腔室掩膜模板，识别必需
ls runs/segment/train4/weights/best.pt                        # 默认权重（约 22 MB）
ls configs/pipeline.example.yaml                              # 批次配置模板
```

---

## 4. 创建 conda 环境并安装依赖

```bash
conda create -n microalgae python=3.11 pip -y
conda activate microalgae

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

### ⚠️ `--extra-index-url` 不能省

`requirements.txt` 里锁的是 CUDA 专用轮子：

```
torch==2.7.1+cu126
torchvision==0.22.1+cu126
```

这两个版本**只存在于 PyTorch 官方索引**，PyPI 上没有。少了 `--extra-index-url`，pip 会直接报：

```
ERROR: Could not find a version that satisfies the requirement torch==2.7.1+cu126
```

已核对过索引里确实有 Linux Python 3.11 的轮子：
- `torch-2.7.1+cu126-cp311-cp311-manylinux_2_28_x86_64.whl`
- `torchvision-0.22.1+cu126-cp311-cp311-manylinux_2_28_x86_64.whl`

> 本机是 Windows 时同上，只是轮子换成 `win_amd64`，索引地址不变。

### 国内网络加速（可选）

```bash
pip install -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://download.pytorch.org/whl/cu126
```

---

## 5. 验证安装

```bash
conda activate microalgae
nvidia-smi                                  # 驱动与显卡
python -c "import torch, ultralytics, cv2, yaml; \
print('torch', torch.__version__); \
print('cuda  ', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''); \
print('ultralytics', ultralytics.__version__)"
```

期望输出：
```
torch 2.7.1+cu126
cuda   True  NVIDIA GeForce RTX ...
ultralytics 8.4.60
```

再确认命令行入口能解析配置（不执行任何计算）：
```bash
python microalgae_identification/run_pipeline.py --dry-run
```
正常会打印 `[批次配置]` 与 6 个阶段的待执行脚本路径。

---

## 6. 准备批次数据与配置

### 6.1 数据放哪

识别流程要求的目录结构（第 ①② 步会自动生成它）：

```
<photo_root>/              ← 批次根目录
├── 2026-Week-1/           ← 原始周目录（第 ① 步的输入）
├── CH1/IMG001x001/…       ← 第 ② 步之后的标准结构
├── CH2/…
└── CH12/…
```

图名格式：`CH{z}_CB{c}_H{h}.tif`（z=流道，c=腔室，h=小时）。

### 6.2 写批次配置

```bash
cp configs/pipeline.example.yaml configs/batch_20261001.yaml
```

```yaml
batch:
  date: "20261001"
  photo_root: "/home/yourname/data/photoes/{date}"   # {date} 会被上面的日期替换
model:
  weights: ""                  # 留空 = runs/segment/train4/weights/best.pt
  chamber_template: ""         # 留空 = 识别脚本同目录下的 chamber_template.png
detection:
  channels: 12                 # CH1…CH12；也可写 [1,2,3] 或 "1-12"
summary:
  valid_hours: "96"
  channel_conditions:          # 第 ⑤ 步需要：流道 → 实验工况
    1: "N287"
    2: "N325"
    # … 12 个流道写全
```

> **默认的 `photo_root` 是 `/home/smart/github_projects/photoes/{date}`，是旧服务器的路径，务必改成自己机器上的真实路径。**
> 除了这里，其它地方都不需要改代码里的路径 —— 这正是本次重构解决的"每换一批数据就要手工改脚本"问题。

**先干跑一次，核对解析出来的路径：**
```bash
python microalgae_identification/run_pipeline.py --config configs/batch_20261001.yaml --dry-run
```

---

## 7. 运行识别主流程

```bash
conda activate microalgae
cd Microalgae_Identification_YOLOv11

# 全流程一键跑（6 个阶段）
python microalgae_identification/run_pipeline.py --config configs/batch_20261001.yaml
```

阶段与脚本对应关系：

| 阶段名 | 脚本 | 作用 |
|---|---|---|
| `sort` | `preprocessing/01_select_ch.py` | 周目录重排，子目录补 `(001)` 序号后缀 |
| `move` | `preprocessing/02_move_photoes.py` | `HEIDSTAR.COM.x.y` → `CH{z}`，图重命名 |
| `detect` | `detection/detect_batch.py` | YOLO 分割；全图 + 分块双通道；腔室掩膜过滤 |
| `merge` | `postprocessing/00_merge_excel_summary.py` | 按流道合并 → `数据汇总/01_原始数据/CH{n}.xlsx` |
| `standardize` | `postprocessing/01_standardize_relative_cell_number.py` | 首行归一化 → `02_标准化数据/CH{n}_标准化.xlsx` |
| `visualize` | `postprocessing/08_growth_curve_batch.py` | 修正 Logistic 拟合 → `03_可视化结果/<N>小时/` |

常用变体：

```bash
# 图片已手工分拣好，跳过前两步
... --from detect

# 只重跑汇总（识别结果有更新时）
... --only merge

# 只跑到识别为止
... --to detect

# 临时换批次，不写配置文件（显式给根目录时不会随 date 重算）
... --date 20261015 --photo-root /home/yourname/data/photoes/20261015 --hours 96 --channels 1-12

# 给某个阶段透传额外参数
... --only detect --extra-args "--conf 0.3"
```

失败即中断，修好后用 `--from <阶段>` 续跑。

**产物命名与历史批次完全一致**，新旧批次可以直接衔接比较。

---

## 8. 数据分析模块（可选，单独环境）

```bash
conda create -n lpr_dp python=3.11 pip -y
conda activate lpr_dp
pip install -r requirements-data-analysis.txt
```

---

## 9. 常见问题排查

| 现象 | 原因 | 解决 |
|---|---|---|
| `Could not find a version ... torch==2.7.1+cu126` | 没加 PyTorch 索引 | 加 `--extra-index-url https://download.pytorch.org/whl/cu126` |
| `ImportError: libGL.so.1: cannot open shared object file` | 缺 OpenCV 运行库 | `sudo apt install -y libgl1 libglib2.0-0` |
| 叠加图上中文是方框 | 缺中文字体 | `sudo apt install -y fonts-wqy-zenhei`，然后 `rm -rf ~/.cache/matplotlib` |
| `torch.cuda.is_available()` 为 `False` | 装成了 CPU 版 / 驱动太旧 | 确认 `torch.__version__` 带 `+cu126`；`nvidia-smi` 看驱动 |
| detect 阶段找不到腔室模板 | 模板不在脚本同目录 | `--chamber-template /path/to/chamber_template.png` |
| 换机器后路径不对 | `photo_root` 没改 | 只改配置里的 `photo_root`，不要改脚本 |
| 新批次面积和历史批次对不上 | 标定值不同 | 2026-10 起用 172.5 × 144.31 μm（旧批次 44.3 × 42.8），跨批次比较前需统一标定 |
| `git clone` 太慢 / 太大 | 历史里有大数据产物 | 用 `--depth 1` |

---

## 10. 已知的平台差异

本项目原先在 Windows 上开发，以下位置仍留有 Windows 绝对路径，**在 Ubuntu 上跑对应脚本前需要改成相对路径或本机路径**（不影响识别主流程）：

| 文件 | 位置 |
|---|---|
| `data.yaml` | `path: E:\...\dataset`（仅训练时需要） |
| `data_analysis/ccf_chlorella/optimum_meaning.py` | `BASE = r"E:\..."` |
| `data_analysis/ccf_chlorella/optimum_meaning2.py` | `BASE = r"E:\..."` |
| `data_analysis/ccf_chlorella/prediction_interval.py` | `BASE = r"E:\..."` |
| `data_analysis/ccf_chlorella/_archive/ccf_simulate.py` | `BASE = "E:/..."` |
| `data_analysis/exploratory/paper_figures/V4.1 …最新.py` | 数据集读路径 |

改法（推荐）：
```python
from pathlib import Path
BASE = Path(__file__).resolve().parent
```

识别主流程（`microalgae_identification/`）已全部改为相对路径 + 集中配置，**不存在这个问题**。
