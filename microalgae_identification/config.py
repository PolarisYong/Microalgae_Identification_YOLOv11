#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批次配置中心 —— 微藻识别主流程的唯一配置入口。

设计目的
--------
重构前，每换一批数据都要手工改 6 个脚本里的日期与绝对路径：

    01_select_ch.py / 02_move_photoes.py / detect_batch.py /
    00_merge_excel_summary.py / 01_standardize_relative_cell_number.py /
    08_growth_curve_batch.py

现在全部收敛到 ``configs/<批次>.yaml``，脚本顶部只读取本模块给出的配置对象。

取值优先级（高 → 低）
--------------------
1. 命令行参数        ``--config`` ``--date`` ``--photo-root`` ``--model`` ``--channels`` ``--hours``
2. 环境变量          ``MICROALGAE_PIPELINE_CONFIG`` 指向的 YAML，或 ``MICROALGAE_*`` 单值覆盖
3. YAML 配置文件     ``--config`` 指定，缺省读 ``configs/pipeline.yaml``
4. 内置默认值        与重构前脚本中的硬编码**逐字一致**，保证不配置时行为完全不变

用法
----
各步骤脚本内::

    from config import load_config
    cfg = load_config()
    print(cfg.photo_root, cfg.channels)

命令行::

    python run_pipeline.py --config configs/batch_20261001.yaml
    python postprocessing/08_growth_curve_batch.py --date 20261001 --hours 96
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

try:  # PyYAML 已在 requirements.txt 中锁定（PyYAML==6.0.2）
    import yaml
except ImportError:  # pragma: no cover - 允许无 YAML 环境下靠 CLI / 默认值运行
    yaml = None

# ------------------------------------------------------------------ 路径锚点
PACKAGE_ROOT = Path(__file__).resolve().parent          # microalgae_identification/
REPO_ROOT = PACKAGE_ROOT.parent                          # 仓库根
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_CONFIG_FILE = CONFIG_DIR / "pipeline.yaml"
ENV_CONFIG_FILE = "MICROALGAE_PIPELINE_CONFIG"

# ------------------------------------------------------------------ 内置默认值
# 这三行与重构前脚本内的硬编码完全一致，任何一项未被配置覆盖时都回退到这里。
DEFAULT_BATCH_DATE = "20261001"
DEFAULT_PHOTO_ROOT_TEMPLATE = "/home/smart/github_projects/photoes/{date}"
DEFAULT_MODEL_WEIGHTS = str(REPO_ROOT / "runs" / "segment" / "train4" / "weights" / "best.pt")
DEFAULT_CHANNELS: List[int] = list(range(1, 13))         # CH1 … CH12
DEFAULT_VALID_HOURS = "96"

# 汇总目录约定（数据盘上的固定层级，历史批次已按此组织，不可更改）
RAW_SUMMARY_DIRNAME = "01_原始数据"
STANDARDIZED_DIRNAME = "02_标准化数据"
VISUALIZATION_DIRNAME = "03_可视化结果"

# 08 步的 CH → 实验工况映射默认值，取自重构前 08_processed_batch.py 的活跃配置块
DEFAULT_CHANNEL_CONDITIONS: Dict[int, str] = {
    1: "N287", 2: "N325", 3: "N400", 4: "N450", 5: "N500", 6: "N550",
    7: "N600", 8: "N700", 9: "N800", 10: "N900", 11: "N600", 12: "N325",
}


@dataclass
class BatchConfig:
    """一次批次运行所需的全部路径与参数。"""

    batch_date: str = DEFAULT_BATCH_DATE
    photo_root: str = ""
    model_weights: str = DEFAULT_MODEL_WEIGHTS
    chamber_template: str = ""
    channels: List[int] = field(default_factory=lambda: list(DEFAULT_CHANNELS))
    valid_hours: str = DEFAULT_VALID_HOURS
    channel_conditions: Dict[int, str] = field(default_factory=dict)
    config_path: Optional[str] = None

    # photo_root 是否由「含 {date} 的模板」派生而来。是则每次日期变化后重算，
    # 避免 --date 只改了日期、目录却仍指上一批次。
    _photo_root_is_template: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.photo_root:
            if "{date}" in self.photo_root:
                self._photo_root_is_template = True
        else:
            self._photo_root_is_template = True

        if not self.photo_root or self._photo_root_is_template:
            self.photo_root = DEFAULT_PHOTO_ROOT_TEMPLATE.format(date=self.batch_date)
            self._photo_root_is_template = True

        if not self.channels:
            self.channels = list(DEFAULT_CHANNELS)
        if not self.valid_hours:
            self.valid_hours = DEFAULT_VALID_HOURS

    # -------------------------------------------------------------- 派生路径
    @property
    def channel_dirs(self) -> List[str]:
        """识别输入目录：<photo_root>/CH<n>，顺序与 channels 一致。"""
        return [str(Path(self.photo_root) / f"CH{c}") for c in self.channels]

    @property
    def summary_root(self) -> str:
        return str(Path(self.photo_root) / "数据汇总")

    @property
    def raw_summary_dir(self) -> str:
        """00 步产物目录：数据汇总/01_原始数据"""
        return str(Path(self.summary_root) / RAW_SUMMARY_DIRNAME)

    @property
    def standardized_dir(self) -> str:
        """01 步产物目录：数据汇总/02_标准化数据"""
        return str(Path(self.summary_root) / STANDARDIZED_DIRNAME)

    @property
    def visualization_dir(self) -> str:
        """08 步产物目录：数据汇总/03_可视化结果/<N>小时"""
        return str(Path(self.summary_root) / VISUALIZATION_DIRNAME / f"{self.valid_hours}小时")

    @property
    def active_channel_conditions(self) -> Dict[int, str]:
        """08 步使用的 CH → 工况映射；未配置时回退到内置默认值。"""
        return self.channel_conditions or dict(DEFAULT_CHANNEL_CONDITIONS)

    def describe(self) -> str:
        lines = [
            f"批次日期      : {self.batch_date}",
            f"批次根目录    : {self.photo_root}",
            f"模型权重      : {self.model_weights}",
            f"流道列表      : {', '.join('CH' + str(c) for c in self.channels)}",
            f"培养时长(h)   : {self.valid_hours}",
            f"原始汇总目录  : {self.raw_summary_dir}",
            f"标准化目录    : {self.standardized_dir}",
            f"可视化结果目录: {self.visualization_dir}",
        ]
        if self.chamber_template:
            lines.append(f"腔室模板图    : {self.chamber_template}")
        if self.config_path:
            lines.append(f"配置文件      : {self.config_path}")
        return "\n".join(lines)


# ------------------------------------------------------------------ 读取 YAML
def _read_yaml(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError(
            f"未安装 PyYAML，无法读取配置文件 {path}。请执行 pip install pyyaml，"
            f"或改用命令行参数 / 环境变量传参。"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件格式错误，顶层应为键值映射：{path}")
    return data


def _config_from_dict(data: dict, config_path: Optional[str] = None) -> BatchConfig:
    batch = data.get("batch") or {}
    model = data.get("model") or {}
    detection = data.get("detection") or {}
    summary = data.get("summary") or {}

    channels = detection.get("channels")
    if isinstance(channels, int):                      # 允许写 channels: 12 → CH1..CH12
        channels = list(range(1, channels + 1))

    conditions = summary.get("channel_conditions") or {}
    conditions = {int(k): str(v) for k, v in conditions.items()} if conditions else {}

    return BatchConfig(
        batch_date=str(batch.get("date") or DEFAULT_BATCH_DATE),
        photo_root=str(batch.get("photo_root") or ""),
        model_weights=str(model.get("weights") or DEFAULT_MODEL_WEIGHTS),
        chamber_template=str(model.get("chamber_template") or ""),
        channels=[int(c) for c in channels] if channels else list(DEFAULT_CHANNELS),
        valid_hours=str(summary.get("valid_hours") or DEFAULT_VALID_HOURS),
        channel_conditions=conditions,
        config_path=config_path,
    )


def _parse_channels(raw: str) -> List[int]:
    """解析 "1-12" / "1,2,3" / "12"（后者等价 1..12）三种写法。"""
    raw = raw.strip()
    if "-" in raw:
        start, end = raw.split("-", 1)
        return list(range(int(start), int(end) + 1))
    if "," in raw:
        return [int(x) for x in raw.split(",") if x.strip()]
    if raw.isdigit() and len(raw) <= 2 and int(raw) < 12:   # 单数字视为上限
        return list(range(1, int(raw) + 1))
    return [int(x) for x in raw.replace(" ", "").split(",") if x]


def add_config_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """给任意步骤脚本挂上统一的配置参数。"""
    parser.add_argument("--config", help="批次 YAML 配置文件（缺省 configs/pipeline.yaml）")
    parser.add_argument("--date", dest="batch_date", help="批次日期，如 20261001")
    parser.add_argument("--photo-root", help="批次根目录（含 CH1..CHn 的目录）")
    parser.add_argument("--model", dest="model_weights", help="模型权重 .pt 路径")
    parser.add_argument("--chamber-template", help="腔室模板图 chamber_template.png 路径")
    parser.add_argument("--channels", help="流道列表：1-12 / 1,3,5 / 12")
    parser.add_argument("--hours", dest="valid_hours", help="培养时长（小时），如 96")
    return parser


def load_config(
    argv: Optional[Sequence[str]] = None,
    parser: Optional[argparse.ArgumentParser] = None,
    allow_unknown: bool = True,
) -> BatchConfig:
    """按「YAML → 环境变量 → 命令行」的优先级构建配置。

    ``allow_unknown=True`` 时忽略步骤脚本自己的额外参数，便于各脚本复用本函数。
    """
    if parser is None:
        parser = argparse.ArgumentParser(add_help=False)
        add_config_arguments(parser)
    known, _unknown = parser.parse_known_args(list(argv) if argv is not None else None)

    # 1) 定位 YAML：命令行 > 环境变量 > 默认文件
    config_path = known.config or os.environ.get(ENV_CONFIG_FILE) or ""
    if not config_path and DEFAULT_CONFIG_FILE.exists():
        config_path = str(DEFAULT_CONFIG_FILE)

    if config_path:
        cfg = _config_from_dict(_read_yaml(Path(config_path)), config_path)
    else:
        cfg = BatchConfig()

    # 2) 命令行覆盖
    if known.batch_date:
        cfg.batch_date = str(known.batch_date)
    if known.photo_root:
        cfg.photo_root = str(known.photo_root)
        # 显式给出的根目录只有写成模板（含 {date}）时才随日期重算
        cfg._photo_root_is_template = "{date}" in cfg.photo_root
    if known.model_weights:
        cfg.model_weights = str(known.model_weights)
    if known.chamber_template:
        cfg.chamber_template = str(known.chamber_template)
    if known.channels:
        cfg.channels = _parse_channels(str(known.channels))
    if known.valid_hours:
        cfg.valid_hours = str(known.valid_hours)

    cfg.__post_init__()
    return cfg


def ensure_parents(*paths: str) -> None:
    """批量创建输出目录的父目录（输入目录不创建，避免掩盖路径配错）。"""
    for p in paths:
        if p:
            Path(p).mkdir(parents=True, exist_ok=True)


def bootstrap_sys_path() -> None:
    """让子目录里的脚本能够 ``from config import load_config``。"""
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
