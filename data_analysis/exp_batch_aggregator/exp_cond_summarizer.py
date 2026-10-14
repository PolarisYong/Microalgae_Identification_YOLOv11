#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按实验自变量设置（工况）汇总流道实验结果数据。

依据"本征动力实验工况标记.xls"中记录的 批次 -> 流道 -> 数据图标签格式(工况) 对应关系，
把各批次、各流道的实验结果文件（如 CH8_可视化结果_96.xlsx）按工况分组复制到同一个文件夹，
并在文件名前加上该实验批次的日期前缀以区分不同批次。

典型目录结构（输入）：
    <INPUT_FOLDER>/<YYYYMMDD>/数据汇总/03_可视化结果/<N>小时/CH<k>_可视化结果_<N>.xlsx

输出结构：
    <OUTPUT_FOLDER>/<工况标签>/<YYYYMMDD>_CH<k>_可视化结果_<N>.xlsx

使用方式（两种任选其一）：

1) 直接修改下面的"配置区"常量，然后运行：
       python exp_cond_summarizer.py

2) 用命令行参数覆盖配置区的值：
       python exp_cond_summarizer.py ^
           --xls    "F:/.../本征动力实验工况标记.xls" ^
           --input  "F:/.../0828_μmax_revision" ^
           --output "F:/.../实验结果汇总"

   # 只预览，不实际复制
       python exp_cond_summarizer.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# ===========================================================================
#                            配 置 区
#          （直接修改这里即可，无需改动下面的代码；命令行参数优先）
# ===========================================================================

# 1. 工况标记文件：记录 批次 -> 流道 -> 数据图标签格式 的对应关系
XLS_FILE = r"F:\Workbuddy_Workspace\Peipei_Workspace\本征动力实验工况标记.xls"
# 若工况标记表格不在第一个工作表，可在此指定工作表名或序号（0 表示第一个）
XLS_SHEET = None

# 2. 输入文件夹：包含各日期子文件夹（如 20260719）的实验数据根目录
INPUT_FOLDER = r"F:\Workbuddy_Workspace\Peipei_Workspace\μmax_revision"

# 3. 输出文件夹：将按工况标签建立子文件夹
OUTPUT_FOLDER = r"F:\Workbuddy_Workspace\Peipei_Workspace\实验分析结果"

# 4. 培养时长（小时）：用于拼接结果文件路径，对应"96小时"这类目录名
VALID_HOURS = "96"

# 5. 目标文件已存在时：True 覆盖，False 跳过
OVERWRITE_EXISTING = True

# 6. 只预览不实际复制：True 表示不写任何文件
DRY_RUN = False

# 7. 是否打印解析细节（批次位置、跳过的行等）
VERBOSE = False

# ===========================================================================
#                     以下为脚本逻辑，通常无需修改
# ===========================================================================

# ---------------------------------------------------------------------------
# 默认配置（可通过命令行参数覆盖）
# ---------------------------------------------------------------------------

#: 批次标题中用于提取实验日期的正则，匹配 "26-05-04" 这类 "YY-MM-DD"
DATE_PATTERN = re.compile(r"(\d{2}-\d{2}-\d{2})")

#: 批次标题必须包含的关键字
BATCH_KEYWORD = "批次实验"

#: 表头行的关键字 -> 逻辑列名。用于在表头行中自动定位各列的列号。
#: 这样 xls 的列顺序或左右布局发生变化时，脚本仍能正确解析。
HEADER_KEYWORDS = {
    "channel": ["流道编号", "流道"],
    "label": ["数据图标签格式", "标签格式", "数据图标签", "标签"],
    "condition": ["工况编号", "工况"],
    "group": ["样本分组", "分组"],
}

#: 批次的结束边界：遇到这些关键字的行，说明该批次表格已结束
BATCH_STOP_KEYWORDS = ["批次实验", "工况标记"]

#: 结果文件名的模板，{ch} 为流道编号（如 CH8），{hours} 为培养时长（如 96）
RESULT_FILENAME_TEMPLATE = "{ch}_可视化结果_{hours}.xlsx"

#: 结果文件所在目录模板，相对 <input>/<YYYYMMDD>/
RESULT_SUBDIR_TEMPLATE = os.path.join("数据汇总", "03_可视化结果", "{hours}小时")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Entry:
    """一条 批次-流道-工况 记录。"""

    date_short: str  # 形如 "26-05-04"
    channel: str  # 形如 "CH8"
    label: str  # 形如 "L120-N160-IC5.25%"
    batch_title: str = ""  # 所属批次标题原文
    row: int = -1  # 在 xls 中的行号（1-based，便于人工核对）

    @property
    def date_folder(self) -> str:
        """转换为数据文件夹日期，如 "26-05-04" -> "20260504"。"""
        return "20" + self.date_short.replace("-", "")

    @property
    def date_prefix(self) -> str:
        """文件名前缀，与数据文件夹日期一致。"""
        return self.date_folder


@dataclass
class Cell:
    """xls 中的一个单元格（用于定位批次表格）。"""

    row: int  # 0-based
    col: int  # 0-based
    text: str


# ---------------------------------------------------------------------------
# xls 解析
# ---------------------------------------------------------------------------


def _read_sheet(path: str, sheet: str | int | None):
    """
    读取 xls/xlsx 的一个工作表，返回 pandas DataFrame（无表头）。

    sheet 为 None 时不指定工作表，而是自动挑选第一个非空工作表；
    这样即使工况标记表格被放在 Sheet2、Sheet3 也能正确读入。
    """
    try:
        import pandas as pd
    except ImportError:
        sys.exit(
            "缺少依赖 pandas。请先安装：\n"
            "    pip install pandas openpyxl xlrd"
        )

    try:
        if sheet is None:
            sheets = pd.read_excel(path, sheet_name=None, header=None)
            if isinstance(sheets, dict):
                for name, frame in sheets.items():
                    if frame is not None and not frame.empty:
                        print(f"[解析] 自动选用工作表：{name}")
                        return frame
                sys.exit(f"工况标记文件 {path!r} 中没有任何非空工作表。")
            return sheets

        return pd.read_excel(path, sheet_name=sheet, header=None)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - 需要把底层错误透传给用户
        sys.exit(f"无法读取工况标记文件 {path!r}（sheet={sheet!r}）：{exc}")


def _cell_text(value) -> str:
    """把单元格值规范化为去除首尾空白的字符串。"""
    import pandas as pd

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    # 某些 xls 会把不间断空格/零宽字符带进来
    return text.replace("\u00a0", " ").replace("\u200b", "").strip()


def collect_cells(df) -> list[Cell]:
    """把 DataFrame 中所有非空单元格收集为一维列表。"""
    cells: list[Cell] = []
    for r in range(df.shape[0]):
        for c in range(df.shape[1]):
            text = _cell_text(df.iat[r, c])
            if text:
                cells.append(Cell(row=r, col=c, text=text))
    return cells


def find_batches(cells: Iterable[Cell]) -> list[Cell]:
    """找出所有批次标题单元格，按 (行, 列) 排序。"""
    batches = [
        cell
        for cell in cells
        if BATCH_KEYWORD in cell.text and DATE_PATTERN.search(cell.text)
    ]
    batches.sort(key=lambda cell: (cell.row, cell.col))
    return batches


def _match_header(text: str) -> str | None:
    """判断一个表头单元格对应哪个逻辑列。返回 None 表示不是表头。"""
    for logical_name, keywords in HEADER_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return logical_name
    return None


def _column_span(batch_row: int, batch_col: int, all_batches: Sequence[Cell]) -> int:
    """
    推断某批次表格占据的列区间右边界（不含）。

    同一行可能存在多个并排的批次表格（例如第1批次在左、第3批次在右），
    此时"右侧相邻批次标题的列号"即为本批次列区间的右边界；
    同一行没有其他批次时返回一个足够大的值，交由表头位置进一步约束。
    """
    for other in all_batches:
        if other.row == batch_row and other.col > batch_col:
            return other.col
    return 10**9


def _find_header_columns(
    cells: Sequence[Cell], batch: Cell, next_batch_row: int, span_end: int
) -> tuple[int, dict[str, int]] | None:
    """
    在批次标题之后、下一个批次之前，找到表头行和其中的列号。

    表头行必须同时包含"流道编号"和"数据图标签格式"两类列，才被认定为有效表头。
    列号必须落在本批次的列区间 [batch.col, span_end) 内，避免匹配到并排批次的表头。
    返回 (表头行号, {逻辑列名: 列号})；找不到则返回 None。
    """
    candidates: dict[int, dict[str, int]] = {}
    for cell in cells:
        if not (batch.row < cell.row < next_batch_row):
            continue
        if not (batch.col <= cell.col < span_end):
            continue
        logical = _match_header(cell.text)
        if logical is None:
            continue
        candidates.setdefault(cell.row, {})[logical] = cell.col

    # 取最靠近批次标题、且列信息齐全的那一行
    for row in sorted(candidates):
        cols = candidates[row]
        if "channel" in cols and "label" in cols:
            return row, cols
    return None


def parse_marking_file(
    xls_path: str, sheet: str | int | None = None, verbose: bool = False
) -> list[Entry]:
    """解析工况标记文件，返回所有 批次-流道-工况 记录。"""
    if not os.path.isfile(xls_path):
        sys.exit(f"工况标记文件不存在：{xls_path}")

    df = _read_sheet(xls_path, sheet)
    cells = collect_cells(df)
    batches = find_batches(cells)

    if not batches:
        sys.exit(
            f"在 {xls_path!r} 中未找到任何包含 {BATCH_KEYWORD!r} 的批次标题，"
            "请确认文件内容或使用 --sheet 指定工作表。"
        )

    if verbose:
        print(f"[解析] 共发现 {len(batches)} 个批次：")
        for cell in batches:
            date = DATE_PATTERN.search(cell.text)
            print(f"        第{cell.row + 1}行 第{cell.col + 1}列  {cell.text}")

    entries: list[Entry] = []
    for idx, batch in enumerate(batches):
        # 本批次的结束行：下一个"行号更大"的批次标题行
        next_row = df.shape[0]
        for later in batches[idx + 1:]:
            if later.row > batch.row:
                next_row = later.row
                break

        # 本批次的列区间：右边界取同一行右侧相邻批次的起始列
        span_end = _column_span(batch.row, batch.col, batches)
        header = _find_header_columns(cells, batch, next_row, span_end)
        if header is None:
            if verbose:
                print(f"[警告] 批次「{batch.text}」未找到表头行，已跳过。")
            continue

        header_row, cols = header
        date_short = DATE_PATTERN.search(batch.text).group(1)  # type: ignore[union-attr]
        ch_col = cols["channel"]
        label_col = cols["label"]

        # 数据行：表头行之后，直到本批次结束
        for r in range(header_row + 1, next_row):
            channel = _cell_text(df.iat[r, ch_col])
            label = _cell_text(df.iat[r, label_col])
            if not channel or not label:
                continue
            if not ch_col < span_end or not label_col < span_end:
                continue
            # 跳过混入的标题/表头行
            if BATCH_KEYWORD in channel:
                break
            if _match_header(channel):
                continue
            entries.append(
                Entry(
                    date_short=date_short,
                    channel=_normalize_channel(channel),
                    label=label,
                    batch_title=batch.text,
                    row=r + 1,
                )
            )

    return entries


def _normalize_channel(text: str) -> str:
    """规范化流道编号：去掉内部空格并转大写，如 "CH2 " -> "CH2"。"""
    return re.sub(r"\s+", "", text).upper()


# ---------------------------------------------------------------------------
# 汇总执行
# ---------------------------------------------------------------------------


def result_path(
    input_root: str, entry: Entry, hours: str, filename_template: str, subdir_template: str
) -> str:
    """拼出某个记录对应的结果文件绝对路径。"""
    subdir = subdir_template.format(hours=hours)
    filename = filename_template.format(ch=entry.channel, hours=hours)
    return os.path.join(input_root, entry.date_folder, subdir, filename)


def plan_operations(
    entries: Sequence[Entry],
    input_root: str,
    output_root: str,
    hours: str,
    filename_template: str = RESULT_FILENAME_TEMPLATE,
    subdir_template: str = RESULT_SUBDIR_TEMPLATE,
    verbose: bool = False,
) -> tuple[list[tuple[Entry, str, str]], list[Entry]]:
    """
    生成复制计划。

    返回 (可执行的操作列表, 源文件缺失的记录列表)，
    操作列表元素为 (记录, 源文件路径, 目标文件路径)。
    """
    operations: list[tuple[Entry, str, str]] = []
    missing: list[Entry] = []

    for entry in entries:
        src = result_path(input_root, entry, hours, filename_template, subdir_template)
        if not os.path.isfile(src):
            missing.append(entry)
            continue

        filename = filename_template.format(ch=entry.channel, hours=hours)
        dst_name = f"{entry.date_prefix}_{filename}"
        dst = os.path.join(output_root, entry.label, dst_name)
        operations.append((entry, src, dst))

    return operations, missing


def run_summary(
    xls_path: str,
    input_root: str,
    output_root: str,
    hours: str = "96",
    sheet: str | int | None = None,
    dry_run: bool = False,
    on_conflict: str = "overwrite",
    verbose: bool = False,
) -> int:
    """执行汇总。返回进程退出码（0 表示成功）。"""
    print("=" * 68)
    print("实验工况汇总")
    print("=" * 68)
    print(f"工况标记文件 : {xls_path}")
    print(f"输入文件夹   : {input_root}")
    print(f"输出文件夹   : {output_root}")
    print(f"培养时长     : {hours} 小时")
    print(f"模式         : {'预览（不复制）' if dry_run else '执行复制'}")
    print("-" * 68)

    if not os.path.isdir(input_root):
        sys.exit(f"输入文件夹不存在：{input_root}")

    entries = parse_marking_file(xls_path, sheet=sheet, verbose=verbose)
    print(f"解析得到 {len(entries)} 条 批次-流道 记录。\n")

    operations, missing = plan_operations(
        entries, input_root, output_root, hours, verbose=verbose
    )

    # 按工况分组统计
    label_map: dict[str, list[tuple[Entry, str, str]]] = defaultdict(list)
    for entry, src, dst in operations:
        label_map[entry.label].append((entry, src, dst))

    if not dry_run:
        os.makedirs(output_root, exist_ok=True)

    copied = 0
    skipped = 0
    conflicts: list[str] = []

    for label in sorted(label_map):
        items = label_map[label]
        print(f"[{label}] {len(items)} 个文件")
        if not dry_run:
            os.makedirs(os.path.join(output_root, label), exist_ok=True)

        seen_names: set[str] = set()
        for entry, src, dst in items:
            name = os.path.basename(dst)
            if name in seen_names:
                conflicts.append(dst)
            seen_names.add(name)

            if dry_run:
                print(f"    (预览) {name}")
                continue

            if os.path.exists(dst) and on_conflict == "skip":
                print(f"    跳过（已存在） {name}")
                skipped += 1
                continue

            shutil.copy2(src, dst)
            print(f"    + {name}")
            copied += 1
        print()

    print("-" * 68)
    if dry_run:
        print(f"预览完成：计划复制 {len(operations)} 个文件到 {len(label_map)} 个工况文件夹。")
    else:
        print(
            f"完成：复制 {copied} 个文件，跳过 {skipped} 个，"
            f"共 {len(label_map)} 个工况文件夹。"
        )
        print(f"输出位置：{output_root}")

    if conflicts:
        print(f"\n[警告] 有 {len(conflicts)} 个目标文件名重复，后者覆盖前者：")
        for path in conflicts:
            print(f"        {path}")

    if missing:
        print(f"\n[警告] 有 {len(missing)} 条记录未找到源文件，已跳过：")
        for entry in missing:
            src = result_path(input_root, entry, hours, RESULT_FILENAME_TEMPLATE, RESULT_SUBDIR_TEMPLATE)
            print(f"        {entry.batch_title} | {entry.channel} -> {src}")
    else:
        print("\n所有记录的源文件均已找到。")

    return 0


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "按实验自变量设置（工况）汇总各批次流道的实验结果数据。\n"
            "所有参数均可省略；省略时使用脚本顶部「配置区」中设置的默认值。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  # 直接使用配置区的路径\n"
            "  python exp_cond_summarizer.py\n"
            "\n"
            "  # 用命令行覆盖配置区的路径\n"
            "  python exp_cond_summarizer.py --xls 工况标记.xls --input ./0828_μmax_revision --output ./实验结果汇总\n"
            "\n"
            "  # 只预览，不实际复制\n"
            "  python exp_cond_summarizer.py --dry-run --verbose\n"
        ),
    )
    parser.add_argument(
        "--xls",
        default=None,
        help=f"工况标记文件路径（xls/xlsx）。默认取配置区 XLS_FILE：{XLS_FILE}",
    )
    parser.add_argument(
        "--input",
        default=None,
        help=f"输入文件夹，即包含各日期子文件夹（如 20260719）的实验数据根目录。"
        f"默认取配置区 INPUT_FOLDER：{INPUT_FOLDER}",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"输出文件夹，将按工况标签建立子文件夹。"
        f"默认取配置区 OUTPUT_FOLDER：{OUTPUT_FOLDER}",
    )
    parser.add_argument(
        "--hours",
        default=None,
        help=f"培养时长，用于拼接结果文件路径。默认取配置区 VALID_HOURS：{VALID_HOURS}（即 {VALID_HOURS}小时）",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="工况标记文件中的工作表名或索引，默认自动取第一个非空表",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览将执行的操作，不实际复制文件",
    )
    parser.add_argument(
        "--on-conflict",
        choices=["overwrite", "skip"],
        default=None,
        help="目标文件已存在时的处理方式：overwrite（覆盖）或 skip（跳过）。"
        "默认跟随配置区 OVERWRITE_EXISTING",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出解析细节（批次位置、跳过的行等）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # 命令行未传的项，回落到脚本顶部配置区的值
    xls_path = args.xls or XLS_FILE
    input_root = args.input or INPUT_FOLDER
    output_root = args.output or OUTPUT_FOLDER
    hours = str(args.hours) if args.hours is not None else str(VALID_HOURS)
    sheet = args.sheet if args.sheet is not None else XLS_SHEET
    if args.on_conflict is not None:
        on_conflict = args.on_conflict
    else:
        on_conflict = "overwrite" if OVERWRITE_EXISTING else "skip"
    dry_run = args.dry_run or DRY_RUN
    verbose = args.verbose or VERBOSE

    return run_summary(
        xls_path=xls_path,
        input_root=input_root,
        output_root=output_root,
        hours=hours,
        sheet=sheet,
        dry_run=dry_run,
        on_conflict=on_conflict,
        verbose=verbose,
    )


if __name__ == "__main__":
    raise SystemExit(main())
