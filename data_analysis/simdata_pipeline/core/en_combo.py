# -*- coding: utf-8 -*-
"""
selected__groups 组合生长曲线图（英文版重绘）
=============================================
selected_N 页签里嵌着一张「全部腔室生长曲线组合图」（2 行网格，
18 腔室 = 2×9 / 12 腔室 = 2×6，整图 6848×1144 或 4562×1144 像素）。
原图是中文版管线遗留：线性 Nrel 尺度、标题写着旧条件与原始腔室名
（如 L30-N20-IC0.5%_CB1 Cell growth curve fitting）。

本模块按约定格式重绘并原位替换：
- 子图标题 Growth curve (CB-<k>, <cond>)（不保留原始腔室名称）
- Y = ln(Nrel) 尺度（与英文版其余图表一致），曲线用已存参数解析生成
- Times New Roman（含 stix 数学字体），像素尺寸与原图完全一致，
  直接替换 xlsx 包内 xl/media/image1.png，锚点/版式不动

腔室顺序：按 selected_N 页行序，经 src_to_index（原始腔室名 → CB 序号）
定位；无数据来源时退化为行序 = 腔室序（并用 μmax 逐位校验）。
"""
import io
import math
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from core import en_plotting
from core.plotting import prepare, clean_label

# selected_N 页表头定位（英文表头，数据列序与中文版一致）
HEADER_MU = "μmax"
HEADER_CHAMBER = "chamber"


# --------------------------------------------------------------------- 读 selected 行序
def read_selected_order(selected_path):
    """
    selected_N 页 → [(原始腔室名, μmax), ...]（按行序）。
    表头为英文（Chamber / Maximum specific growth rate, μmax (h-1)）。
    """
    import openpyxl

    wb = openpyxl.load_workbook(selected_path, read_only=True, data_only=True)
    try:
        name = next((n for n in wb.sheetnames if n.startswith("selected_")), None)
        if name is None:
            raise ValueError(f"{Path(selected_path).name} 里没有 selected_ 页签")
        ws = wb[name]
        header = [str(v) if v is not None else "" for v in
                  next(ws.iter_rows(min_row=1, max_row=1, values_only=True))]
        col_ch = next((i for i, h in enumerate(header)
                       if HEADER_CHAMBER in h.lower()), None)
        col_mu = next((i for i, h in enumerate(header)
                       if HEADER_MU in h), None)
        if col_ch is None:
            raise ValueError(f"{Path(selected_path).name} 表头里找不到 Chamber 列")
        rows = []
        for row in ws.iter_rows(min_row=2, values_only=True):
            chamber = row[col_ch] if len(row) > col_ch else None
            if not isinstance(chamber, str) or not chamber.strip():
                continue
            mu = row[col_mu] if (col_mu is not None and len(row) > col_mu) else None
            rows.append((chamber.strip(), mu))
        return name, rows
    finally:
        wb.close()


def order_chambers(chambers, sel_rows, src_to_index, logger=None, tag=""):
    """
    按 selected 行序重排腔室。返回 (ordered_chambers, notes)。
    优先原始腔室名映射（{名: [序号...]}，重复腔室按出现顺序消费）；
    缺失时退化为位置序并做 μmax 校验。
    """
    from core.en_pptx import _as_queue

    by_index = {ch["index"]: ch for ch in chambers}
    src_queue = _as_queue(src_to_index)
    ordered, notes = [], []
    fallback = not src_queue
    for pos, (chamber_name, mu) in enumerate(sel_rows, 1):
        index = None
        if not fallback:
            queue = src_queue.get(chamber_name)
            index = queue.popleft() if queue else None
        if index is None:
            index = pos
            if not fallback:
                notes.append(f"第 {pos} 行 {chamber_name} 未映射，按位置取 CB-{pos}")
        ch = by_index.get(index)
        if ch is None:
            raise ValueError(f"selected 第 {pos} 行对应腔室序号 {index} 不存在")
        # μmax 校验（selected 表值 vs 工作簿参数，容差 1e-3）
        mu_wb = (ch.get("params") or {}).get("mu")
        if mu is not None and mu_wb is not None \
                and abs(float(mu) - float(mu_wb)) > 1e-3:
            notes.append(
                f"第 {pos} 行 μmax 不一致：selected={mu} vs CB-{index}={mu_wb}")
        ordered.append(ch)
    if logger:
        for note in notes:
            logger.warning("%s 组合图顺序：%s", tag, note)
    return ordered, notes


# --------------------------------------------------------------------- 重绘组合图
def combo_figure_en(ordered_chambers, cond_label, size_px):
    """
    2 行网格组合图，像素尺寸与 size_px=(w, h) 完全一致。
    子图：ln(实测) 散点 + 拟合曲线（Y 尺度）+ 小参数框，
    标题 Growth curve (CB-<k>, <cond>)。
    """
    import matplotlib.pyplot as plt

    prepare()
    cond = clean_label(cond_label)
    n = len(ordered_chambers)
    cols = math.ceil(n / 2)
    width, height = size_px
    dpi = 100
    fig, axes = plt.subplots(2, cols, figsize=(width / dpi, height / dpi),
                             dpi=dpi, squeeze=False)
    for pos, ch in enumerate(ordered_chambers):
        ax = axes[pos // cols][pos % cols]
        counts = np.asarray([v for v in ch["counts"] if v is not None],
                            dtype=float)
        t = np.arange(counts.size, dtype=float)
        p = ch["params"]
        ax.scatter(t, en_plotting._ln(counts), color="blue", alpha=0.6, s=8,
                   label="Actual data")
        if all(p.get(k) is not None for k in ("mu", "K", "lam")):
            ax.plot(t, en_plotting._y_curve(t, p["mu"], p["K"], p["lam"]),
                    color="red", linewidth=1.5, label="Fitted curve")
            if p.get("r2") is not None and p.get("N0") is not None:
                ax.text(0.04, 0.96,
                        en_plotting._param_text(p["mu"], p["K"], p["N0"],
                                                p["lam"], p["r2"]),
                        transform=ax.transAxes, va="top", fontsize=8.5,
                        bbox=dict(boxstyle="round", facecolor="white",
                                  alpha=0.8, pad=0.3))
        ax.set_title(f"Growth curve ({ch['cb_id']}, {cond})", fontsize=13)
        ax.set_xlabel(en_plotting.X_LABEL, fontsize=9)
        ax.set_ylabel(en_plotting.Y_LABEL_GROWTH, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
    # 腔室数为奇数时关掉最后一个空子图
    for pos in range(n, 2 * cols):
        axes[pos // cols][pos % cols].axis("off")
    fig.subplots_adjust(left=0.035, right=0.995, top=0.94, bottom=0.075,
                        wspace=0.28, hspace=0.55)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)     # 不用 tight：保证像素尺寸精确
    plt.close(fig)
    buf.seek(0)
    return buf


# --------------------------------------------------------------------- 原位替换图片
def replace_media_png(xlsx_path, png_bytes, expect_size=None, logger=None,
                      tag=""):
    """
    把 xlsx 包内唯一的媒体图片替换为 png_bytes（其余条目逐字节保留）。
    expect_size=(w, h)：原图像素尺寸，新图不一致时告警（锚点会拉伸）。
    返回替换的媒体条目名。
    """
    from PIL import Image

    xlsx_path = Path(xlsx_path)
    with zipfile.ZipFile(xlsx_path) as zin:
        entries = [(info, zin.read(info.filename)) for info in zin.infolist()]
    media = [(info, data) for info, data in entries
             if info.filename.startswith("xl/media/")]
    if len(media) != 1:
        raise ValueError(f"{xlsx_path.name} 里有 {len(media)} 个媒体文件，"
                         f"预期 1 个，未替换")
    media_name, old_data = media[0][0].filename, media[0][1]
    old_size = Image.open(io.BytesIO(old_data)).size
    new_size = Image.open(io.BytesIO(png_bytes)).size
    if expect_size and new_size != expect_size:
        raise ValueError(f"新图尺寸 {new_size} 与预期 {expect_size} 不符")
    if logger and new_size != old_size:
        logger.warning("%s 新图尺寸 %s 与原图 %s 不一致，显示会被拉伸",
                       tag, new_size, old_size)

    fd, tmp_name = tempfile.mkstemp(suffix=".xlsx",
                                    dir=str(xlsx_path.parent))
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp_name, "w", zipfile.ZIP_DEFLATED) as zout:
            for info, data in entries:
                zout.writestr(info, png_bytes if info.filename == media_name
                              else data)
        os.replace(tmp_name, xlsx_path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise
    return media_name


def rebuild_combo(selected_path, chambers, cond, src_to_index,
                  logger=None, tag=""):
    """
    重绘 selected_N 页的组合生长曲线图并原位替换。
    返回 (子图数, 替换的媒体名)；无图片或校验失败时抛异常。
    """
    from PIL import Image

    sel_name, sel_rows = read_selected_order(selected_path)
    ordered, _ = order_chambers(chambers, sel_rows, src_to_index,
                                logger=logger, tag=tag)

    with zipfile.ZipFile(selected_path) as z:
        media = [n for n in z.namelist() if n.startswith("xl/media/")]
        if not media:
            raise ValueError(f"{Path(selected_path).name} 里没有嵌入图片")
        old_size = Image.open(io.BytesIO(z.read(media[0]))).size

    buf = combo_figure_en(ordered, cond["title_label"], old_size)
    name = replace_media_png(selected_path, buf.read(), expect_size=old_size,
                             logger=logger, tag=tag)
    if logger:
        logger.info("  %s %s 组合图已重绘：%d 子图 → %s（%dx%d）",
                    tag, sel_name, len(ordered), name, *old_size)
    return len(ordered), name
