# -*- coding: utf-8 -*-
"""
读取 batch_grouping 已选好的腔室分组
====================================
每个工况目录下形如 `18_batch_<时间戳>/` 的文件夹里放着 batch_grouping 的产物，
其中 `selected__groups_<i>_<n>.xlsx` 的第 i 组含 n 个腔室：

    表头：源文件名称(source_file) | 微流道编号(flow) | 腔室名称(chamber) |
          最大比生长速率μmax (h^-1) | 环境容纳量K | ... 

「腔室数据不是之前选取的吗」——是的，选取结果就在这里，本模块只做读取，不重新筛选。

组与 CCF 表「原始数据来源」的 resN 对应关系：
    某工况的 Response N μmax 均值 == batch_grouping 第 N 组的 μmax 均值
    （已用 n=12 / n=17 等样本量核对一致）
因此定位分组时**优先按 μmax 均值匹配**，匹配不上再退回 resN → 第 N 组。
"""
import re
from pathlib import Path

import config


def find_batch_dir(cond_dir):
    """在工况目录下找 batch_grouping 的输出目录（18_batch_*），取最新一个。"""
    cond_dir = Path(cond_dir)
    if not cond_dir.exists():
        return None
    cands = sorted([d for d in cond_dir.iterdir()
                    if d.is_dir() and d.name.startswith(config.BATCH_DIR_PREFIX)])
    return cands[-1] if cands else None


def load_groups(cond_dir, logger=None):
    """
    读取某工况下所有已选分组。
    返回 list[dict]：{index, size, mean_mu, members}
        members: [(source_file, chamber_name, mu_max), ...]
    """
    import openpyxl

    cond_dir = Path(cond_dir)
    batch_dir = find_batch_dir(cond_dir)
    if batch_dir is None:
        if logger:
            logger.warning("工况 %s 下未找到 %s* 目录，无已选分组",
                           cond_dir.name, config.BATCH_DIR_PREFIX)
        return []

    groups = []
    for f in sorted(batch_dir.glob("selected__groups_*.xlsx")):
        m = re.search(r"selected__groups_(\d+)_(\d+)\.xlsx$", f.name)
        if not m:
            continue
        idx, size = int(m.group(1)), int(m.group(2))

        wb = openpyxl.load_workbook(f, data_only=True)
        sn = next((s for s in wb.sheetnames if s.startswith("selected_")), None)
        if sn is None:
            if logger:
                logger.warning("%s 内缺少 selected_* 页签，已跳过", f.name)
            continue
        ws = wb[sn]
        rows = list(ws.iter_rows(values_only=True))
        header = [str(h).strip() if h else "" for h in rows[0]]
        try:
            i_file = header.index("源文件名称(source_file)")
            i_ch = header.index("腔室名称(chamber)")
            i_mu = header.index("最大比生长速率μmax (h^-1)")
        except ValueError:
            if logger:
                logger.warning("%s 表头缺少必要列（源文件/腔室/μmax），已跳过", f.name)
            continue

        members = []
        for r in rows[1:]:
            if not r or r[i_file] is None:
                continue
            mu = r[i_mu]
            members.append((str(r[i_file]).strip(),
                            str(r[i_ch]).strip(),
                            float(mu) if isinstance(mu, (int, float)) else None))

        mus = [m[2] for m in members if m[2] is not None]
        mean_mu = sum(mus) / len(mus) if mus else None
        groups.append({"index": idx, "size": size, "mean_mu": mean_mu,
                       "members": members, "file": f.name})

    if logger and groups:
        logger.info("  工况 %s：读取到 %d 个已选分组 %s",
                    cond_dir.name, len(groups),
                    [(g["index"], len(g["members"]),
                      round(g["mean_mu"], 4) if g["mean_mu"] else None)
                     for g in groups])
    return groups


def read_members(group_file, logger=None, tag=""):
    """
    读取单个 selected__groups_*.xlsx 的已选腔室清单（保持原始顺序）。

    返回 [(source_file, chamber, mu_max), ...]；读不出来返回 []。
    与 load_groups 的区别：后者按工况目录批量读，这里只读指定文件，
    供「未使用数据」等需要按单个分组文件处理的场景使用。
    """
    import openpyxl

    path = Path(group_file)
    wb = openpyxl.load_workbook(path, data_only=True)
    try:
        sn = next((s for s in wb.sheetnames if s.startswith("selected_")), None)
        if sn is None:
            if logger:
                logger.warning("%s %s 内缺少 selected_* 页签", tag, path.name)
            return []
        ws = wb[sn]
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            return []
        header = [str(h).strip() if h else "" for h in rows[0]]
        try:
            i_file = header.index("源文件名称(source_file)")
            i_ch = header.index("腔室名称(chamber)")
            i_mu = header.index("最大比生长速率μmax (h^-1)")
        except ValueError:
            if logger:
                logger.warning("%s %s 表头缺少必要列（源文件/腔室/μmax）", tag, path.name)
            return []
        members = []
        for r in rows[1:]:
            if not r or r[i_file] is None:
                continue
            mu = r[i_mu]
            members.append((str(r[i_file]).strip(),
                            str(r[i_ch]).strip(),
                            float(mu) if isinstance(mu, (int, float)) else None))
        return members
    finally:
        wb.close()


def pick_group(groups, mu_target, res_index=None, logger=None, tag=""):
    """
    按 μmax 均值定位分组；失败则退回 resN → 第 N 组。
    返回 (group, match_kind)，定位失败返回 (None, None)。
    """
    if not groups:
        return None, None

    tol = 5e-4  # 原始表保留 4 位小数
    if mu_target is not None:
        hits = [g for g in groups
                if g["mean_mu"] is not None and abs(g["mean_mu"] - mu_target) <= tol]
        if len(hits) == 1:
            return hits[0], "μmax匹配"
        if len(hits) > 1:
            if logger:
                logger.warning("%s μmax=%.4f 命中多个分组 %s，改用 resN 定位",
                               tag, mu_target, [h["index"] for h in hits])

    if res_index is not None:
        for g in groups:
            if g["index"] == res_index:
                if logger:
                    logger.info("%s 按 res%d 定位到第 %d 组", tag, res_index, g["index"])
                return g, "resN"
        if logger:
            logger.warning("%s 请求 res%d，但该工况只有 %d 个分组，定位失败",
                           tag, res_index, len(groups))
    if logger:
        logger.warning("%s 无法确定分组（μmax=%.4f, res=%s）", tag, mu_target, res_index)
    return None, None
