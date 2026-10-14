# -*- coding: utf-8 -*-
"""
日志配置
========
同时输出到控制台和日志文件。任务要求：遇到缺失文件 / 腔室数量异常时
输出明确提示而非中断，因此业务代码统一用 WARNING 记录异常并继续。
"""
import io
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(log_dir=None, name="build_20groups"):
    """创建 logger；日志文件写到 <log_dir>/build_20groups_<时间戳>.log"""
    import logging

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(log_dir / f"{name}_{ts}.log",
                                 encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
