"""日志落盘 —— 按月归档,一个入口脚本一个前缀。

    logs/2026-07/snapshot-2026-07-30.log
    logs/2026-07/weekly-2026-07-30.log

三个入口(每日快照、周报、api_server)本来各写一份几乎一样的 `basicConfig`,其中一份
忘了压 httpx 的 INFO,于是全量近千个批次每个打一行,业务日志被淹掉。

这里不知道项目在干什么,只知道「按月分目录、同时写文件和控制台、压掉几个吵闹的库」。
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

# 这几个库在 INFO 级别逐请求打日志。全量采集近千个批次,不压的话业务日志一行都看不见。
NOISY = ("httpx", "httpcore", "urllib3", "asyncio")


def setup(directory: Path, prefix: str, *, day: date | None = None,
          level: int = logging.INFO, console: bool = True) -> Path:
    """配好日志,返回日志文件路径。"""
    day = day or datetime.now().date()
    month_dir = directory / f"{day:%Y-%m}"
    month_dir.mkdir(parents=True, exist_ok=True)
    path = month_dir / f"{prefix}-{day}.log"

    handlers: list[logging.Handler] = [logging.FileHandler(path, encoding="utf-8")]
    if console:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,     # 入口脚本可能在 import 时已被别处配过,以这里为准
    )
    for name in NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
    return path
