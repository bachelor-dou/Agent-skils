"""
github-hot-projects 包入口
==========================
支持 python -m github-hot-projects 直接运行。
"""

import logging
import os
import sys

from .config import LOG_DIR
from .pipeline import main

# ── 日志配置 ──
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(LOG_DIR, "discover_hot_projects.log"),
            encoding="utf-8",
        ),
    ],
)

if __name__ == "__main__":
    main()
