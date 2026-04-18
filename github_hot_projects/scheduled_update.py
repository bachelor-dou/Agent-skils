#!/usr/bin/env python
"""
定时更新脚本：后台自动执行综合热度搜索、增长计算、排名并生成报告。

无需 LLM 对话，直接调用 Tool 函数完成完整流程：
  1. search_hot_projects — 全类别关键词搜索
  2. scan_star_range     — star 范围补充扫描
  3. fetch_trending      — GitHub Trending 三档补源
  4. batch_check_growth  — 批量计算增长并筛选候选
  5. rank_candidates     — 综合评分排序
  6. generate_report     — 生成 Markdown 报告

用法：
  python scheduled_update.py [--top-n 100] [--mode comprehensive]
"""
# ============================================================
# 部署为定时任务（cron）
# ============================================================
#
# 1. 手动运行测试：
#    export LLM_API_KEY="sk-xxx"
#    cd /home/openeuler/Agent-skils/github_hot_projects
#    python scheduled_update.py --top-n 100 --mode comprehensive
#
# 2. 编辑 crontab（每天凌晨 2 点自动执行）：
#    crontab -e
#    添加以下行：
#    0 2 * * * export LLM_API_KEY="sk-xxx" && cd /home/openeuler/Agent-skils/github_hot_projects && /usr/bin/python3 scheduled_update.py >> logs/scheduled.log 2>&1
#
# 3. 或使用 systemd timer：
#    sudo cp scheduled_update.service /etc/systemd/system/
#    sudo cp scheduled_update.timer   /etc/systemd/system/
#    sudo systemctl enable --now scheduled_update.timer
#
# 日志：logs/scheduled-YYYY-MM-DD.log
# ============================================================
import argparse
import logging
import logging.handlers
import os
import sys
from datetime import datetime

# 确保可以 import 项目包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from github_hot_projects.common.config import (
    HOT_PROJECT_COUNT,
    LOG_DIR,
    STAR_GROWTH_THRESHOLD,
)
from github_hot_projects.common.db import load_db
from github_hot_projects.common.token_manager import TokenManager
from github_hot_projects.execution.pipeline import DiscoveryPipeline


def setup_logging() -> str:
    """配置日志：同时输出到终端和文件，文件使用 RotatingFileHandler 防止过大。"""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(
        LOG_DIR,
        f"scheduled-{datetime.now().strftime('%Y-%m-%d')}.log",
    )
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=50 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            file_handler,
            logging.StreamHandler(),
        ],
        force=True,
    )
    return log_path


logger = logging.getLogger("scheduled_update")


def run_update(top_n: int, mode: str) -> None:
    """执行完整的搜索→增长→排名→报告流程（委托给 DiscoveryPipeline）。"""
    token_mgr = TokenManager()
    db = load_db()

    logger.info(
        f"开始定时更新: mode={mode}, top_n={top_n}, "
        f"DB projects={len(db.get('projects', {}))}, valid={db.get('valid')}"
    )

    pipeline = DiscoveryPipeline(token_mgr, db)
    result = pipeline.run(mode=mode, top_n=top_n)

    report_path = result.get("report_path", "")
    if report_path:
        logger.info(f"定时更新完成! 报告: {report_path}")
    elif result.get("error"):
        logger.error(f"定时更新失败: {result['error']}")
    else:
        logger.error("报告生成失败。")


def main():
    parser = argparse.ArgumentParser(
        description="定时更新：自动搜索、计算增长、排名并生成 GitHub 热门项目报告"
    )
    parser.add_argument(
        "--top-n", type=int, default=HOT_PROJECT_COUNT,
        help=f"取前 N 个项目 (默认: {HOT_PROJECT_COUNT})",
    )
    parser.add_argument(
        "--mode", default="comprehensive",
        choices=["comprehensive", "hot_new"],
        help="排名模式 (默认: comprehensive)",
    )
    args = parser.parse_args()

    log_path = setup_logging()
    logger.info(f"日志: {log_path}")

    try:
        run_update(args.top_n, args.mode)
    except Exception:
        logger.exception("定时更新异常终止")
        sys.exit(1)


if __name__ == "__main__":
    main()


