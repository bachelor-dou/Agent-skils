"""
Agent CLI 交互入口
==================
    python -m hot_projects.agent_cli

交互示例：
  > 帮我找最近 AI Agent 方向的热门项目
  > 查一下 vllm-project/vllm 最近的 star 增长
  > 把增长阈值降到 300 再搜一次
  > quit
"""

import logging
import logging.handlers
import os
from datetime import datetime

try:
    import readline  # noqa: F401
except ImportError:  # pragma: no cover
    readline = None

from .config import LOG_DIR
from .agent import build_agent

logger = logging.getLogger("hot_projects")


def setup_logging() -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"agent-{datetime.now().strftime('%Y-%m-%d')}.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=50 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[file_handler],
        force=True,
    )
    return log_path


def main() -> None:
    log_path = setup_logging()
    if readline is not None:
        readline.set_history_length(200)

    print("=" * 60)
    print("  GitHub 热门项目发现 Agent（ReAct 模式）")
    print("  输入自然语言指令；quit / exit / q 退出")
    print(f"  日志文件: {log_path}")
    print("=" * 60)
    print()

    agent = build_agent()

    while True:
        try:
            user_input = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        print("Agent 思考中...\n")
        reply = agent.chat(user_input)
        print(f"Agent> {reply}\n")


if __name__ == "__main__":
    main()
