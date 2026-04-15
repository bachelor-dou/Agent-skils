"""
Agent CLI 交互入口
==================
提供命令行交互界面，用户可通过终端对话与 Agent 交互。

用法：
  python -m github_hot_projects.agent_cli

交互示例：
  > 帮我找最近 AI Agent 方向的热门项目
  > 查一下 vllm-project/vllm 最近的 star 增长
  > 把增长阈值降到 300 再搜一次
  > 为排名第一的项目生成描述
  > 生成完整报告
  > quit
"""

import logging
import os
from datetime import datetime

try:
    import readline
except ImportError:  # pragma: no cover - Linux 通常可用，兜底给极简环境
    readline = None

from .common.config import LOG_DIR
from .agent import HotProjectAgent


def setup_logging() -> str:
    """配置日志：仅落盘，避免污染交互输入区域。"""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(
        LOG_DIR,
        f"agent-{datetime.now().strftime('%Y-%m-%d')}.log",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    return log_path


def main() -> None:
    """CLI 交互主循环。"""
    log_path = setup_logging()

    if readline is not None:
        readline.set_history_length(200)

    print("=" * 60)
    print("  GitHub 热门项目发现 Agent（ReAct 模式）")
    print("  输入自然语言指令，Agent 会自主规划并执行")
    print("  输入 quit / exit / q 退出")
    print(f"  日志文件: {log_path}")
    print("=" * 60)
    print()

    agent = HotProjectAgent()

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
