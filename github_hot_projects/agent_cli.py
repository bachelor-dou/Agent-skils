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
import sys

from .config import LOG_DIR
from .agent import HotProjectAgent


def setup_logging() -> None:
    """配置日志：文件 + 控制台。"""
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(LOG_DIR, "agent.log"), encoding="utf-8"
            ),
        ],
    )


def main() -> None:
    """CLI 交互主循环。"""
    setup_logging()

    print("=" * 60)
    print("  GitHub 热门项目发现 Agent（ReAct 模式）")
    print("  输入自然语言指令，Agent 会自主规划并执行")
    print("  输入 quit / exit / q 退出")
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
