"""
包入口：python -m github_hot_projects
=======================================
默认启动 Web/API 服务，实际启动逻辑统一复用 api_server.main()。
"""

from .api_server import main


if __name__ == "__main__":
    main()
