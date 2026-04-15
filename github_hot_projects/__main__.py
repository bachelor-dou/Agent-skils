"""
包入口：python -m github_hot_projects
=======================================
等价于 uvicorn github_hot_projects.api_server:app --host 0.0.0.0 --port 8000
"""

from .api_server import app  # noqa: F401 — 确保模块可导入

if __name__ == "__main__":
    import uvicorn
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    uvicorn.run(
        "github_hot_projects.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
