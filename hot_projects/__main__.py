"""包入口：python -m hot_projects → 启动 Web/API 服务。"""

try:
    from .api_server import main
except ModuleNotFoundError as exc:
    if exc.name == "fastapi":
        raise SystemExit(
            "启动 Web/API 服务失败：缺少依赖 fastapi。\n"
            "请先在项目根目录执行：\n"
            "  cd /root/code/Agent-skils\n"
            "  source .venv/bin/activate\n"
            "  pip install -r hot_projects/requirements.txt\n"
            "然后使用以下命令启动：\n"
            "  /root/code/Agent-skils/.venv/bin/python -m hot_projects"
        ) from exc
    raise

if __name__ == "__main__":
    main()
