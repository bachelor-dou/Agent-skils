"""包入口:python -m hot_project → 启动 Web/API 服务。

CI 只装 requirements-ci.txt,所以「缺 fastapi」是常态而不是环境坏了,直接说装法。
"""

try:
    from .api_server import main
except ModuleNotFoundError as exc:
    if exc.name == "fastapi":
        raise SystemExit(
            "启动 Web/API 服务失败:缺少依赖 fastapi。\n"
            "在项目根目录执行:\n"
            "  pip install -r hot_project/requirements.txt\n"
        ) from exc
    raise

if __name__ == "__main__":
    main()
