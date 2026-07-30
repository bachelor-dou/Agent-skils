"""fetch_trending 基础能力：守护函数内懒加载 import 的相对路径不再写错（回归 web.log 的 bug）。

core.fetch_trending 里对 `...datasource.github.trending` 仍是函数内导入——工具函数自己就叫
fetch_trending，顶层导入会和它同名。这类写错相对路径的 bug 只在调用时才炸，所以需要本测试守。
batch_condense_descriptions 则是顶层导入（写错路径会在 import 阶段立刻炸，不需要守），
因此要 patch 在 core 自己的绑定上，而不是 infra.llm 模块属性上。
"""

import hot_projects.datasource.github.trending as T
import hot_projects.tools.basic.core as C
from hot_projects.tools.basic import fetch_trending


def test_weekly_import_paths_resolve(monkeypatch):
    # 避免联网：桩掉真实抓取与 LLM 浓缩
    monkeypatch.setattr(T, "fetch_trending", lambda since="weekly": [
        {"full_name": "a/b", "star": 100, "forks": 5, "stars_today": 20,
         "language": "Python", "description": "x"},
    ])
    monkeypatch.setattr(C, "batch_condense_descriptions", lambda repos, max_chars=70: ["简介"])

    out = fetch_trending("weekly")
    assert out["trending_range"] == "weekly"
    assert out["repos"][0]["full_name"] == "a/b"
    assert out["repos"][0]["description"] == "简介"
