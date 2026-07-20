"""fetch_trending 基础能力：守护函数内懒加载 import 的相对路径不再写错（回归 web.log 的 bug）。"""

import hot_projects.datasource.github.trending as T
import hot_projects.infra.llm as L
from hot_projects.tools.basic import fetch_trending


def test_weekly_import_paths_resolve(monkeypatch):
    # 避免联网：桩掉真实抓取与 LLM 浓缩；关键是函数内的 `...datasource` / `...infra` import 必须能解析
    monkeypatch.setattr(T, "fetch_trending", lambda since="weekly": [
        {"full_name": "a/b", "star": 100, "forks": 5, "stars_today": 20,
         "language": "Python", "description": "x"},
    ])
    monkeypatch.setattr(L, "batch_condense_descriptions", lambda repos, max_chars=70: ["简介"])

    out = fetch_trending("weekly")
    assert out["trending_range"] == "weekly"
    assert out["repos"][0]["full_name"] == "a/b"
    assert out["repos"][0]["description"] == "简介"
