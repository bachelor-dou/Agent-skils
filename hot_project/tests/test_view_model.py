"""视图模型:报告页的事实计算,不碰 HTML。

这些断言在拆出 view_model 之前只能写成「HTML 里有没有某个片段」,脆且读不出意图;
现在升降、上新、描述覆盖、附栏归属都是普通数据,直接比。
"""

import pytest

from hot_project.infra.data_access import reports
from hot_project.web import view_model


def _entry(rank: int, repo: str, metadata: dict | None = None,
           sections: list | None = None) -> reports.Entry:
    return reports.Entry(rank=rank, repo=repo, link=f"https://github.com/{repo}",
                         metadata=metadata or {"总 Star": "1,000"},
                         sections=sections or [])


@pytest.fixture
def no_db(monkeypatch):
    monkeypatch.setattr(view_model, "_desc_index", dict)


def test_rank_delta_and_fresh_are_read_off_the_previous_report(no_db):
    """升降是「上期名次 − 本期名次」,上期没有的是「上新」—— 两者互斥。"""
    report = reports.Report("榜", "", [_entry(1, "a/x"), _entry(2, "b/y"), _entry(3, "c/z")])
    diff = view_model.Diff("2026-07-01.md", {"a/x": 3, "b/y": 1}, added=1, removed=0)

    views = {v.repo: v for v in view_model.entry_views(report, diff)}

    assert (views["a/x"].delta, views["a/x"].is_fresh) == (2, False)     # 3 → 1,升 2
    assert (views["b/y"].delta, views["b/y"].is_fresh) == (-1, False)    # 1 → 2,降 1
    assert (views["c/z"].delta, views["c/z"].is_fresh) == (0, True)


def test_without_a_previous_report_nothing_is_marked_fresh(no_db):
    """没得比就都不标:第一期报告满页「上新」等于没有信息。"""
    report = reports.Report("榜", "", [_entry(1, "a/x")])
    view = view_model.entry_views(report, None)[0]
    assert view.delta == 0 and view.is_fresh is False


def test_a_db_description_overrides_the_report_and_empty_sections_get_a_fallback(monkeypatch):
    """DB 里刷新过的介绍要盖过报告原文;空段落必须有兜底文案,不能渲染成空框。"""
    monkeypatch.setattr(view_model, "_desc_index",
                        lambda: {"a/x": "项目定位与用途:DB 里新的。"})
    report = reports.Report("榜", "", [_entry(
        1, "a/x", sections=[{"title": "项目定位与用途", "content": "报告里旧的。"},
                            {"title": "解决的问题", "content": ""}])])

    sections = view_model.entry_views(report, None)[0].sections

    assert sections[0]["paragraphs"] == ["DB 里新的。"]
    assert sections[1]["paragraphs"] == ["暂无补充信息，可进入仓库查看 README。"]


def test_a_trend_card_gets_the_same_db_override_as_a_ranked_card(monkeypatch):
    """附栏满卡和正文条目走同一份事实(`card_sections`):DB 里刷新过的介绍
    对两条渲染路径同权,不能正文是新的、附栏还念旧稿。"""
    monkeypatch.setattr(view_model, "_desc_index",
                        lambda: {"d/new": "项目定位与用途:DB 里新的。"})
    entry = _entry(1, "d/new", metadata={"总 Star": "500"},
                   sections=[{"title": "项目定位与用途", "content": "报告里旧的。"},
                             {"title": "解决的问题", "content": ""}])

    sections = view_model.card_sections(entry)

    assert sections[0]["paragraphs"] == ["DB 里新的。"]
    assert sections[1]["paragraphs"] == ["暂无补充信息，可进入仓库查看 README。"]


def test_trending_rows_keep_order_and_only_on_board_repos_get_the_badge(no_db):
    """附栏行序跟报告走(卡片和锚点行交错);TRENDING 角标只挂给已上榜的。"""
    on_board = _entry(1, "a/x")
    card = _entry(1, "d/new", metadata={"总 Star": "500"})
    hit = reports.Entry(2, "a/x", "", {}, [])          # 无 metadata = 「见正文 #N」行
    report = reports.Report("榜", "", [on_board], trending={"weekly": [card, hit]})

    trend = view_model.trending_views(report)[0]
    assert [(r.card, r.rank) for r in trend.rows] == [(True, None), (False, 1)]
    assert trend.listed == frozenset({"a/x"})
    assert (trend.total, trend.hits) == (2, 1)

    views = view_model.entry_views(report, None, [trend])
    assert views[0].trend_labels == [trend.label]
    assert "trending" in views[0].search_blob
