"""star_trend 从历史报告推导 star 轨迹的测试。"""

import hot_projects.tools.tool.star_trend as S


def _report_md(date: str, repo: str, star: str, rank: int = 1) -> str:
    return "\n".join([
        f"# GitHub 热门项目 — {date}", "",
        f"> 共 1 个项目 | 报告生成: {date}", "",
        f"## {rank}. {repo}", "",
        f"链接: https://github.com/{repo}", "",
        "- 创建时间: 2026-01-01",
        "- 主语言: Python",
        f"- 总 Star: {star}",
        "- 近7天增长: +500", "",
        "### 项目定位与用途", "", "描述。", "",
    ])


def _prep(tmp_path, monkeypatch, files: dict[str, str]):
    monkeypatch.setattr(S, "REPORT_DIR", str(tmp_path))
    for name, content in files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")


def test_trend_builds_ascending_series(tmp_path, monkeypatch):
    _prep(tmp_path, monkeypatch, {
        "2026-07-01.md": _report_md("2026-07-01", "a/b", "1,000", rank=5),
        "2026-07-08.md": _report_md("2026-07-08", "a/b", "3,500", rank=2),
        "2026-07-15.md": _report_md("2026-07-15", "a/b", "9,000", rank=1),
    })
    out = S.star_trend("a/b")
    assert out["points"] == 3
    assert [p["star"] for p in out["series"]] == [1000, 3500, 9000]
    assert [p["date"] for p in out["series"]] == ["2026-07-01", "2026-07-08", "2026-07-15"]
    assert out["star_change"] == 8000
    assert out["series"][-1]["rank"] == 1


def test_trend_gap_when_unranked(tmp_path, monkeypatch):
    # 中间一周该项目没上榜（报告里换成别的项目）→ 该周缺点
    _prep(tmp_path, monkeypatch, {
        "2026-07-01.md": _report_md("2026-07-01", "a/b", "1,000"),
        "2026-07-08.md": _report_md("2026-07-08", "x/y", "2,000"),
        "2026-07-15.md": _report_md("2026-07-15", "a/b", "5,000"),
    })
    out = S.star_trend("a/b")
    assert out["points"] == 2
    assert [p["date"] for p in out["series"]] == ["2026-07-01", "2026-07-15"]


def test_trend_not_found(tmp_path, monkeypatch):
    _prep(tmp_path, monkeypatch, {"2026-07-15.md": _report_md("2026-07-15", "a/b", "1,000")})
    out = S.star_trend("no/such")
    assert out["points"] == 0
    assert "message" in out


def test_registry_has_star_trend():
    from hot_projects.tools.registry import build_default_registry
    names = {s["function"]["name"] for s in build_default_registry().schemas()}
    assert "star_trend" in names
