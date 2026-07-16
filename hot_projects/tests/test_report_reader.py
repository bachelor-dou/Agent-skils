"""analyze_report 报告读取工具测试。"""

import hot_projects.tools.tool.analyze_report as R


def _report_md() -> str:
    return "\n".join([
        "# GitHub 热门项目 — 2026-07-08", "",
        "> 共 2 个项目 | 增长阈值: >=1000 stars", "",
        "## 1. alibaba/page-agent", "",
        "链接: https://github.com/alibaba/page-agent", "",
        "- 创建时间: 2026-01-01",
        "- 主语言: Python",
        "- 总 Star: 25,072",
        "- 近7天增长: +4,218",
        "- 主题标签: agent, ai", "",
        "### 项目定位与用途", "",
        "这是一个页面代理项目。", "",
        "### 解决的问题", "",
        "解决自动化操作网页的问题。", "",
        "---", "",
        "## 2. facebook/astryx", "",
        "链接: https://github.com/facebook/astryx", "",
        "- 创建时间: 2026-05-01",
        "- 主语言: TypeScript",
        "- 总 Star: 7,019",
        "- 近7天增长: +5,014", "",
        "### 项目定位与用途", "",
        "一个前端框架。", "",
    ])


def _prep(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "REPORT_DIR", str(tmp_path))
    (tmp_path / "2026-07-08.md").write_text(_report_md(), encoding="utf-8")


def test_list_when_no_name(tmp_path, monkeypatch):
    _prep(tmp_path, monkeypatch)
    out = R.analyze_report()
    assert "reports" in out
    assert out["reports"][0]["name"] == "2026-07-08.md"


def test_no_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "REPORT_DIR", str(tmp_path))
    assert "error" in R.analyze_report(name="x")


def test_overview_table(tmp_path, monkeypatch):
    _prep(tmp_path, monkeypatch)
    out = R.analyze_report(name="2026-07-08.md")
    assert out["project_count"] == 2
    assert "alibaba/page-agent" in out["projects_table"]
    assert "facebook/astryx" in out["projects_table"]
    # 分段内容不应出现在整体清单里（控制 token）
    assert "页面代理项目" not in out["projects_table"]


def test_latest_alias_and_no_suffix(tmp_path, monkeypatch):
    _prep(tmp_path, monkeypatch)
    assert R.analyze_report(name="最新")["project_count"] == 2
    assert R.analyze_report(name="2026-07-08")["project_count"] == 2  # 自动补 .md


def test_repo_detail(tmp_path, monkeypatch):
    _prep(tmp_path, monkeypatch)
    out = R.analyze_report(name="2026-07-08.md", repo="alibaba/page-agent")
    assert out["repo"] == "alibaba/page-agent"
    assert out["rank"] == 1
    titles = [s["title"] for s in out["sections"]]
    assert "项目定位与用途" in titles


def test_repo_not_in_report(tmp_path, monkeypatch):
    _prep(tmp_path, monkeypatch)
    out = R.analyze_report(name="2026-07-08.md", repo="no/such")
    assert "error" in out
    assert "alibaba/page-agent" in out["repos_in_report"]


def test_path_traversal_blocked(tmp_path, monkeypatch):
    _prep(tmp_path, monkeypatch)
    assert "error" in R.analyze_report(name="../secret.md")


def test_registry_has_analyze_report():
    from hot_projects.tools.registry import build_default_registry
    names = {s["function"]["name"] for s in build_default_registry().schemas()}
    assert "analyze_report" in names
