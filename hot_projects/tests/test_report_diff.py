"""报告专用文件名规则 + 上期对比标注（蓝色「上新」/排名箭头）测试。"""

import os
from datetime import datetime, timezone

import hot_projects.tools.basic.report as R
import hot_projects.api_server as S


TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ── 文件名规则 ──

def _gen(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setattr(R, "REPORT_DIR", str(tmp_path))
    db = {"projects": {"a/b": {"desc": "现成描述", "desc_updated_at": TODAY,
                               "created_at": "2026-01-01T00:00:00Z", "star": 1500}}}
    path = R.step3_generate_report([("a/b", {"growth": 100, "star": 1500})], db, **kwargs)
    return os.path.basename(path)


def test_comprehensive_default_name(tmp_path, monkeypatch):
    assert _gen(tmp_path, monkeypatch, mode="comprehensive") == f"{TODAY}.md"


def test_comprehensive_custom_window_suffix(tmp_path, monkeypatch):
    assert _gen(tmp_path, monkeypatch, mode="comprehensive",
                growth_calc_days=10) == f"{TODAY}_10d.md"


def test_hot_new_name(tmp_path, monkeypatch):
    from hot_projects.config import DAYS_SINCE_CREATED
    # 默认新项目窗口 → 只有 _NEW，不加区间尾缀
    assert _gen(tmp_path, monkeypatch, mode="hot_new",
                days_since_created=DAYS_SINCE_CREATED) == f"{TODAY}_NEW.md"
    # 自定义窗口 → 尾部追加区间
    assert _gen(tmp_path, monkeypatch, mode="hot_new",
                days_since_created=30) == f"{TODAY}_NEW_30d.md"


def test_keyword_name(tmp_path, monkeypatch):
    # 关键词榜文件名带上方向，避免同日多方向覆盖
    assert _gen(tmp_path, monkeypatch, mode="keyword",
                topic="向量库") == f"{TODAY}_KEY_向量库.md"
    assert _gen(tmp_path, monkeypatch, mode="keyword",
                topic="OCR识别") == f"{TODAY}_KEY_OCR识别.md"


# ── 上期对比 ──

def _report_md(date: str, repos: list[str], title="GitHub 热门项目") -> str:
    lines = [f"# {title} — {date}", "", f"> 共 {len(repos)} 个项目 | 报告生成: {date}", ""]
    for i, repo in enumerate(repos, 1):
        lines += [
            f"## {i}. {repo}", "",
            f"链接: https://github.com/{repo}", "",
            "- 创建时间: 2026-01-01",
            "- 总 Star: 1,000",
            "- 近7天增长: +500", "",
            "### 项目定位与用途", "", "描述文本。", "",
            "---", "",
        ]
    return "\n".join(lines)


def _write_reports(tmp_path, monkeypatch, files: dict[str, str]):
    monkeypatch.setattr(S, "REPORT_DIR", str(tmp_path))
    S._prev_report_cache.clear()
    for name, content in files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")


def test_diff_marks_fresh_and_delta(tmp_path, monkeypatch):
    _write_reports(tmp_path, monkeypatch, {
        "2026-06-24.md": _report_md("2026-06-24", ["a/old", "b/moved"]),
        "2026-07-01.md": _report_md("2026-07-01", ["b/moved", "c/fresh"]),
    })
    html = S._render_report_html("2026-07-01.md", (tmp_path / "2026-07-01.md").read_text())

    assert "上新" in html                      # c/fresh 挂蓝色徽章
    assert 'data-fresh="1"' in html           # 供前端「仅上新」过滤
    assert "↑1" in html                       # b/moved 从第 2 → 第 1
    assert "较上期 2026-06-24" in html          # hero 对比统计条


def test_diff_skips_different_family(tmp_path, monkeypatch):
    # _NEW 榜不会拿综合榜作对比基准
    _write_reports(tmp_path, monkeypatch, {
        "2026-06-24.md": _report_md("2026-06-24", ["a/old"]),
        "2026-07-01_NEW.md": _report_md("2026-07-01", ["c/fresh"], title="GitHub 新项目热度榜"),
    })
    html = S._render_report_html("2026-07-01_NEW.md", (tmp_path / "2026-07-01_NEW.md").read_text())
    assert "上新" not in html
    assert "较上期" not in html


def test_diff_title_guard_blocks_legacy_keyword(tmp_path, monkeypatch):
    # 旧版关键词榜与综合榜共用无尾缀文件名 → 标题前缀不同时跳过对比
    _write_reports(tmp_path, monkeypatch, {
        "2026-06-24.md": _report_md("2026-06-24", ["a/old"], title="GitHub 热门项目｜方向：向量库"),
        "2026-07-01.md": _report_md("2026-07-01", ["c/fresh"]),
    })
    html = S._render_report_html("2026-07-01.md", (tmp_path / "2026-07-01.md").read_text())
    assert "较上期" not in html


def test_diff_keyword_same_topic_compares(tmp_path, monkeypatch):
    # 同方向关键词榜跨日可对比；不同方向互不干扰（文件名带 topic）
    kw_title = "GitHub 热门项目｜方向：向量库"
    _write_reports(tmp_path, monkeypatch, {
        "2026-06-24_KEY_向量库.md": _report_md("2026-06-24", ["a/old", "b/keep"], title=kw_title),
        "2026-07-01_KEY_向量库.md": _report_md("2026-07-01", ["b/keep", "c/fresh"], title=kw_title),
        "2026-07-01_KEY_OCR.md": _report_md("2026-07-01", ["x/other"], title="GitHub 热门项目｜方向：OCR"),
    })
    html = S._render_report_html(
        "2026-07-01_KEY_向量库.md", (tmp_path / "2026-07-01_KEY_向量库.md").read_text())
    assert "上新" in html                       # c/fresh 相对上一份同方向榜是新增
    assert "较上期 2026-06-24_KEY_向量库" in html


def test_diff_none_when_no_previous(tmp_path, monkeypatch):
    _write_reports(tmp_path, monkeypatch, {
        "2026-07-01.md": _report_md("2026-07-01", ["a/solo"]),
    })
    html = S._render_report_html("2026-07-01.md", (tmp_path / "2026-07-01.md").read_text())
    assert "上新" not in html
    assert "较上期" not in html
