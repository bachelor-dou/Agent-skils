import hot_projects.tools.basic.report as R


def test_report_fetches_readme_when_no_desc(tmp_path, monkeypatch):
    """无现成描述时，提供 token_mgr → 实时抓 README 并喂给 LLM（兜底，不凭空猜）。"""
    monkeypatch.setattr(R, "REPORT_DIR", str(tmp_path))
    captured = {}

    def fake_describe(name, repo_info, url, detail_level="detailed"):
        captured["repo_info"] = repo_info
        return "总结描述"

    monkeypatch.setattr(R, "call_llm_describe", fake_describe)
    monkeypatch.setattr(R, "fetch_repo_readme_excerpt", lambda tm, o, r, i: {"text": "README 真实内容"})
    monkeypatch.setattr(R, "fetch_repo_recent_commits", lambda tm, o, r, i, n: [])

    db = {"projects": {"a/b": {"gh_desc": ""}}}  # 无 desc、无简介
    path = R.step3_generate_report(
        [("a/b", {"growth": 100, "star": 1500})], db, token_mgr=object()
    )
    assert path  # 报告已生成
    assert captured["repo_info"].get("readme_excerpt") == "README 真实内容"
