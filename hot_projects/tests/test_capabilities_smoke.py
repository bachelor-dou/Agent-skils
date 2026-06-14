def test_capabilities_import():
    from hot_projects.capabilities import (
        search_by_keywords, scan_star_range, fetch_trending, trending_repo_to_search_repo,
        check_repo_growth, batch_check_growth, rank_candidates,
        describe_project, get_db_info, generate_report,
    )
    assert callable(search_by_keywords)
    assert callable(rank_candidates)
    assert callable(generate_report)


def test_get_db_info_overview():
    from hot_projects.capabilities.describe import get_db_info
    db = {"valid": True, "date": "2026-06-10", "projects": {"a/b": {"star": 1}}}
    out = get_db_info(db=db, repo=None)
    assert out["total_projects"] == 1 and out["valid"] is True


def test_get_db_info_repo_found_and_missing():
    from hot_projects.capabilities.describe import get_db_info
    db = {"valid": True, "date": "2026-06-10", "projects": {"a/b": {"star": 9}}}
    assert get_db_info(db=db, repo="a/b")["found"] is True
    assert get_db_info(db=db, repo="x/y")["found"] is False


def test_rank_candidates_orders_by_score():
    from hot_projects.capabilities.rank import rank_candidates
    candidates = {
        "a/b": {"growth": 1000, "star": 2000, "created_at": ""},
        "c/d": {"growth": 200, "star": 1500, "created_at": ""},
    }
    out = rank_candidates(candidates, top_n=10, mode="comprehensive", db={"projects": {}})
    assert out["ranked_projects"][0]["repo"] == "a/b"


def test_provider_wiring_smoke():
    # GitHubProvider 能引用 capabilities（不触发网络）
    from hot_projects.providers.github.provider import GitHubProvider
    p = GitHubProvider(token_mgr=object())
    assert hasattr(p, "search_by_keywords") and hasattr(p, "batch_growth")
