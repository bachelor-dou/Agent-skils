"""
共享 Fixtures
=============
为所有测试提供常用的 mock 对象和测试数据。
"""

import os
import pytest

# 在导入项目模块之前设置环境变量，避免 TokenManager sys.exit
os.environ.setdefault("GITHUB_TOKENS", "ghp_test_token_001,ghp_test_token_002")
os.environ.setdefault("LLM_API_KEY", "test-llm-key")
os.environ.setdefault("LLM_API_URL", "https://test-llm-api.example.com/v1/chat/completions")


@pytest.fixture
def mock_token_mgr():
    """创建一个不会 sys.exit 的 TokenManager mock。"""
    from unittest.mock import MagicMock
    mgr = MagicMock()
    mgr.tokens = ["ghp_test_token_001", "ghp_test_token_002"]
    mgr.get_rest_headers.return_value = {
        "Authorization": "token ghp_test_token_001",
        "Accept": "application/vnd.github.v3+json",
    }
    mgr.get_star_headers.return_value = {
        "Authorization": "token ghp_test_token_001",
        "Accept": "application/vnd.github.v3.star+json",
    }
    mgr.get_graphql_headers.return_value = {
        "Authorization": "bearer ghp_test_token_001",
        "Content-Type": "application/json",
    }
    return mgr


@pytest.fixture
def sample_db():
    """标准测试 DB。"""
    return {
        "date": "2026-04-14",
        "valid": True,
        "projects": {
            "test-org/test-repo": {
                "star": 5000,
                "desc": "一个测试项目的详细描述。",
                "short_desc": "A test repository",
                "language": "Python",
                "topics": ["ai", "test"],
                "created_at": "2026-03-01T00:00:00Z",
              "refreshed_at": "2026-04-14T08:00:00Z",
            },
            "old-org/old-repo": {
                "star": 20000,
                "desc": "一个老项目。",
                "short_desc": "An old repository",
                "language": "Go",
                "topics": ["database"],
                "created_at": "2020-01-01T00:00:00Z",
              "refreshed_at": "2026-04-10T08:00:00Z",
            },
        },
    }


@pytest.fixture
def sample_candidates():
    """标准候选列表。"""
    return {
        "hot-org/hot-repo": {
            "growth": 2000,
            "star": 15000,
            "created_at": "2026-03-15T00:00:00Z",
        },
        "new-org/new-repo": {
            "growth": 800,
            "star": 1200,
            "created_at": "2026-04-01T00:00:00Z",
        },
        "old-org/stable-repo": {
            "growth": 500,
            "star": 80000,
            "created_at": "2019-06-01T00:00:00Z",
        },
        "tiny-org/tiny-repo": {
            "growth": 100,
            "star": 300,
            "created_at": "2026-04-10T00:00:00Z",
        },
    }


TRENDING_HTML_FIXTURE = """
<html><body>
<article class="Box-row">
  <h2 class="h3 lh-condensed">
    <a href="/trending-org/trending-repo" data-hydro-click>trending-org / trending-repo</a>
  </h2>
  <p class="col-9 color-fg-muted my-1 pr-4">A trending test project for AI</p>
  <span itemprop="programmingLanguage">Python</span>
  <a href="/trending-org/trending-repo/stargazers">
    <svg></svg>
    12,345
  </a>
  <a href="/trending-org/trending-repo/forks">
    <svg></svg>
    1,234
  </a>
  <span class="d-inline-block float-sm-right">
    567 stars this week
  </span>
</article>
<article class="Box-row">
  <h2 class="h3 lh-condensed">
    <a href="/another-org/another-repo">another-org / another-repo</a>
  </h2>
  <p class="col-9 color-fg-muted my-1 pr-4">Another project</p>
  <span itemprop="programmingLanguage">Rust</span>
  <a href="/another-org/another-repo/stargazers">
    <svg></svg>
    5,678
  </a>
  <a href="/another-org/another-repo/forks">
    <svg></svg>
    456
  </a>
  <span class="d-inline-block float-sm-right">
    234 stars this week
  </span>
</article>
</body></html>
"""
