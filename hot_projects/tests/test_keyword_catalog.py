"""get_keyword_catalog 工具测试：按需返回配置词表，且不再常驻 system 提示词。"""

from hot_projects.config import SEARCH_KEYWORDS
from hot_projects.tools.tool.get_keyword_catalog import get_keyword_catalog_handler


def test_returns_full_catalog():
    out = get_keyword_catalog_handler(ctx=None, args={})
    assert out["categories"] is SEARCH_KEYWORDS
    assert "AI-Agent" in out["categories"]


def test_registry_contains_tool():
    from hot_projects.tools.registry import build_default_registry
    names = {s["function"]["name"] for s in build_default_registry().schemas()}
    assert "get_keyword_catalog" in names


def test_catalog_not_in_system_prompt():
    from hot_projects.agent.prompts import SYSTEM_PROMPT
    assert "关键词类别参考" not in SYSTEM_PROMPT
    # 词表内容不应常驻（抽查一组关键词）
    assert "mcp server" not in SYSTEM_PROMPT
