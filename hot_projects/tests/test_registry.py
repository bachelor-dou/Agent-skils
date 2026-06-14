from hot_projects.tools.registry import ToolSpec, ToolRegistry, build_default_registry


def test_register_and_dispatch():
    reg = ToolRegistry()
    reg.register(ToolSpec(name="echo", schema={}, handler=lambda ctx, args: {"echo": args}))
    assert reg.get("echo").expensive is False
    assert reg.dispatch("echo", ctx=None, args={"x": 1}) == {"echo": {"x": 1}}


def test_unknown_tool():
    out = ToolRegistry().dispatch("nope", ctx=None, args={})
    assert "error" in out


def test_default_registry_has_seven_tools():
    reg = build_default_registry()
    assert len(reg.schemas()) == 7
    assert reg.get("comprehensive_ranking").expensive is True
    assert reg.get("repo_growth").expensive is False
    names = {s["function"]["name"] for s in reg.schemas()}
    assert names == {"comprehensive_ranking", "hot_new_ranking", "keyword_ranking",
                     "repo_growth", "describe_project", "get_db_info", "fetch_trending"}
