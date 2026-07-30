import hot_projects.tools.tool.ranking as RT
from hot_projects.tools.tool.ranking import make_ranking_handler


class _State:
    def __init__(self):
        self.pending_confirmation_signature = None
        self.tool_state = {}


class _Ctx:
    def __init__(self):
        self.state = _State()
        self.provider = None
        self.db = {"valid": False, "projects": {}}


def test_first_call_asks_confirmation():
    ctx = _Ctx()
    out = make_ranking_handler("comprehensive")(ctx, {"min_star": 1200})
    assert out.get("needs_confirmation") is True
    assert ctx.state.pending_confirmation_signature is not None


def test_second_call_same_sig_executes(monkeypatch):
    monkeypatch.setattr(RT, "run_ranking", lambda *a, **k: {"ranked": [("a/b", {"growth": 9, "star": 1500})],
                                                            "report_path": "/x.md", "mode": "comprehensive",
                                                            "candidates_count": 1, "growth_calc_days": 7})
    monkeypatch.setattr(RT, "save_db_desc_only", lambda db: 0)
    ctx = _Ctx()
    h = make_ranking_handler("comprehensive")
    h(ctx, {"min_star": 1200})            # 第一次 → 确认
    out = h(ctx, {"min_star": 1200})      # 第二次同签名 → 执行
    assert out["ranked_count"] == 1
    assert out["report_path"] == "/x.md"
    assert ctx.state.pending_confirmation_signature is None


def test_param_change_requires_new_confirmation(monkeypatch):
    monkeypatch.setattr(RT, "run_ranking", lambda *a, **k: {"ranked": [], "report_path": "", "mode": "comprehensive",
                                                            "candidates_count": 0, "growth_calc_days": 7})
    monkeypatch.setattr(RT, "save_db_desc_only", lambda db: 0)
    ctx = _Ctx()
    h = make_ranking_handler("comprehensive")
    h(ctx, {"min_star": 1200})
    h(ctx, {"min_star": 1200})            # 执行，清空签名
    out = h(ctx, {"min_star": 2000})      # 新参数 → 再次确认
    assert out.get("needs_confirmation") is True


def test_confirm_true_executes_stored_params_ignoring_drift(monkeypatch):
    # 用户"开始"→ 模型带 confirm=true 复调但关键词漂移；应按首次存下的参数执行（展示=执行）
    seen = {}

    def fake_run(provider, mode, params, db, **k):
        seen["params"] = params
        return {"ranked": [], "report_path": "", "mode": mode,
                "candidates_count": 0, "growth_calc_days": 7}

    monkeypatch.setattr(RT, "run_ranking", fake_run)
    monkeypatch.setattr(RT, "save_db_desc_only", lambda db: 0)
    ctx = _Ctx()
    h = make_ranking_handler("keyword")
    first = h(ctx, {"keywords": ["a", "b"], "top_n": 10, "growth_threshold": 0})
    assert first.get("needs_confirmation") is True
    # confirm=true，但关键词漂移成 4 个
    h(ctx, {"keywords": ["a", "b", "c", "d"], "top_n": 10, "growth_threshold": 0, "confirm": True})
    assert seen["params"]["keywords"] == ["a", "b"]  # 按首次回显执行，忽略漂移
    assert ctx.state.pending_confirmation_signature is None


def test_keyword_ranking_growth_threshold_defaults_zero():
    """关键词榜默认 0（不过滤增长），综合榜继承配置里的出榜阈值。

    综合榜这条断言比的是配置本身而不是字面量 1000：STAR_GROWTH_THRESHOLD 是预期会往上调的
    旋钮，写死数字会让调阈值时莫名红一个与阈值无关的测试。本测试要守的是"两榜默认值不同"。
    """
    from hot_projects.config import STAR_GROWTH_THRESHOLD
    from hot_projects.tools.arg_validator import validate_tool_args_strict
    kw, errs = validate_tool_args_strict("keyword_ranking", {"keywords": ["x"]})
    assert errs == []
    assert kw["growth_threshold"] == 0
    comp, _ = validate_tool_args_strict("comprehensive_ranking", {})
    assert comp["growth_threshold"] == STAR_GROWTH_THRESHOLD
    assert STAR_GROWTH_THRESHOLD > 0, "综合榜必须真的过滤增长，否则和关键词榜没区别"


def test_format_confirm_lists_all_effective_params():
    from hot_projects.tools.tool.ranking import _format_confirm
    msg = _format_confirm("keyword", {"keywords": ["a"], "top_n": 10,
                                      "min_star": 1, "growth_threshold": 0})
    assert "增长阈值=0" in msg and "Top 10" in msg and "最低 star=1" in msg


# ── 报告开关：默认不落报告文件，用户明确要才生成 ──

def _run_confirmed(monkeypatch, mode, args):
    """跑完「确认 → 执行」两步，返回传给 run_ranking 的 do_report。"""
    seen = {}

    def fake_run(provider, mode, params, db, **kwargs):
        seen["do_report"] = kwargs.get("do_report")
        return {"ranked": [], "report_path": "", "mode": mode,
                "candidates_count": 0, "growth_calc_days": 7}

    monkeypatch.setattr(RT, "run_ranking", fake_run)
    monkeypatch.setattr(RT, "save_db_desc_only", lambda db: 0)
    ctx = _Ctx()
    h = make_ranking_handler(mode)
    first = h(ctx, dict(args))
    h(ctx, dict(args, confirm=True))
    return seen["do_report"], first["message"]


def test_ranking_skips_report_by_default(monkeypatch):
    for mode in ("comprehensive", "hot_new", "keyword"):
        do_report, msg = _run_confirmed(monkeypatch, mode, {"min_star": 1200})
        assert do_report is False, f"{mode} 默认不该生成报告"
        assert "不生成报告文件" in msg  # 确认文案要说清楚，展示=执行


def test_ranking_generates_report_when_asked(monkeypatch):
    do_report, msg = _run_confirmed(
        monkeypatch, "comprehensive", {"min_star": 1200, "generate_report": True})
    assert do_report is True
    assert "生成报告文件" in msg


def test_generate_report_defaults_false_in_schema():
    from hot_projects.tools.arg_validator import validate_tool_args_strict
    for tool in ("comprehensive_ranking", "hot_new_ranking", "keyword_ranking"):
        args, errs = validate_tool_args_strict(tool, {})
        assert errs == []
        assert args["generate_report"] is False, f"{tool} 的报告开关默认必须是关"
