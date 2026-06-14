import hot_projects.tools.ranking_tools as RT
from hot_projects.tools.ranking_tools import make_ranking_handler


class _State:
    def __init__(self):
        self.pending_confirmation_signature = None
        self.ranking_cache = None


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
