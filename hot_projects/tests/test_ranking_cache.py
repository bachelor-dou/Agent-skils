from hot_projects.pipeline.cache import RankingCache


def test_downstream_invalidated_when_upstream_changes():
    c = RankingCache()
    c.set("collect", {"min_star": 1200}, payload="C")
    c.set("growth_calc", {"min_star": 1200, "growth_calc_days": 7}, payload="G")
    c.set("threshold", {"min_star": 1200, "growth_calc_days": 7, "t": 800}, payload="T")
    # 重新 set 上游 collect → growth_calc/threshold 应失效
    c.set("collect", {"min_star": 2000}, payload="C2")
    assert c.get("growth_calc", {"min_star": 2000, "growth_calc_days": 7}) is None
    assert c.get("threshold", {"min_star": 2000, "growth_calc_days": 7, "t": 800}) is None


def test_threshold_change_keeps_growth():
    c = RankingCache()
    c.set("growth_calc", {"g": 7}, payload="G")
    c.set("threshold", {"g": 7, "t": 800}, payload="T800")
    # 只改 threshold 阶段，不动 growth_calc
    c.set("threshold", {"g": 7, "t": 500}, payload="T500")
    assert c.get("growth_calc", {"g": 7}) == "G"  # growth 仍在
    assert c.get("threshold", {"g": 7, "t": 500}) == "T500"
