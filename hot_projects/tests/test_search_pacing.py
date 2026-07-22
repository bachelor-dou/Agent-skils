"""Search API 按 token 主动配速的自检。"""

import asyncio

from hot_projects.datasource.github import token_pool as tp


def test_throttle_search_spacing_per_token(monkeypatch):
    clock = {"t": 1000.0}
    slept: list[float] = []
    pool = tp.AsyncTokenPool(
        tokens=["a", "b"], time_fn=lambda: clock["t"], search_min_interval=2.0
    )

    async def fake_sleep(seconds):
        slept.append(seconds)
        clock["t"] += seconds

    monkeypatch.setattr(tp.asyncio, "sleep", fake_sleep)

    async def run():
        await pool.throttle_search(0)   # 首次：不等待，next_at[0]=1002
        await pool.throttle_search(0)   # 立刻再来：需等 2s
        await pool.throttle_search(1)   # 另一个 token 独立：不等待

    asyncio.run(run())
    assert slept == [2.0]  # 仅同 token 的第二次等待；token 之间互不影响


def test_throttle_search_disabled_when_interval_zero(monkeypatch):
    slept: list[float] = []
    pool = tp.AsyncTokenPool(tokens=["a"], search_min_interval=0.0)

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(tp.asyncio, "sleep", fake_sleep)

    async def run():
        await pool.throttle_search(0)
        await pool.throttle_search(0)

    asyncio.run(run())
    assert slept == []  # 间隔为 0 时完全不介入
