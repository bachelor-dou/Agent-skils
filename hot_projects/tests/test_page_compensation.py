"""页级补偿多轮重跑的自检。

覆盖两条曾经出问题的行为：
  · 补偿前必须等 token 脱离限流冷却（否则重跑立刻再撞限流，整轮白跑）；
  · 一轮补偿收不回的失败页要继续下一轮，而不是直接报 ERROR 丢弃。
"""

import asyncio

from hot_projects.datasource.github import token_pool as tp
from hot_projects.tools.basic import core


class _FakeTask:
    """按预设脚本决定每轮还剩哪些失败页。"""

    def __init__(self, pages, recover_at_round):
        self.failed_pages = list(pages)
        self.recover_at_round = recover_at_round
        self.round_no = 0

    def run(self):
        # 到达约定轮次才收回全部失败页，之前每轮只收回一页。
        if self.round_no >= self.recover_at_round:
            self.failed_pages = []
        else:
            self.failed_pages = self.failed_pages[1:]


class _FakeDispatcher:
    def __init__(self):
        self.submitted = []

    async def submit(self, task):
        self.submitted.append(task)

    async def wait_all_done(self):
        for task in self.submitted:
            task.run()
        self.submitted = []

    async def drain_results(self):
        pass


def _clone(task, pages, round_no):
    retry = _FakeTask(pages, task.recover_at_round)
    retry.round_no = round_no
    return retry


def test_waits_for_cooldown_then_recovers_in_later_round(monkeypatch, caplog):
    clock = {"t": 1000.0}
    pool = tp.AsyncTokenPool(
        tokens=["a", "b"], time_fn=lambda: clock["t"], recovery_buffer_seconds=0.0
    )
    pool.record_rate_limited(0, clock["t"] + 40)   # 一个 token 还在冷却 40s
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)
        clock["t"] += seconds

    monkeypatch.setattr(core.asyncio, "sleep", fake_sleep)

    # 第 2 轮才收回全部失败页：单轮补偿会漏，多轮才补齐。
    task = _FakeTask([3, 4, 5], recover_at_round=2)
    asyncio.run(
        core._compensate_failed_pages(
            _FakeDispatcher(), pool, [task], _clone,
            lambda t, page: f"kw/page={page}", "自检",
        )
    )

    assert slept and slept[0] == 40, f"首轮补偿前应等满冷却，实际 {slept}"
    assert not any(r.levelname == "ERROR" for r in caplog.records), "多轮后应补齐，不该报残留"


def test_reports_error_when_rounds_exhausted(caplog):
    pool = tp.AsyncTokenPool(tokens=["a"])
    # 每轮只收回一页，3 轮补不完 5 页 → 应报残留而不是静默丢弃。
    task = _FakeTask([1, 2, 3, 4, 5], recover_at_round=99)

    asyncio.run(
        core._compensate_failed_pages(
            _FakeDispatcher(), pool, [task], _clone,
            lambda t, page: f"kw/page={page}", "自检",
        )
    )

    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert errors and "仍有 2 个失败页" in errors[0], errors


def test_seconds_until_all_cool():
    clock = {"t": 500.0}
    pool = tp.AsyncTokenPool(tokens=["a", "b"], time_fn=lambda: clock["t"], recovery_buffer_seconds=0.0)
    assert pool.seconds_until_all_cool() == 0.0

    pool.record_rate_limited(1, clock["t"] + 30)
    assert pool.seconds_until_all_cool() == 30.0  # 取最晚恢复的那个

    clock["t"] += 30
    assert pool.seconds_until_all_cool() == 0.0
