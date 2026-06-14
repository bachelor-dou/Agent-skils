"""Tests for async token pool state transitions."""

import asyncio

import pytest

from hot_projects.providers.github.token_pool import AsyncTokenPool


class TestAsyncTokenPool:
    def test_requires_non_empty_tokens(self):
        with pytest.raises(ValueError):
            AsyncTokenPool([])

    def test_acquire_release_reuses_token(self):
        async def scenario() -> None:
            pool = AsyncTokenPool(["t1", "t2"], recovery_buffer_seconds=0.0)
            first = await pool.acquire()
            second = await pool.acquire()
            assert {first, second} == {0, 1}

            await pool.release(first)
            third = await pool.acquire()
            assert third == first

        asyncio.run(scenario())

    def test_mark_rate_limited_applies_cooldown(self):
        async def scenario() -> None:
            clock = [100.0]
            pool = AsyncTokenPool(
                ["t1"],
                recovery_buffer_seconds=0.0,
                time_fn=lambda: clock[0],
            )

            idx = await pool.acquire()
            await pool.mark_rate_limited(idx, reset_time=100.05, reason="hit limit")

            snap = await pool.snapshot()
            assert snap[0]["in_use"] is False
            assert snap[0]["rate_limited_count"] == 1
            assert snap[0]["last_error"] == "hit limit"

            done = asyncio.Event()
            acquired: dict[str, int] = {}

            async def waiter() -> None:
                acquired["idx"] = await pool.acquire()
                done.set()

            task = asyncio.create_task(waiter())
            await asyncio.sleep(0)
            assert not done.is_set()

            clock[0] = 100.10
            await asyncio.wait_for(done.wait(), timeout=0.5)
            assert acquired["idx"] == 0
            task.cancel()

        asyncio.run(scenario())

    def test_mark_invalid_removes_token(self):
        async def scenario() -> None:
            pool = AsyncTokenPool(["t1", "t2"], recovery_buffer_seconds=0.0)

            idx = await pool.acquire()
            await pool.mark_invalid(idx, reason="401")

            next_idx = await pool.acquire()
            assert next_idx != idx

            await pool.mark_invalid(next_idx)
            with pytest.raises(RuntimeError):
                await pool.acquire()

        asyncio.run(scenario())

    def test_waits_until_release_when_all_in_use(self):
        async def scenario() -> None:
            pool = AsyncTokenPool(["t1"], recovery_buffer_seconds=0.0)

            idx = await pool.acquire()
            done = asyncio.Event()
            acquired: dict[str, int] = {}

            async def waiter() -> None:
                acquired["idx"] = await pool.acquire()
                done.set()

            task = asyncio.create_task(waiter())
            await asyncio.sleep(0)
            assert not done.is_set()

            await pool.release(idx)
            await asyncio.wait_for(done.wait(), timeout=0.5)
            assert acquired["idx"] == idx
            task.cancel()

        asyncio.run(scenario())

    def test_earliest_available_delay(self):
        async def scenario() -> None:
            clock = [10.0]
            pool = AsyncTokenPool(["t1"], recovery_buffer_seconds=0.0, time_fn=lambda: clock[0])
            idx = await pool.acquire()
            await pool.mark_rate_limited(idx, reset_time=11.0)

            delay = await pool.earliest_available_delay()
            assert delay is not None
            assert 0.9 <= delay <= 1.0

        asyncio.run(scenario())

    def test_sync_record_methods_update_state(self):
        pool = AsyncTokenPool(["t1"], recovery_buffer_seconds=0.0)

        pool.record_rate_limited(0, reset_time=123.0, reason="limit")
        assert pool._states[0].available_at == 123.0
        assert pool._states[0].rate_limited_count == 1
        assert pool._states[0].last_error == "limit"

        pool.record_invalid(0, reason="401")
        assert pool._states[0].invalid is True
        assert pool._states[0].available_at == float("inf")
        assert pool._states[0].last_error == "401"
