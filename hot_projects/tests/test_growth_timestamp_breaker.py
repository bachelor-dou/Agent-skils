"""star 时间戳路径熔断的自检。

GitHub 已停供 star 时间戳（REST stargazers 404 / GraphQL 空 edges），逐仓库硬试只会空转。
这里验证：连续失败到阈值后熔断，且熔断期间不再发任何请求；拿到时间戳则立刻恢复。
"""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from hot_projects.datasource.github import growth_estimator as ge


@pytest.fixture(autouse=True)
def _clean_breaker():
    ge.reset_timestamp_path_state()
    yield
    ge.reset_timestamp_path_state()


def _drain_timestamps(times: int) -> None:
    """模拟 times 个仓库采样后一条时间戳都没拿到。"""
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    for _ in range(times):
        assert ge._estimate_growth_from_sampling_timestamps(
            "o", "r", [], cutoff, 7
        ) == ge.GROWTH_ESTIMATION_UNRESOLVED


def test_breaker_trips_only_after_strike_limit():
    _drain_timestamps(ge.TIMESTAMP_PATH_STRIKE_LIMIT - 1)
    assert ge.timestamp_path_unavailable() is False, "未到阈值不应熔断"

    _drain_timestamps(1)
    assert ge.timestamp_path_unavailable() is True


def test_no_request_while_tripped():
    _drain_timestamps(ge.TIMESTAMP_PATH_STRIKE_LIMIT)

    # token_mgr 传 None：一旦真去发请求就会 AttributeError，能证明是直接短路返回的。
    growth = asyncio.run(
        ge.estimate_star_growth_binary_async(None, "o", "r", total_stars=50_000, token_idx=0)
    )
    assert growth == ge.GROWTH_ESTIMATION_UNRESOLVED

    assert ge.estimate_star_growth_binary(None, "o", "r", 50_000) == ge.GROWTH_ESTIMATION_UNRESOLVED


def test_success_resets_strikes():
    _drain_timestamps(ge.TIMESTAMP_PATH_STRIKE_LIMIT - 1)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=7)
    # 两条窗口内、且最老一条早于 cutoff 之后 → 走正常估算，视为该路径可用。
    ge._estimate_growth_from_sampling_timestamps(
        "o", "r", [now - timedelta(days=1), now - timedelta(hours=2)], cutoff, 7
    )

    _drain_timestamps(ge.TIMESTAMP_PATH_STRIKE_LIMIT - 1)
    assert ge.timestamp_path_unavailable() is False, "成功一次后计数应清零，不该提前熔断"
