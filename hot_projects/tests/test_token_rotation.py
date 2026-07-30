"""同步 REST 链路的 token 轮换：撞限流自动换下一个 token，而不是死守 token_idx=0。

这条链路没有 AsyncTokenPool.acquire 帮它调度，调用方全部写死 token_idx=0。
一旦第一个 token 额度耗尽，report.py 那层是 except+warning：仓库静默失去 README/提交素材，
LLM 只能靠元数据编描述——不崩，但整篇报告质量一起悄悄塌。所以这里要锁住轮换行为。
"""

import time

import pytest

import hot_projects.datasource.github.api as api
from hot_projects.datasource.github.token_pool import GitHubTokenPool
from hot_projects.infra.exceptions import RateLimitError, TokenInvalidError


class _Resp:
    def __init__(self, status_code, payload=None, reset_time=None):
        self.status_code = status_code
        self._payload = payload
        self.text = ""
        self.headers = {}
        if reset_time is not None:
            self.headers["X-RateLimit-Reset"] = str(reset_time)

    def json(self):
        return self._payload


def _pool(n=3):
    return GitHubTokenPool(tokens=[f"t{i}" for i in range(n)], recovery_buffer_seconds=0.0)


def _patch_get(monkeypatch, handler):
    """按 Authorization 头分派响应，并记录被用过的 token 顺序。"""
    used = []

    def fake_get(url, headers=None, timeout=None, params=None):
        token = (headers or {}).get("Authorization", "").split()[-1]
        used.append(token)
        return handler(token)

    monkeypatch.setattr(api.requests, "get", fake_get)
    return used


def test_rate_limited_token_rotates_to_next(monkeypatch):
    pool = _pool()
    reset = time.time() + 600
    used = _patch_get(
        monkeypatch,
        lambda tok: _Resp(200, {"full_name": "o/r"}) if tok == "t1"
        else _Resp(403, reset_time=reset),
    )

    info = api.fetch_repo_info(pool, "o", "r", token_idx=0)

    assert info == {"full_name": "o/r"}       # 拿到了数据，没有因为 t0 限流而失败
    assert used == ["t0", "t1"]               # t0 撞限流后顺延到 t1
    assert pool.seconds_until_all_cool() > 0  # t0 的限流已记进池子，不是默默丢掉


def test_cooling_token_is_skipped_upfront(monkeypatch):
    """t0 已知在冷却时，下一次调用直接从可用 token 开始，不再白撞一个 403。"""
    clock = {"t": 1000.0}
    pool = GitHubTokenPool(
        tokens=["t0", "t1"], recovery_buffer_seconds=0.0, time_fn=lambda: clock["t"]
    )
    pool.record_rate_limited(0, clock["t"] + 600)
    used = _patch_get(monkeypatch, lambda tok: _Resp(200, {"full_name": "o/r"}))

    api.fetch_repo_info(pool, "o", "r", token_idx=0)

    assert used == ["t1"]  # 冷却中的 t0 被跳过，一个请求都没浪费


def test_all_tokens_limited_raises_earliest_reset(monkeypatch):
    """全军覆没时抛 reset 最早的那个，调用方才知道最短要等多久。"""
    pool = _pool()
    now = time.time()
    resets = {"t0": now + 900, "t1": now + 300, "t2": now + 600}
    _patch_get(monkeypatch, lambda tok: _Resp(403, reset_time=resets[tok]))

    with pytest.raises(RateLimitError) as err:
        api.fetch_repo_info(pool, "o", "r", token_idx=0)

    assert err.value.reset_time == pytest.approx(resets["t1"], abs=1)


def test_missing_repo_does_not_burn_other_tokens(monkeypatch):
    """404 是「查到了，没有」，属成功结果——不能触发换 token 把额度白烧一圈。"""
    pool = _pool()
    used = _patch_get(monkeypatch, lambda tok: _Resp(404))

    assert api.fetch_repo_info(pool, "o", "r", token_idx=0) is None
    assert used == ["t0"]


def test_endpoint_json_rotation_covers_readme_path(monkeypatch):
    """readme/releases/commits/tree 都走 _fetch_repo_endpoint_json，轮换必须在那一层生效。"""
    pool = _pool()
    used = _patch_get(
        monkeypatch,
        lambda tok: _Resp(200, {"content": "", "encoding": "base64"}) if tok == "t2"
        else _Resp(403, reset_time=time.time() + 600),
    )

    api.fetch_repo_readme_excerpt(pool, "o", "r", 0)

    assert used == ["t0", "t1", "t2"]  # 逐个顺延，直到有 token 能用


def test_invalid_token_rotates_then_raises(monkeypatch):
    """401 也换 token；全部失效才抛，且走 strikes 而非永久拉黑。"""
    pool = _pool(n=2)
    _patch_get(monkeypatch, lambda tok: _Resp(401))

    with pytest.raises(TokenInvalidError):
        api.fetch_repo_info(pool, "o", "r", token_idx=0)


def test_provider_search_rotates(monkeypatch):
    """搜索接口 30 次/分钟，是最容易撞限流的一条，Agent 交互式搜索必须能顺延。"""
    from hot_projects.datasource.github.provider import GitHubProvider

    pool = _pool()
    used = _patch_get(
        monkeypatch,
        lambda tok: _Resp(200, {"items": [{"full_name": "o/r", "stargazers_count": 9}]})
        if tok == "t1" else _Resp(403, reset_time=time.time() + 60),
    )

    repos = GitHubProvider(pool).search_top_repos("agent", top_n=3)

    assert [r.full_name for r in repos] == ["o/r"]
    assert used == ["t0", "t1"]


def test_worker_search_path_does_not_self_rotate(monkeypatch):
    """异步 worker 的 token 由调度器分配：search_github_repos 内部绝不能自己换 token，
    否则会用上调度器以为空闲的 token，两边争用同一批额度。"""
    pool = _pool()
    used = _patch_get(monkeypatch, lambda tok: _Resp(403, reset_time=time.time() + 60))

    with pytest.raises(RateLimitError):
        api.search_github_repos(pool, "q", token_idx=2, page=1)

    assert used == ["t2"]  # 只用了 worker 绑定的那个 token，原样把限流抛回调度器


def test_single_token_behaviour_unchanged(monkeypatch):
    """只有一个 token 时无可轮换，异常语义与从前完全一致（直接透传）。"""
    pool = _pool(n=1)
    used = _patch_get(monkeypatch, lambda tok: _Resp(403, reset_time=time.time() + 600))

    with pytest.raises(RateLimitError):
        api.fetch_repo_info(pool, "o", "r", token_idx=0)

    assert used == ["t0"]
