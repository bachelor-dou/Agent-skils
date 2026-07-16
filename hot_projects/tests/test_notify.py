"""Server酱 微信推送测试。"""

import hot_projects.infra.notify as notify


def test_no_key_skips(monkeypatch):
    monkeypatch.setattr(notify, "SERVERCHAN_SENDKEY", "")
    called = {"n": 0}
    monkeypatch.setattr(notify.requests, "post", lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    assert notify.send("t", "d") is False
    assert called["n"] == 0  # 未配 key 时不发请求


class _Resp:
    status_code = 200

    def json(self):
        return {"code": 0}

    text = "ok"


def test_with_key_posts(monkeypatch):
    monkeypatch.setattr(notify, "SERVERCHAN_SENDKEY", "SCTxxx")
    captured = {}

    def fake_post(url, data=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        return _Resp()

    monkeypatch.setattr(notify.requests, "post", fake_post)
    assert notify.send("标题", "正文") is True
    assert "SCTxxx" in captured["url"]
    assert captured["data"]["title"] == "标题"
    assert captured["data"]["desp"] == "正文"


def test_failure_never_raises(monkeypatch):
    monkeypatch.setattr(notify, "SERVERCHAN_SENDKEY", "SCTxxx")

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(notify.requests, "post", boom)
    assert notify.send("t", "d") is False  # 异常被吞，不抛出
