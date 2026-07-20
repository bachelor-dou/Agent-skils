from unittest.mock import patch

from hot_projects.infra import llm_client
from hot_projects.infra.llm_client import build_payload, build_headers, LLMClient, LLMScheme


# ── 参数/头适配 ──

def test_azure_payload_uses_max_completion_tokens_and_drops_thinking():
    payload = build_payload(
        backend="azure", model="gpt-5.4",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100, temperature=0.3, enable_thinking=True, thinking_budget=512,
        tools=None,
    )
    assert payload["model"] == "gpt-5.4"
    assert payload["max_completion_tokens"] == 100
    assert "max_tokens" not in payload
    assert "enable_thinking" not in payload
    assert "thinking_budget" not in payload
    assert "temperature" not in payload


def test_openai_payload_keeps_legacy_params():
    payload = build_payload(
        backend="openai", model="GLM-5",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100, temperature=0.3, enable_thinking=True, thinking_budget=512,
        tools=None,
    )
    assert payload["max_tokens"] == 100
    assert payload["temperature"] == 0.3
    assert payload["enable_thinking"] is True
    assert payload["thinking_budget"] == 512


def test_payload_adds_tools():
    payload = build_payload(
        backend="azure", model="gpt-5.4",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "f"}}],
    )
    assert payload["tool_choice"] == "auto"
    assert payload["tools"][0]["function"]["name"] == "f"


def test_headers_by_backend():
    assert build_headers("azure", "K")["api-key"] == "K"
    assert "Authorization" not in build_headers("azure", "K")
    assert build_headers("openai", "K")["Authorization"] == "Bearer K"
    assert "api-key" not in build_headers("openai", "K")


# ── 多平台逐调用回退 ──

def _ok_response(content="ok"):
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {}}


def _schemes():
    a = LLMScheme("azure", "urlA", "kA", "gpt-5.4", ["gpt-5.4-mini"], id="gpt5", label="GPT")
    b = LLMScheme("openai", "urlB", "kB", "GLM-5", ["Qwen"], id="glm5", label="GLM")
    return a, b


def test_failover_uses_b_when_a_fails():
    a, b = _schemes()
    calls = []

    def fake_call(scheme, model, **kw):
        calls.append((scheme.backend, model))
        return None if scheme.backend == "azure" else _ok_response("from-B")

    with patch.object(llm_client, "_request_once", side_effect=fake_call):
        resp = LLMClient([a, b]).chat([{"role": "user", "content": "hi"}], lite=False)
    assert resp["choices"][0]["message"]["content"] == "from-B"
    assert calls[0][0] == "azure" and calls[1][0] == "openai"


def test_no_failover_when_a_ok():
    a, b = _schemes()
    calls = []

    def fake_call(scheme, model, **kw):
        calls.append(scheme.backend)
        return _ok_response("from-A")

    with patch.object(llm_client, "_request_once", side_effect=fake_call):
        resp = LLMClient([a, b]).chat([{"role": "user", "content": "hi"}], lite=False)
    assert resp["choices"][0]["message"]["content"] == "from-A"
    assert calls == ["azure"]


def test_lite_uses_lite_model():
    a, b = _schemes()
    seen = []

    def fake_call(scheme, model, **kw):
        seen.append(model)
        return _ok_response()

    with patch.object(llm_client, "_request_once", side_effect=fake_call):
        LLMClient([a, b]).chat([{"role": "user", "content": "hi"}], lite=True)
    assert seen == ["gpt-5.4-mini"]


def test_both_fail_returns_none():
    a, b = _schemes()
    with patch.object(llm_client, "_request_once", side_effect=lambda *a, **k: None):
        assert LLMClient([a, b]).chat([{"role": "user", "content": "hi"}]) is None


def test_model_id_hard_switch_uses_only_selected():
    a, b = _schemes()
    calls = []

    def fake_call(scheme, model, **kw):
        calls.append(scheme.id)
        return _ok_response(f"from-{scheme.id}")

    with patch.object(llm_client, "_request_once", side_effect=fake_call):
        resp = LLMClient([a, b]).chat([{"role": "user", "content": "hi"}], model_id="glm5")
    assert resp["choices"][0]["message"]["content"] == "from-glm5"
    assert calls == ["glm5"]  # 只用选中的，不碰 gpt5


def test_model_id_hard_switch_no_fallback_on_failure():
    a, b = _schemes()
    calls = []

    def fake_call(scheme, model, **kw):
        calls.append(scheme.id)
        return None  # 选中的失败

    with patch.object(llm_client, "_request_once", side_effect=fake_call):
        resp = LLMClient([a, b]).chat([{"role": "user", "content": "hi"}], model_id="gpt5")
    assert resp is None
    assert calls == ["gpt5"]  # 不回退到 glm5


def test_model_id_unknown_returns_none_without_calling():
    a, b = _schemes()
    with patch.object(llm_client, "_request_once", side_effect=AssertionError("不应被调用")):
        assert LLMClient([a, b]).chat([{"role": "user", "content": "hi"}], model_id="nope") is None


def test_usable_filters_unconfigured():
    a = LLMScheme("azure", "urlA", "kA", "m", ["lm"], id="a")
    b = LLMScheme("openai", "", "", "m", ["lm"], id="b")  # 无 key/url
    assert [s.id for s in LLMClient([a, b]).usable()] == ["a"]


# ── 子模型池（跨平台共享）与预检 ──

def test_lite_falls_back_to_main_only_when_no_platform_has_lite():
    # 所有可用平台都没配子模型时，才最终退回主模型（保证还能调通）
    a = LLMScheme("azure", "urlA", "kA", "gpt-5.4", [], id="gpt5")
    seen = []

    def fake_call(scheme, model, **kw):
        seen.append(model)
        return _ok_response()

    with patch.object(llm_client, "_request_once", side_effect=fake_call):
        LLMClient([a]).chat([{"role": "user", "content": "hi"}], lite=True, model_id="gpt5")
    assert seen == ["gpt-5.4"]


def test_lite_auto_borrows_submodel_from_other_platform_when_preferred_has_none():
    # 主模型平台没子模型 → 不用它的主模型，按 config 顺序借下一个有子模型的平台
    a = LLMScheme("azure", "urlA", "kA", "gpt-5.4", [], id="gpt5")          # 无子模型
    b = LLMScheme("openai", "urlB", "kB", "GLM-5", ["Qwen"], id="glm5")     # 有子模型
    seen = []

    def fake_call(scheme, model, **kw):
        seen.append((scheme.id, model))
        return _ok_response()

    with patch.object(llm_client, "_request_once", side_effect=fake_call):
        LLMClient([a, b]).chat([{"role": "user", "content": "hi"}], lite=True, model_id="gpt5")
    assert seen == [("glm5", "Qwen")]  # 借了 glm5 的子模型，而非用 gpt5 主模型


def test_lite_auto_prefers_main_platform_submodel_then_others():
    # 主模型平台有子模型 → 先用它自己的；失败才顺延借其它平台的子模型
    a = LLMScheme("azure", "urlA", "kA", "gpt-5.4", ["gpt-5.4-mini"], id="gpt5")
    b = LLMScheme("openai", "urlB", "kB", "GLM-5", ["Qwen"], id="glm5")
    seen = []

    def fake_call(scheme, model, **kw):
        seen.append((scheme.id, model))
        return None if scheme.id == "gpt5" else _ok_response()  # 主平台子模型失败

    with patch.object(llm_client, "_request_once", side_effect=fake_call):
        LLMClient([a, b]).chat([{"role": "user", "content": "hi"}], lite=True, model_id="gpt5")
    assert seen == [("gpt5", "gpt-5.4-mini"), ("glm5", "Qwen")]  # 先自己，再借别人


def test_lite_id_borrows_other_platform_submodel():
    # 主模型选 gpt5，但子模型指定 glm5 平台的 Qwen：请求应发给 glm5 平台
    a, b = _schemes()
    seen = []

    def fake_call(scheme, model, **kw):
        seen.append((scheme.id, model))
        return _ok_response()

    with patch.object(llm_client, "_request_once", side_effect=fake_call):
        LLMClient([a, b]).chat([{"role": "user", "content": "hi"}],
                               lite=True, model_id="gpt5", lite_id="glm5:Qwen")
    assert seen == [("glm5", "Qwen")]


def test_lite_id_unknown_returns_none_without_calling():
    a, b = _schemes()
    with patch.object(llm_client, "_request_once", side_effect=AssertionError("不应被调用")):
        client = LLMClient([a, b])
        assert client.chat([{"role": "user", "content": "hi"}], lite=True, lite_id="glm5:nope") is None
        assert client.chat([{"role": "user", "content": "hi"}], lite=True, lite_id="nope:Qwen") is None


def test_test_model_preflight():
    a, b = _schemes()
    with patch.object(llm_client, "_request_once", side_effect=lambda *x, **k: _ok_response()):
        assert LLMClient([a, b]).test_model(model_id="gpt5") is True
        assert LLMClient([a, b]).test_model(lite_id="glm5:Qwen") is True
    with patch.object(llm_client, "_request_once", side_effect=lambda *x, **k: None):
        assert LLMClient([a, b]).test_model(model_id="gpt5") is False
        assert LLMClient([a, b]).test_model(lite_id="glm5:Qwen") is False


def test_config_normalizes_lite_models_and_enabled():
    # config 加载时：enabled=False 条目被过滤；lite_model 逗号串解析为 lite_models 列表
    from hot_projects import config as cfg

    for m in cfg.LLM_MODELS:
        assert m.get("enabled", True) is True
        assert isinstance(m["lite_models"], list)
        for name in m["lite_models"]:
            assert name == name.strip() and name
        # 仅平台内去重：各平台保留自己的子模型（跨平台去重放在 api 层，不污染内部回退）
        assert len(m["lite_models"]) == len(set(m["lite_models"]))


def test_enabled_accepts_int_bool_and_string():
    # enabled 支持 1/0、True/False、"1"/"0" 等；字符串 "0" 不会被误判为开启
    from hot_projects.infra.llm_client import normalize_models as _normalize_models

    raw = [
        {"id": "a", "enabled": 1},
        {"id": "b", "enabled": 0},        # int 0 关闭
        {"id": "c", "enabled": True},
        {"id": "d", "enabled": False},    # False 关闭
        {"id": "e", "enabled": "0"},      # 字符串 "0" 关闭（防真值 footgun）
        {"id": "f", "enabled": "1"},
        {"id": "g"},                       # 缺省 → 开启
    ]
    kept = [m["id"] for m in _normalize_models(raw)]
    assert kept == ["a", "c", "f", "g"]


def test_api_models_lite_pool_dedup_across_platforms(monkeypatch):
    # /api/models 的融合池按名字跨平台去重，保留先出现的平台；各平台内部 lite_models 不受影响
    import asyncio
    from hot_projects import api_server

    fake = [
        {"id": "p1", "label": "P1", "key": "k", "lite_models": ["a", "b"]},
        {"id": "p2", "label": "P2", "key": "k", "lite_models": ["b", "c"]},  # b 与 p1 撞名
        {"id": "p3", "label": "P3", "key": "", "lite_models": ["d"]},        # 无 key，整条不出现
    ]
    monkeypatch.setattr(api_server, "LLM_MODELS", fake)
    out = asyncio.run(api_server.list_models())
    assert [m["id"] for m in out["models"]] == ["p1", "p2"]
    assert [x["id"] for x in out["lite_models"]] == ["p1:a", "p1:b", "p2:c"]
