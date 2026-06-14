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


# ── A/B 逐调用回退 ──

def _ok_response(content="ok"):
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {}}


def _schemes():
    a = LLMScheme("azure", "urlA", "kA", "gpt-5.4", "gpt-5.4-mini")
    b = LLMScheme("openai", "urlB", "kB", "GLM-5", "Qwen")
    return a, b


def test_failover_uses_b_when_a_fails():
    a, b = _schemes()
    calls = []

    def fake_call(scheme, model, **kw):
        calls.append((scheme.backend, model))
        return None if scheme.backend == "azure" else _ok_response("from-B")

    with patch.object(llm_client, "_request_once", side_effect=fake_call):
        resp = LLMClient(a, b).chat([{"role": "user", "content": "hi"}], lite=False)
    assert resp["choices"][0]["message"]["content"] == "from-B"
    assert calls[0][0] == "azure" and calls[1][0] == "openai"


def test_no_failover_when_a_ok():
    a, b = _schemes()
    calls = []

    def fake_call(scheme, model, **kw):
        calls.append(scheme.backend)
        return _ok_response("from-A")

    with patch.object(llm_client, "_request_once", side_effect=fake_call):
        resp = LLMClient(a, b).chat([{"role": "user", "content": "hi"}], lite=False)
    assert resp["choices"][0]["message"]["content"] == "from-A"
    assert calls == ["azure"]


def test_lite_uses_lite_model():
    a, b = _schemes()
    seen = []

    def fake_call(scheme, model, **kw):
        seen.append(model)
        return _ok_response()

    with patch.object(llm_client, "_request_once", side_effect=fake_call):
        LLMClient(a, b).chat([{"role": "user", "content": "hi"}], lite=True)
    assert seen == ["gpt-5.4-mini"]


def test_both_fail_returns_none():
    a, b = _schemes()
    with patch.object(llm_client, "_request_once", side_effect=lambda *a, **k: None):
        assert LLMClient(a, b).chat([{"role": "user", "content": "hi"}]) is None
