"""LLM 层的守卫:目录归一化、后端参数白名单、回退顺序、流式拼装。

一次网络都不发 —— `requests.post` 整个被替换掉。这些逻辑历史上出过的问题
(给 azure 发 temperature 直接 400、硬切换悄悄回退到别家、流式重试导致前端文字重复)
全都不需要真实的 LLM 才能复现。
"""

import json

import pytest

from hot_project.infra.llm import wire
from hot_project.infra.llm.client import LLMClient
from hot_project.infra.llm.schemes import Scheme, build


def scheme(sid="p1", backend="openai", key="k", lite=()):
    return Scheme(id=sid, label=sid, backend=backend, url=f"https://{sid}.test",
                  model=f"{sid}-main", key=key, lite_models=list(lite))


class Recorder:
    """记下每次请求发到了哪、发了什么;按预设脚本回复。"""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url, headers=None, json=None, timeout=None, stream=False):
        self.calls.append((url, json))
        reply = self.replies.pop(0) if self.replies else _ok("ok")
        if callable(reply):
            reply = reply()
        return reply

    @property
    def urls(self):
        return [url for url, _ in self.calls]


class _Resp:
    def __init__(self, status=200, body=None, lines=None, text=""):
        self.status_code = status
        self._body = body
        self._lines = lines or []
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)

    def close(self):
        self.closed = True


def _ok(content, tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return _Resp(body={"choices": [{"message": msg}]})


def _sse(*chunks):
    lines = [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]
    return _Resp(lines=lines)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """退避重试的 sleep 直接跳过 —— 测的是重试逻辑,不是计时器。"""
    monkeypatch.setattr(wire.time, "sleep", lambda _s: None)


@pytest.fixture
def post(monkeypatch):
    def install(*replies):
        rec = Recorder(*replies)
        monkeypatch.setattr(wire.requests, "post", rec)
        return rec
    return install


# ── 目录归一化 ────────────────────────────────────────────────────

def test_key_comes_from_the_environment_not_the_catalog():
    """目录里只有 key_env,真值到这里才落地 —— 这是 LLM_MODELS 能安全序列化的前提。"""
    got = build([{"id": "a", "url": "u", "model": "m", "key_env": "SOME_KEY"}],
                lambda name: "secret" if name == "SOME_KEY" else "")
    assert got[0].key == "secret"
    assert "secret" not in json.dumps(got[0].public())


def test_a_disabled_platform_disappears_entirely():
    got = build([{"id": "a", "enabled": 0}, {"id": "b", "enabled": 1}], lambda _n: "k")
    assert [s.id for s in got] == ["b"]


def test_enabled_as_the_string_zero_still_means_off():
    """手写配置里 `"enabled": "0"` 很自然(旁边几行都带引号)。裸 bool() 会判成真,
    后果是关掉的平台照样被调用 —— 不报错,只是账单上多一笔。"""
    assert build([{"id": "a", "enabled": "0"}], lambda _n: "k") == []


def test_duplicate_ids_are_a_hard_error():
    """id 是整条选择链路的键;重复会让「选 A 却调到 B」静默发生。"""
    with pytest.raises(ValueError, match="id 重复"):
        build([{"id": "same"}, {"id": "same"}], lambda _n: "k")


def test_lite_models_split_and_dedupe():
    got = build([{"id": "a", "lite_model": " x , y , x "}], lambda _n: "k")
    assert got[0].lite_models == ["x", "y"]


def test_a_platform_without_a_key_is_not_usable():
    assert not scheme(key="").usable


# ── 后端参数白名单 ────────────────────────────────────────────────

def test_azure_gets_max_completion_tokens_and_no_temperature():
    """发了 azure 不认的参数不是被忽略,是整个请求 400。"""
    body = wire.payload("azure", "m", [], max_tokens=100, temperature=0.5,
                        enable_thinking=True, thinking_budget=64)
    assert body["max_completion_tokens"] == 100
    assert "max_tokens" not in body
    assert "temperature" not in body
    assert "enable_thinking" not in body
    assert "thinking_budget" not in body


def test_openai_gets_the_full_parameter_set():
    body = wire.payload("openai", "m", [], max_tokens=100, temperature=0.5,
                        enable_thinking=True, thinking_budget=64)
    assert body["max_tokens"] == 100
    assert body["temperature"] == 0.5
    assert body["enable_thinking"] is True


def test_azure_authenticates_with_a_header_of_its_own():
    assert wire.headers("azure", "k") == {"api-key": "k", "Content-Type": "application/json"}
    assert wire.headers("openai", "k")["Authorization"] == "Bearer k"


# ── 回退顺序 ──────────────────────────────────────────────────────

def test_an_internal_call_falls_through_to_the_next_platform(post):
    rec = post(_Resp(status=500, text="boom"), _Resp(status=500, text="boom"),
               _Resp(status=500, text="boom"), _ok("from second"))
    client = LLMClient([scheme("p1"), scheme("p2")])
    assert client.text("hi") == "from second"
    assert rec.urls[-1] == "https://p2.test"


def test_a_hard_switch_never_falls_back(post):
    """用户明确选了 A,悄悄给出 B 的答案是另一回事 —— 宁可返回 None。"""
    rec = post(*[_Resp(status=500, text="boom")] * 3)
    client = LLMClient([scheme("p1"), scheme("p2")])
    assert client.chat([{"role": "user", "content": "hi"}], model_id="p1") is None
    assert set(rec.urls) == {"https://p1.test"}


def test_asking_for_a_platform_that_is_not_configured_fails_fast(post):
    rec = post()
    assert LLMClient([scheme("p1")]).chat([], model_id="nope") is None
    assert rec.calls == []


def test_an_empty_reply_counts_as_failure(post):
    """200 但正文为空是最难查的一类失败:不换平台的话,整批描述静默变空。"""
    rec = post(_ok(""), _ok(""), _ok(""), _ok("real"))
    assert LLMClient([scheme("p1"), scheme("p2")]).text("hi") == "real"
    assert rec.urls[-1] == "https://p2.test"


def test_a_reply_with_only_tool_calls_is_not_empty(post):
    """agent 那条路上模型只回 tool_calls、正文为空是正常的,不能当失败重试。"""
    post(_ok(None, tool_calls=[{"id": "1"}]))
    data = LLMClient([scheme("p1")]).chat([], tools=[{"type": "function"}])
    assert data["choices"][0]["message"]["tool_calls"]


def test_a_stream_of_only_tool_calls_is_not_empty_either(post):
    """流式下同理,而且这是 agent 用工具时的常态。当成空响应会重试三遍、重复调工具。"""
    rec = post(_sse({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "c1", "function": {"name": "search", "arguments": "{}"}}]}}]}))
    data = LLMClient([scheme("p1")]).chat([], tools=[{"type": "function"}],
                                          on_delta=lambda _p: None)
    assert data["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "search"
    assert len(rec.calls) == 1


def test_lite_borrows_a_sub_model_from_another_platform(post):
    """主模型选 p1,但只有 p2 配了子模型 —— 便宜活儿该借 p2 干。"""
    rec = post(_ok("cheap"))
    client = LLMClient([scheme("p1"), scheme("p2", lite=["tiny"])])
    assert client.text("hi", lite=True, model_id="p1") == "cheap"
    assert rec.calls[0][1]["model"] == "tiny"


def test_lite_falls_back_to_the_main_model_when_nobody_has_one(post):
    """全都没配子模型时退回主模型:贵,但至少调得通。"""
    rec = post(_ok("x"))
    LLMClient([scheme("p1")]).text("hi", lite=True)
    assert rec.calls[0][1]["model"] == "p1-main"


def test_an_unconfigured_platform_is_skipped_without_a_request(post):
    rec = post(_ok("x"))
    LLMClient([scheme("p1", key=""), scheme("p2")]).text("hi")
    assert rec.urls == ["https://p2.test"]


# ── 流式 ──────────────────────────────────────────────────────────

def test_streaming_pieces_are_emitted_live_and_joined(post):
    post(_sse({"choices": [{"delta": {"content": "你"}}]},
              {"choices": [{"delta": {"content": "好"}}]}))
    seen = []
    data = LLMClient([scheme("p1")]).chat([], on_delta=seen.append)
    assert seen == ["你", "好"]
    assert data["choices"][0]["message"]["content"] == "你好"


def test_tool_call_fragments_merge_by_index():
    """OpenAI 流式约定:首片带 name,后续片只带 arguments 的一截,要按 index 拼。"""
    acc = {}
    wire.merge_toolcall_fragment(acc, {"index": 0, "id": "c1",
                                       "function": {"name": "search", "arguments": '{"q"'}})
    wire.merge_toolcall_fragment(acc, {"index": 0, "function": {"arguments": ':"x"}'}})
    assert acc[0]["function"] == {"name": "search", "arguments": '{"q":"x"}'}


def test_repeated_index_in_one_chunk_lands_in_the_same_slot():
    """vLLM 的推测解码会在同一 chunk 里重复 index;各成一项的话参数会被切两半。"""
    acc = {}
    for frag in ({"index": 0, "function": {"arguments": "ab"}},
                 {"index": 0, "function": {"arguments": "cd"}}):
        wire.merge_toolcall_fragment(acc, frag)
    assert len(acc) == 1 and acc[0]["function"]["arguments"] == "abcd"


def test_a_leaked_toolcall_blob_never_reaches_the_user(post):
    """有些模型把本该走 tool_calls 的调用当文本吐在正文开头,这段不能给用户看见。"""
    leak = '{"tool_uses":[{"recipient_name":"functions.x","parameters":{}}]}真正的回答'
    post(_sse({"choices": [{"delta": {"content": leak}}]}))
    seen = []
    LLMClient([scheme("p1")]).chat([], on_delta=seen.append)
    assert "".join(seen) == "真正的回答"


def test_an_answer_that_legitimately_starts_with_json_is_left_alone(post):
    """用户问「给我一段 JSON」时,回答本来就以 `{` 开头 —— 不能误剥。"""
    answer = '{"name": "demo", "version": 1}'
    post(_sse({"choices": [{"delta": {"content": answer}}]}))
    seen = []
    LLMClient([scheme("p1")]).chat([], on_delta=seen.append)
    assert "".join(seen) == answer


def test_an_unclosed_opening_brace_is_still_released_at_the_end(post):
    """收流结束仍卡在闸门里(花括号始终没闭合):放行,不能把用户的内容吞了。"""
    post(_sse({"choices": [{"delta": {"content": '{"half": '}}]}))
    seen = []
    LLMClient([scheme("p1")]).chat([], on_delta=seen.append)
    assert "".join(seen) == '{"half": '


def test_prose_is_emitted_without_waiting_for_the_gate(post):
    """绝大多数答案以散文开头。一律缓冲到能判定的话,每个回答的首字延迟都要买单。"""
    post(_sse({"choices": [{"delta": {"content": "这"}}]},
              {"choices": [{"delta": {"content": "是答案"}}]}))
    seen = []
    LLMClient([scheme("p1")]).chat([], on_delta=seen.append)
    assert seen[0] == "这"          # 第一片就出去了,没等第二片


class _DiesMidStream(_Resp):
    """发出一片正文之后连接断掉 —— 前端已经看到「半」了。"""

    def iter_lines(self, decode_unicode=False):
        yield f'data: {json.dumps({"choices": [{"delta": {"content": "半"}}]})}'
        raise wire.requests.ConnectionError("连接断了")


def test_a_stream_that_already_emitted_is_never_retried(post):
    """重试会让前端看到重复的文字。宁可只返回已收到的部分。"""
    rec = post(_DiesMidStream(), _sse({"choices": [{"delta": {"content": "完整答案"}}]}))
    seen = []
    LLMClient([scheme("p1"), scheme("p2")]).chat([], on_delta=seen.append)
    assert seen == ["半"]
    assert len(rec.calls) == 1      # 没有第二次请求,也没换平台


def test_a_stream_that_died_before_emitting_can_still_retry(post):
    rec = post(_Resp(status=500, text="boom"), _sse({"choices": [{"delta": {"content": "好"}}]}))
    seen = []
    LLMClient([scheme("p1")]).chat([], on_delta=seen.append)
    assert seen == ["好"]
    assert len(rec.calls) == 2


def test_malformed_sse_lines_are_skipped_not_fatal(post):
    post(_Resp(lines=["", ": comment", "event: ping", "data: {half",
                      f'data: {json.dumps({"choices": [{"delta": {"content": "ok"}}]})}',
                      "data: [DONE]"]))
    seen = []
    LLMClient([scheme("p1")]).chat([], on_delta=seen.append)
    assert seen == ["ok"]


def test_the_connection_is_returned_on_every_path(post):
    """非 200 早退这条路上忘了 close,连接池会慢慢耗干 —— 症状出现在几小时之后。"""
    dead = _Resp(status=500, text="boom")
    post(dead, dead, dead)
    LLMClient([scheme("p1")]).chat([], on_delta=lambda _p: None)
    assert dead.closed
