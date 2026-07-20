"""工具结果卸载：大结果在下一轮开头转存根，recall_tool_result 可取回。"""

from hot_projects.agent.agent import HotProjectAgent, OFFLOAD_THRESHOLD
from hot_projects.tools.registry import ToolSpec, ToolRegistry
from hot_projects.tools.tool.recall_tool_result import recall_tool_result_handler


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def chat(self, messages, **kw):
        self.calls += 1
        return self._responses.pop(0)


def _toolcall(name, args="{}"):
    return {"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": name, "arguments": args}}]}}]}


def _text(t):
    return {"choices": [{"message": {"content": t}}]}


def _registry_with(name, handler):
    reg = ToolRegistry()
    schema = {"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {}}}}
    reg.register(ToolSpec(name=name, schema=schema, handler=handler))
    return reg


def test_large_result_offloaded_next_turn_and_recallable():
    big = "X" * (OFFLOAD_THRESHOLD + 500)
    reg = _registry_with("big_tool", lambda ctx, args: {"blob": big})
    # 第1轮：调 big_tool 后给结论；第2轮：直接给结论
    llm = FakeLLM([_toolcall("big_tool"), _text("第一轮结论"), _text("第二轮结论")])
    agent = HotProjectAgent(llm=llm, registry=reg, provider=None, db={"projects": {}})

    agent.chat("跑个大工具")
    # 第1轮内：tool 结果保持完整（同回合推理需要）
    tool_msgs = [m for m in agent.state.conversation if m.get("role") == "tool"]
    assert tool_msgs and big in tool_msgs[0]["content"]
    assert "offloaded" not in tool_msgs[0]["content"]

    agent.chat("继续")  # 第2轮开头触发卸载
    tool_msgs = [m for m in agent.state.conversation if m.get("role") == "tool"]
    assert big not in tool_msgs[0]["content"]          # 已被存根替换
    assert '"offloaded": true' in tool_msgs[0]["content"]

    store = agent.state.tool_state["offloaded"]
    assert len(store) == 1
    ref = next(iter(store))
    out = recall_tool_result_handler(agent.ctx, {"ref": ref})
    assert out["result"]["blob"] == big               # 取回完整内容


def test_small_result_not_offloaded():
    reg = _registry_with("small_tool", lambda ctx, args: {"ok": 1})
    llm = FakeLLM([_toolcall("small_tool"), _text("done"), _text("again")])
    agent = HotProjectAgent(llm=llm, registry=reg, provider=None, db={"projects": {}})
    agent.chat("小工具")
    agent.chat("继续")
    tool_msgs = [m for m in agent.state.conversation if m.get("role") == "tool"]
    assert "offloaded" not in tool_msgs[0]["content"]
    assert agent.state.tool_state.get("offloaded", {}) == {}


def test_recall_unknown_ref():
    reg = _registry_with("x", lambda ctx, args: {})
    llm = FakeLLM([_text("hi")])
    agent = HotProjectAgent(llm=llm, registry=reg, provider=None, db={"projects": {}})
    out = recall_tool_result_handler(agent.ctx, {"ref": "tr999"})
    assert "error" in out


def test_registry_contains_recall():
    from hot_projects.tools.registry import build_default_registry
    names = {s["function"]["name"] for s in build_default_registry().schemas()}
    assert "recall_tool_result" in names
