from hot_projects.agent.agent import HotProjectAgent
from hot_projects.tools.registry import ToolSpec, ToolRegistry


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


def test_react_executes_tool_then_replies():
    reg = _registry_with("get_db_info", lambda ctx, args: {"total_projects": 1})
    llm = FakeLLM([_toolcall("get_db_info"), _text("数据库共有 1 个项目。")])
    agent = HotProjectAgent(llm=llm, registry=reg, provider=None, db={"projects": {}})
    reply = agent.chat("数据库里有多少项目")
    assert "1" in reply
    assert llm.calls == 2


def test_direct_text_reply_no_tool():
    reg = _registry_with("get_db_info", lambda ctx, args: {})
    llm = FakeLLM([_text("你好，我可以帮你查 GitHub 热门项目。")])
    agent = HotProjectAgent(llm=llm, registry=reg, provider=None, db={"projects": {}})
    reply = agent.chat("你好")
    assert "热门" in reply
    assert llm.calls == 1


def test_llm_failure_returns_message():
    reg = _registry_with("get_db_info", lambda ctx, args: {})
    llm = FakeLLM([None])
    agent = HotProjectAgent(llm=llm, registry=reg, provider=None, db={"projects": {}})
    reply = agent.chat("hi")
    assert "失败" in reply


def test_bad_tool_args_recorded_then_replies():
    # get_db_info 参数非法 JSON → 工具返回错误 → LLM 下一轮给出文本
    reg = _registry_with("get_db_info", lambda ctx, args: {"total_projects": 0})
    llm = FakeLLM([_toolcall("get_db_info", "{bad json"), _text("已处理。")])
    agent = HotProjectAgent(llm=llm, registry=reg, provider=None, db={"projects": {}})
    reply = agent.chat("查询")
    assert reply == "已处理。"
