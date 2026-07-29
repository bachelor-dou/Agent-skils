from hot_projects.agent.agent import HotProjectAgent
from hot_projects.agent.state import AgentState
from hot_projects.tools.registry import ToolSpec, ToolRegistry


def test_agent_state_defaults():
    s = AgentState(db={"projects": {}})
    assert isinstance(s.tool_state, dict)
    assert s.active_repo is None
    assert s.pending_confirmation_signature is None
    assert isinstance(s.conversation, list)


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


def test_needs_confirmation_short_circuits():
    # 工具返回 needs_confirmation → agent 直接把 message 回显给用户，不再走一轮 LLM 转述
    msg = "将执行【关键词热榜】，参数：Top 10；最低 star=1；增长阈值=0（不过滤增长）。确认无误请回复『开始』。"
    reg = _registry_with("keyword_ranking",
                         lambda ctx, args: {"needs_confirmation": True, "message": msg})
    llm = FakeLLM([_toolcall("keyword_ranking")])
    agent = HotProjectAgent(llm=llm, registry=reg, provider=None, db={"projects": {}})
    reply = agent.chat("跑个关键词榜")
    assert reply == msg
    assert llm.calls == 1  # 短路：没有第二轮 LLM


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


def test_compress_keeps_tool_pairing():
    """压缩后保留的历史不能以孤儿 tool 消息开头（否则 OpenAI 兼容接口报 400）。"""
    from hot_projects.agent.state import MAX_CONVERSATION_MESSAGES

    reg = _registry_with("get_db_info", lambda ctx, args: {})
    # 第 1 个响应给 _summarize（lite 摘要），第 2 个给正常对话
    llm = FakeLLM([_text("历史摘要"), _text("好的。")])
    agent = HotProjectAgent(llm=llm, registry=reg, provider=None, db={"projects": {}})

    # 构造超长历史：结尾是「assistant(tool_calls) + 大量 tool 结果」，
    # 使朴素的 [-KEEP_RECENT:] 切片必然切在配对中间。
    conv = agent.state.conversation  # [system]
    for i in range(MAX_CONVERSATION_MESSAGES - 8):
        conv.append({"role": "user", "content": f"问题{i}"})
        conv.append({"role": "assistant", "content": f"回答{i}"})
    conv.append({"role": "user", "content": "跑一批工具"})
    conv.append({"role": "assistant", "content": None, "tool_calls": [
        {"id": str(i), "type": "function",
         "function": {"name": "get_db_info", "arguments": "{}"}}
        for i in range(12)
    ]})
    for i in range(12):
        conv.append({"role": "tool", "tool_call_id": str(i), "content": "{}"})

    reply = agent.chat("继续")
    assert reply == "好的。"

    rebuilt = agent.state.conversation
    assert rebuilt[0]["role"] == "system"
    # 找到摘要之后的第一条历史消息，不能是 tool
    first_kept = next(
        m for m in rebuilt[1:]
        if not (m.get("role") == "user" and str(m.get("content", "")).startswith("[对话历史摘要]"))
    )
    assert first_kept.get("role") != "tool"
    # 且全历史中每条 tool 消息前必须能追溯到带 tool_calls 的 assistant
    for idx, m in enumerate(rebuilt):
        if m.get("role") == "tool":
            prev = rebuilt[idx - 1]
            assert prev.get("role") == "tool" or prev.get("tool_calls")


def test_step_cap_finalizes_without_tools():
    """命中步数护栏后：不再给工具，额外做一次无工具收口，返回其正文。"""
    from hot_projects.agent.agent import MAX_AGENT_STEPS

    reg = _registry_with("get_db_info", lambda ctx, args: {"total_projects": 1})
    # 前 MAX_AGENT_STEPS 步一直返回 tool_calls（模拟不收敛），最后一次收口返回文本
    responses = [_toolcall("get_db_info")] * MAX_AGENT_STEPS + [_text("已尽力收口。")]
    llm = FakeLLM(responses)
    agent = HotProjectAgent(llm=llm, registry=reg, provider=None, db={"projects": {}})
    reply = agent.chat("反复调用")
    assert reply == "已尽力收口。"
    assert llm.calls == MAX_AGENT_STEPS + 1  # 15 步循环 + 1 次无工具收口


def test_bad_tool_args_recorded_then_replies():
    # get_db_info 参数非法 JSON → 工具返回错误 → LLM 下一轮给出文本
    reg = _registry_with("get_db_info", lambda ctx, args: {"total_projects": 0})
    llm = FakeLLM([_toolcall("get_db_info", "{bad json"), _text("已处理。")])
    agent = HotProjectAgent(llm=llm, registry=reg, provider=None, db={"projects": {}})
    reply = agent.chat("查询")
    assert reply == "已处理。"
