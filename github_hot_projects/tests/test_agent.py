"""
测试 agent 模块
================
覆盖：Agent 主状态机、提示词识别、对话压缩、状态缓存与异常兜底。
"""

from unittest.mock import MagicMock, patch


def _tool_call(name: str, arguments: str = "{}", tool_call_id: str = "call-1") -> dict:
    return {
        "id": tool_call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


class TestAgentPromptRecognition:
    def test_new_project_workflow_detects_explicit_phrase(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        agent.state.conversation.append({"role": "user", "content": "给我新项目热榜前20"})

        assert agent._is_new_project_workflow_request() is True

    def test_new_project_workflow_detects_semantic_combo(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        agent.state.conversation.append({"role": "user", "content": "看看最近新项目里比较火的有哪些"})

        assert agent._is_new_project_workflow_request() is True

    def test_new_project_workflow_ignores_non_ranking_prompt(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        agent.state.conversation.append({"role": "user", "content": "帮我写一个新项目 README 模板"})

        assert agent._is_new_project_workflow_request() is False

    def test_realtime_refresh_detects_latest_hot_rank(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        agent.state.conversation.append({"role": "user", "content": "给我最新 GitHub 热榜前 50"})

        assert agent._is_realtime_refresh_request() is True

    def test_realtime_refresh_ignores_recent_but_not_realtime(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        agent.state.conversation.append({"role": "user", "content": "看看近期 GitHub 热门项目"})

        assert agent._is_realtime_refresh_request() is False

    def test_resolve_new_project_days_uses_workflow_default(self):
        from github_hot_projects.agent import HotProjectAgent, NEW_PROJECT_DAYS

        agent = HotProjectAgent()
        agent.state.conversation.append({"role": "user", "content": "给我新项目热榜前20"})

        assert agent._resolve_new_project_days("search_hot_projects", {}) == NEW_PROJECT_DAYS


class TestAgentStateMachine:
    def test_chat_returns_direct_llm_reply(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        with patch.object(
            agent,
            "_call_llm",
            return_value={"choices": [{"message": {"content": "直接回复", "tool_calls": []}}]},
        ):
            reply = agent.chat("你好")

        assert reply == "直接回复"
        assert agent.state.conversation[-1] == {"role": "assistant", "content": "直接回复"}

    def test_chat_executes_tool_then_returns_final_reply(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        llm_responses = [
            {
                "choices": [{
                    "message": {
                        "content": "先查一下",
                        "tool_calls": [
                            _tool_call("get_db_info", '{"repo": "org/repo"}', "tool-1")
                        ],
                    }
                }]
            },
            {"choices": [{"message": {"content": "这是最终结果", "tool_calls": []}}]},
        ]

        with patch.object(agent, "_call_llm", side_effect=llm_responses):
            with patch.object(agent, "_execute_tool", return_value={"found": True, "info": {"star": 1}}) as mock_exec:
                reply = agent.chat("查一下 org/repo")

        assert reply == "这是最终结果"
        mock_exec.assert_called_once_with("get_db_info", {"repo": "org/repo"})
        assert any(msg.get("role") == "tool" for msg in agent.state.conversation)

    def test_chat_returns_tool_error_on_execute_exception(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        llm_responses = [
            {
                "choices": [{
                    "message": {
                        "content": "执行工具",
                        "tool_calls": [_tool_call("get_db_info", "{}", "tool-1")],
                    }
                }]
            },
            {"choices": [{"message": {"content": "已处理错误", "tool_calls": []}}]},
        ]

        with patch.object(agent, "_call_llm", side_effect=llm_responses):
            with patch.object(agent, "_execute_tool", side_effect=RuntimeError("boom")):
                reply = agent.chat("测一下异常")

        assert reply == "已处理错误"
        tool_messages = [msg for msg in agent.state.conversation if msg.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert "工具执行异常" in tool_messages[0]["content"]

    def test_chat_tolerates_invalid_tool_arguments_json(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        llm_responses = [
            {
                "choices": [{
                    "message": {
                        "content": "执行工具",
                        "tool_calls": [_tool_call("get_db_info", "{not-json}", "tool-1")],
                    }
                }]
            },
            {"choices": [{"message": {"content": "已处理", "tool_calls": []}}]},
        ]

        with patch.object(agent, "_call_llm", side_effect=llm_responses):
            with patch.object(agent, "_execute_tool", return_value={"ok": True}) as mock_exec:
                reply = agent.chat("测一下非法参数")

        assert reply == "已处理"
        mock_exec.assert_called_once_with("get_db_info", {})

    def test_chat_returns_guard_message_when_tool_loop_exceeds_limit(self):
        from github_hot_projects.agent import HotProjectAgent, MAX_TOOL_CALLS_PER_TURN

        agent = HotProjectAgent()
        looping_response = {
            "choices": [{
                "message": {
                    "content": "继续调用",
                    "tool_calls": [_tool_call("get_db_info", "{}")],
                }
            }]
        }

        with patch.object(agent, "_call_llm", side_effect=[looping_response] * MAX_TOOL_CALLS_PER_TURN):
            with patch.object(agent, "_execute_tool", return_value={"ok": True}):
                reply = agent.chat("无限循环")

        assert "已达到单轮最大 Tool 调用次数" in reply

    def test_execute_tool_search_caches_results(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        raw_repos = [{"full_name": "org/repo", "star": 123}]

        with patch("github_hot_projects.agent.tool_search_hot_projects", return_value={"repos": [], "total": 1, "_raw_repos": raw_repos}):
            result = agent._execute_tool("search_hot_projects", {})

        assert result["total"] == 1
        assert agent.state.last_search_repos == raw_repos
        assert "org/repo" in agent.state.seen_repos

    def test_execute_tool_returns_error_for_unknown_tool(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()

        result = agent._execute_tool("unknown_tool", {})

        assert "未知 Tool" in result["error"]

    def test_execute_tool_batch_check_growth_uses_force_refresh_intent(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        agent.state.last_search_repos = [{"full_name": "org/repo", "_raw": {}}]
        agent.state.conversation.append({"role": "user", "content": "给我最新 GitHub 热榜，强制刷新"})

        with patch("github_hot_projects.agent.tool_batch_check_growth", return_value={"candidates": {}, "total": 0}) as mock_batch:
            agent._execute_tool("batch_check_growth", {})

        assert mock_batch.call_args.kwargs["force_refresh"] is True

    def test_execute_tool_batch_check_growth_requires_search_results(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()

        result = agent._execute_tool("batch_check_growth", {})

        assert "没有搜索结果" in result["error"]


class TestAgentStateHelpers:
    def test_serialize_result_truncates_large_lists(self):
        from github_hot_projects.agent import HotProjectAgent

        payload = {"repos": [{"repo": f"org/repo-{i}"} for i in range(100)]}
        result = HotProjectAgent._serialize_result(payload, max_len=400)

        assert "_repos_note" in result
        assert "结果已截断" in result or "已截取前" in result

    def test_compress_conversation_preserves_recent_messages_and_adds_summary(self):
        from github_hot_projects.agent import HotProjectAgent, KEEP_RECENT_MESSAGES, MAX_CONVERSATION_MESSAGES

        agent = HotProjectAgent()
        for idx in range(MAX_CONVERSATION_MESSAGES + 5):
            agent.state.conversation.append({"role": "user", "content": f"用户消息 {idx}"})
            agent.state.conversation.append({"role": "assistant", "content": f"助手回复 {idx}"})

        agent._compress_conversation()

        system_messages = [msg for msg in agent.state.conversation if msg.get("role") == "system"]
        assert len(system_messages) >= 2
        assert any("对话历史摘要" in (msg.get("content") or "") for msg in system_messages)
        recent_contents = [msg.get("content") for msg in agent.state.conversation[-KEEP_RECENT_MESSAGES:]]
        assert any("助手回复" in (content or "") for content in recent_contents)
