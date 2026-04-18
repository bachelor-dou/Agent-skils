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


def _render_log_messages(mock_logger) -> list[str]:
    messages: list[str] = []
    for call in mock_logger.call_args_list:
        if not call.args:
            continue
        template = call.args[0]
        if len(call.args) > 1:
            messages.append(template % call.args[1:])
        else:
            messages.append(template)
    return messages


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

    def test_extract_requested_time_window_days_from_prompt(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        agent.state.conversation.append({"role": "user", "content": "给我近10天github综合热榜的前130名"})

        assert agent._resolve_time_window_days("batch_check_growth", {}) == 10

    def test_resolve_new_project_days_uses_creation_window_only_when_explicit(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        agent.state.conversation.append({"role": "user", "content": "给我30天内创建的新项目热榜前20"})

        assert agent._resolve_new_project_days("search_hot_projects", {}) == 30

    def test_new_project_prompt_uses_time_window_but_keeps_default_creation_window(self):
        from github_hot_projects.agent import HotProjectAgent, NEW_PROJECT_DAYS

        agent = HotProjectAgent()
        agent.state.conversation.append({"role": "user", "content": "给我近10天新项目热榜前20"})

        assert agent._resolve_time_window_days("batch_check_growth", {}) == 10
        assert agent._resolve_new_project_days("batch_check_growth", {}) == NEW_PROJECT_DAYS

    def test_comprehensive_rank_request_with_time_window_is_not_new_project(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        agent.state.conversation.append({"role": "user", "content": "给我近10天github综合热榜的前130名"})

        assert agent._is_new_project_workflow_request() is False
        assert agent._is_explicit_comprehensive_ranking_request() is True


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

    def test_chat_returns_tool_error_when_tool_arguments_json_invalid(self):
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
        mock_exec.assert_not_called()
        tool_messages = [msg for msg in agent.state.conversation if msg.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert "Tool arguments JSON 解析失败" in tool_messages[0]["content"]

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

    def test_execute_tool_search_resets_discovery_state_on_new_turn(self):
        from github_hot_projects.agent import HotProjectAgent, TIME_WINDOW_DAYS

        agent = HotProjectAgent()
        agent.state.current_user_turn = 2
        agent.state.discovery_turn_id = 1
        agent.state.last_search_repos = [{"full_name": "old/repo", "star": 1}]
        agent.state.last_candidates = {"old/repo": {"growth": 1, "star": 1}}
        agent.state.last_candidate_new_project_days = 45
        agent.state.last_ranked = [("old/repo", {"growth": 1, "star": 1})]
        agent.state.last_mode = "hot_new"
        agent.state.last_time_window_days = 30
        agent.state.seen_repos = {"old/repo"}
        raw_repos = [{"full_name": "new/repo", "star": 123}]

        with patch(
            "github_hot_projects.agent.tool_search_hot_projects",
            return_value={"repos": [], "total": 1, "_raw_repos": raw_repos},
        ):
            agent._execute_tool("search_hot_projects", {})

        assert agent.state.last_search_repos == raw_repos
        assert agent.state.last_candidates == {}
        assert agent.state.last_candidate_new_project_days is None
        assert agent.state.last_ranked == []
        assert agent.state.last_mode == "comprehensive"
        assert agent.state.last_time_window_days == TIME_WINDOW_DAYS
        assert agent.state.seen_repos == {"new/repo"}
        assert agent.state.discovery_turn_id == 2

    def test_execute_tool_search_logs_default_effective_params(self):
        from github_hot_projects.agent import HotProjectAgent, STAR_RANGE_MIN

        agent = HotProjectAgent()

        with patch(
            "github_hot_projects.agent.tool_search_hot_projects",
            return_value={"repos": [], "total": 0, "_raw_repos": []},
        ) as mock_search:
            with patch("github_hot_projects.agent.logger.info") as mock_log:
                agent._execute_tool("search_hot_projects", {})

        assert mock_search.call_args.kwargs["min_stars"] == STAR_RANGE_MIN
        messages = _render_log_messages(mock_log)
        assert any("[Agent] Tool 生效参数: search_hot_projects" in message for message in messages)
        assert any("workflow_mode=comprehensive(default)" in message for message in messages)
        assert any(f"min_stars={STAR_RANGE_MIN}(default)" in message for message in messages)
        assert any("creation_window_days=None(unused)" in message for message in messages)

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

    def test_execute_tool_batch_check_growth_resolves_time_window_for_comprehensive_request(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        agent.state.last_search_repos = [{"full_name": "org/repo", "_raw": {}}]
        agent.state.conversation.append({"role": "user", "content": "给我近10天github综合热榜的前130名"})

        with patch("github_hot_projects.agent.tool_batch_check_growth", return_value={"candidates": {}, "total": 0}) as mock_batch:
            agent._execute_tool("batch_check_growth", {"new_project_days": 10})

        assert mock_batch.call_args.kwargs["new_project_days"] is None
        assert mock_batch.call_args.kwargs["time_window_days"] == 10

    def test_execute_tool_batch_check_growth_logs_prompt_window_and_default_creation_window(self):
        from github_hot_projects.agent import HotProjectAgent, NEW_PROJECT_DAYS

        agent = HotProjectAgent()
        agent.state.last_search_repos = [{"full_name": "org/repo", "_raw": {}}]
        agent.state.conversation.append({"role": "user", "content": "给我近10天新项目热榜前20"})

        with patch(
            "github_hot_projects.agent.tool_batch_check_growth",
            return_value={"candidates": {}, "total": 0},
        ) as mock_batch:
            with patch("github_hot_projects.agent.save_db"):
                with patch("github_hot_projects.agent.logger.info") as mock_log:
                    agent._execute_tool("batch_check_growth", {})

        assert mock_batch.call_args.kwargs["time_window_days"] == 10
        assert mock_batch.call_args.kwargs["new_project_days"] == NEW_PROJECT_DAYS
        messages = _render_log_messages(mock_log)
        assert any("[Agent] Tool 生效参数: batch_check_growth" in message for message in messages)
        assert any("workflow_mode=hot_new(prompt)" in message for message in messages)
        assert any("growth_window_days=10(prompt)" in message for message in messages)
        assert any(f"creation_window_days={NEW_PROJECT_DAYS}(default)" in message for message in messages)

    def test_execute_tool_search_strips_new_project_days_for_explicit_comprehensive_request(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        agent.state.conversation.append({"role": "user", "content": "给我近10天github综合热榜的前130名"})

        with patch(
            "github_hot_projects.agent.tool_search_hot_projects",
            return_value={"repos": [], "total": 0, "_raw_repos": []},
        ) as mock_search:
            agent._execute_tool("search_hot_projects", {"new_project_days": 10, "max_pages": 2})

        assert mock_search.call_args.kwargs["new_project_days"] is None

    def test_execute_tool_rank_candidates_overrides_hot_new_for_explicit_comprehensive_request(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        agent.state.last_candidates = {"org/repo": {"growth": 100, "star": 1000}}
        agent.state.conversation.append({"role": "user", "content": "给我近10天github综合热榜的前130名"})

        with patch(
            "github_hot_projects.agent.tool_rank_candidates",
            return_value={"ranked_projects": [], "_ordered_tuples": [], "total_candidates": 0, "returned": 0, "mode": "comprehensive"},
        ) as mock_rank:
            agent._execute_tool("rank_candidates", {"mode": "hot_new", "new_project_days": 10, "top_n": 130})

        assert mock_rank.call_args.kwargs["mode"] == "comprehensive"
        assert mock_rank.call_args.kwargs["new_project_days"] is None
        assert mock_rank.call_args.kwargs["top_n"] == 130

    def test_execute_tool_rank_candidates_keeps_hot_new_for_explicit_new_project_request(self):
        from github_hot_projects.agent import HotProjectAgent, NEW_PROJECT_DAYS

        agent = HotProjectAgent()
        agent.state.last_candidates = {"org/repo": {"growth": 100, "star": 1000}}
        agent.state.conversation.append({"role": "user", "content": "给我近10天新项目热榜前20"})

        with patch(
            "github_hot_projects.agent.tool_rank_candidates",
            return_value={"ranked_projects": [], "_ordered_tuples": [], "total_candidates": 0, "returned": 0, "mode": "hot_new"},
        ) as mock_rank:
            agent._execute_tool("rank_candidates", {"mode": "hot_new", "new_project_days": 10, "top_n": 20})

        assert mock_rank.call_args.kwargs["mode"] == "hot_new"
        assert mock_rank.call_args.kwargs["new_project_days"] == NEW_PROJECT_DAYS

    def test_execute_tool_rank_candidates_logs_default_top_n(self):
        from github_hot_projects.agent import HotProjectAgent, HOT_PROJECT_COUNT

        agent = HotProjectAgent()
        agent.state.last_candidates = {"org/repo": {"growth": 100, "star": 1000}}

        with patch(
            "github_hot_projects.agent.tool_rank_candidates",
            return_value={"ranked_projects": [], "_ordered_tuples": [], "total_candidates": 1, "returned": 0, "mode": "comprehensive"},
        ) as mock_rank:
            with patch("github_hot_projects.agent.logger.info") as mock_log:
                agent._execute_tool("rank_candidates", {})

        assert mock_rank.call_args.kwargs["top_n"] == HOT_PROJECT_COUNT
        messages = _render_log_messages(mock_log)
        assert any("[Agent] Tool 生效参数: rank_candidates" in message for message in messages)
        assert any(f"top_n={HOT_PROJECT_COUNT}(default)" in message for message in messages)
        assert any("mode=comprehensive(default)" in message for message in messages)

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
        agent.state.last_search_repos = [{"full_name": "org/repo", "star": 1}]
        agent.state.last_candidates = {"org/repo": {"growth": 1, "star": 1}}
        agent.state.last_ranked = [("org/repo", {"growth": 1, "star": 1})]
        agent.state.last_mode = "hot_new"
        agent.state.last_time_window_days = 10
        agent.state.last_candidate_new_project_days = 45
        for idx in range(MAX_CONVERSATION_MESSAGES + 5):
            agent.state.conversation.append({"role": "user", "content": f"用户消息 {idx}"})
            agent.state.conversation.append({"role": "assistant", "content": f"助手回复 {idx}"})

        agent._compress_conversation()

        system_messages = [msg for msg in agent.state.conversation if msg.get("role") == "system"]
        assert len(system_messages) >= 2
        assert any("对话历史摘要" in (msg.get("content") or "") for msg in system_messages)
        assert any("榜单模式: hot_new" in (msg.get("content") or "") for msg in system_messages)
        assert any("增长窗口: 10 天" in (msg.get("content") or "") for msg in system_messages)
        assert any("创建窗口: 45" in (msg.get("content") or "") for msg in system_messages)
        recent_contents = [msg.get("content") for msg in agent.state.conversation[-KEEP_RECENT_MESSAGES:]]
        assert any("助手回复" in (content or "") for content in recent_contents)
