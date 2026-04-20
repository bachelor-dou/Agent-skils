"""
测试 agent 模块
================
覆盖：Agent 状态机、Tool 路由、参数校验、对话压缩、状态缓存与异常兜底。
"""

from unittest.mock import patch


def _tool_call(name: str, arguments: str = "{}", tool_call_id: str = "call-1") -> dict:
    return {
        "id": tool_call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


class TestArgValidation:
    """测试 validate_tool_args 通过 _execute_tool 的实际效果。"""

    def test_search_preserves_llm_new_project_days(self):
        """LLM 传了 new_project_days=20，不应被剥离。"""
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()

        with patch(
            "github_hot_projects.agent.tool_search_hot_projects",
            return_value={"repos": [], "total": 0, "_raw_repos": []},
        ) as mock_search:
            agent._execute_tool("search_hot_projects", {"new_project_days": 20})

        assert mock_search.call_args.kwargs["new_project_days"] == 20

    def test_search_defaults_when_no_new_project_days(self):
        """LLM 没传 new_project_days，默认不过滤。"""
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()

        with patch(
            "github_hot_projects.agent.tool_search_hot_projects",
            return_value={"repos": [], "total": 0, "_raw_repos": []},
        ) as mock_search:
            agent._execute_tool("search_hot_projects", {})

        assert mock_search.call_args.kwargs["new_project_days"] is None

    def test_batch_check_preserves_both_windows(self):
        """LLM 同时传 time_window_days 和 new_project_days，都应保留。"""
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        agent.state.last_search_repos = [{"full_name": "org/repo", "_raw": {}}]

        with patch(
            "github_hot_projects.agent.tool_batch_check_growth",
            return_value={"candidates": {}, "total": 0},
        ) as mock_batch:
            with patch("github_hot_projects.agent.save_db"):
                agent._execute_tool("batch_check_growth", {
                    "time_window_days": 10,
                    "new_project_days": 20,
                })

        assert mock_batch.call_args.kwargs["time_window_days"] == 10
        assert mock_batch.call_args.kwargs["new_project_days"] == 20

    def test_rank_candidates_preserves_llm_mode_and_top_n(self):
        """LLM 传了 mode=hot_new + top_n=10，都应保留。"""
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        agent.state.last_candidates = {"org/repo": {"growth": 100, "star": 1000}}

        with patch(
            "github_hot_projects.agent.tool_rank_candidates",
            return_value={"ranked_projects": [], "_ordered_tuples": [], "total_candidates": 1, "returned": 0, "mode": "hot_new"},
        ) as mock_rank:
            agent._execute_tool("rank_candidates", {"mode": "hot_new", "top_n": 10, "new_project_days": 20})

        assert mock_rank.call_args.kwargs["mode"] == "hot_new"
        assert mock_rank.call_args.kwargs["top_n"] == 10
        assert mock_rank.call_args.kwargs["new_project_days"] == 20

    def test_rank_candidates_coerces_invalid_values(self):
        """无效参数应被纠正为默认值。"""
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        agent.state.last_candidates = {"org/repo": {"growth": 100, "star": 1000}}

        with patch(
            "github_hot_projects.agent.tool_rank_candidates",
            return_value={"ranked_projects": [], "_ordered_tuples": [], "total_candidates": 1, "returned": 0, "mode": "comprehensive"},
        ) as mock_rank:
            agent._execute_tool("rank_candidates", {"top_n": -1, "mode": "bad-mode"})

        assert mock_rank.call_args.kwargs["top_n"] == 1  # clamped to min
        assert mock_rank.call_args.kwargs["mode"] == "comprehensive"

    def test_search_coerces_invalid_project_min_star(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()

        with patch(
            "github_hot_projects.agent.tool_search_hot_projects",
            return_value={"repos": [], "total": 0, "_raw_repos": []},
        ) as mock_search:
            agent._execute_tool("search_hot_projects", {"project_min_star": -5, "max_pages": 0})

        assert mock_search.call_args.kwargs["project_min_star"] >= 1
        assert mock_search.call_args.kwargs["max_pages"] >= 1

    def test_fetch_trending_strips_unknown_params(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()

        with patch(
            "github_hot_projects.agent.tool_fetch_trending",
            return_value={"repos": [], "count": 0, "_raw_repos": []},
        ) as mock_trending:
            agent._execute_tool(
                "fetch_trending",
                {"since": "monthly", "language": "python", "spoken_language": "zh"},
            )

        assert mock_trending.call_args.kwargs == {
            "since": "monthly",
            "include_all_periods": False,
        }

    def test_rank_defaults_top_n_by_mode(self):
        from github_hot_projects.agent import HotProjectAgent, HOT_NEW_PROJECT_COUNT, NEW_PROJECT_DAYS

        agent = HotProjectAgent()
        agent.state.last_candidates = {"org/repo": {"growth": 100, "star": 1000}}

        with patch(
            "github_hot_projects.agent.tool_rank_candidates",
            return_value={"ranked_projects": [], "_ordered_tuples": [], "total_candidates": 1, "returned": 0, "mode": "hot_new"},
        ) as mock_rank:
            agent._execute_tool("rank_candidates", {"mode": "hot_new"})

        assert mock_rank.call_args.kwargs["top_n"] == HOT_NEW_PROJECT_COUNT
        assert mock_rank.call_args.kwargs["new_project_days"] == NEW_PROJECT_DAYS

    def test_batch_check_growth_passes_force_refresh(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        agent.state.last_search_repos = [{"full_name": "org/repo", "_raw": {}}]

        with patch("github_hot_projects.agent.tool_batch_check_growth", return_value={"candidates": {}, "total": 0}) as mock_batch:
            with patch("github_hot_projects.agent.save_db"):
                agent._execute_tool("batch_check_growth", {"force_refresh": True})

        assert mock_batch.call_args.kwargs["force_refresh"] is True


class TestAgentStateMachine:
    def test_chat_returns_direct_llm_reply(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        with patch.object(agent, "_maybe_handle_confirmation_gate", return_value=(None, False)):
            with patch.object(
                agent,
                "_call_llm",
                return_value={"choices": [{"message": {"content": "直接回复", "tool_calls": []}}]},
            ):
                reply = agent.chat("你好")

        assert reply == "直接回复"
        assert agent.state.conversation[-1] == {"role": "assistant", "content": "直接回复"}

    def test_chat_requests_confirmation_before_tool_execution(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()

        with patch.object(agent, "_build_confirmation_message", return_value="收到！我理解为：综合热榜，统计近10天增长，返回前100名。确认请回复“开始”，或直接告诉我需要修改的地方。") as mock_confirm:
            with patch.object(agent, "_call_llm") as mock_llm:
                reply = agent.chat("给我实时综合热榜前100名，窗口为10天")

        assert "确认请回复“开始”" in reply
        mock_confirm.assert_called_once()
        mock_llm.assert_not_called()
        assert agent.state.awaiting_confirmation is True

    def test_chat_executes_after_confirmation_ack(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()

        with patch.object(agent, "_build_confirmation_message", return_value="收到！我理解为：综合热榜，统计近10天增长，返回前100名。确认请回复“开始”，或直接告诉我需要修改的地方。"):
            first_reply = agent.chat("给我实时综合热榜前100名，窗口为10天")

        assert "确认请回复“开始”" in first_reply

        with patch.object(
            agent,
            "_call_llm",
            return_value={"choices": [{"message": {"content": "这是最终结果", "tool_calls": []}}]},
        ) as mock_llm:
            reply = agent.chat("开始")

        assert reply == "这是最终结果"
        assert agent.state.awaiting_confirmation is False
        assert mock_llm.call_args.kwargs["execution_confirmed"] is True

    def test_chat_reconfirms_when_user_modifies_unconfirmed_request(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()

        with patch.object(agent, "_build_confirmation_message", side_effect=[
            "收到！我理解为：综合热榜，统计近10天增长，返回前100名。确认请回复“开始”，或直接告诉我需要修改的地方。",
            "收到！我理解为：综合热榜，统计近10天增长，返回前50名。确认请回复“开始”，或直接告诉我需要修改的地方。",
        ]) as mock_confirm:
            first_reply = agent.chat("给我实时综合热榜前100名，窗口为10天")
            second_reply = agent.chat("改成前50名")

        assert "前100名" in first_reply
        assert "前50名" in second_reply
        assert mock_confirm.call_count == 2
        assert agent.state.awaiting_confirmation is True

    def test_chat_returns_scoped_capability_reply_for_greeting(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()

        with patch.object(agent, "_call_llm") as mock_llm:
            reply = agent.chat("你好")

        assert "GitHub 热门项目助手" in reply
        assert "编程问题" not in reply
        mock_llm.assert_not_called()
        assert agent.state.awaiting_confirmation is False

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

        with patch.object(agent, "_maybe_handle_confirmation_gate", return_value=(None, False)):
            with patch.object(agent, "_call_llm", side_effect=llm_responses):
                with patch.object(agent, "_execute_tool", return_value={"found": True, "info": {"star": 1}}) as mock_exec:
                    reply = agent.chat("查一下 org/repo")

        assert reply == "这是最终结果"
        mock_exec.assert_called_once_with("get_db_info", {"repo": "org/repo"})

    def test_chat_returns_tool_error_on_execute_exception(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        llm_responses = [
            {"choices": [{"message": {"content": "执行", "tool_calls": [_tool_call("get_db_info", "{}", "tool-1")]}}]},
            {"choices": [{"message": {"content": "已处理错误", "tool_calls": []}}]},
        ]

        with patch.object(agent, "_maybe_handle_confirmation_gate", return_value=(None, False)):
            with patch.object(agent, "_call_llm", side_effect=llm_responses):
                with patch.object(agent, "_execute_tool", side_effect=RuntimeError("boom")):
                    reply = agent.chat("测一下异常")

        assert reply == "已处理错误"
        tool_messages = [msg for msg in agent.state.conversation if msg.get("role") == "tool"]
        assert "工具执行异常" in tool_messages[0]["content"]

    def test_chat_returns_tool_error_when_tool_arguments_json_invalid(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        llm_responses = [
            {"choices": [{"message": {"content": "执行", "tool_calls": [_tool_call("get_db_info", "{not-json}", "tool-1")]}}]},
            {"choices": [{"message": {"content": "已处理", "tool_calls": []}}]},
        ]

        with patch.object(agent, "_maybe_handle_confirmation_gate", return_value=(None, False)):
            with patch.object(agent, "_call_llm", side_effect=llm_responses):
                with patch.object(agent, "_execute_tool", return_value={"ok": True}) as mock_exec:
                    reply = agent.chat("测一下非法参数")

        mock_exec.assert_not_called()
        tool_messages = [msg for msg in agent.state.conversation if msg.get("role") == "tool"]
        assert "Tool arguments JSON 解析失败" in tool_messages[0]["content"]

    def test_chat_returns_guard_message_when_tool_loop_exceeds_limit(self):
        from github_hot_projects.agent import HotProjectAgent, MAX_TOOL_CALLS_PER_TURN

        agent = HotProjectAgent()
        looping_response = {
            "choices": [{"message": {"content": "继续", "tool_calls": [_tool_call("get_db_info", "{}")]}}]
        }

        with patch.object(agent, "_maybe_handle_confirmation_gate", return_value=(None, False)):
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
        agent.state.last_ranked = [("old/repo", {"growth": 1, "star": 1})]
        agent.state.seen_repos = {"old/repo"}
        raw_repos = [{"full_name": "new/repo", "star": 123}]

        with patch(
            "github_hot_projects.agent.tool_search_hot_projects",
            return_value={"repos": [], "total": 1, "_raw_repos": raw_repos},
        ):
            agent._execute_tool("search_hot_projects", {})

        assert agent.state.last_search_repos == raw_repos
        assert agent.state.last_candidates == {}
        assert agent.state.last_ranked == []
        assert agent.state.last_mode == "comprehensive"
        assert agent.state.last_time_window_days == TIME_WINDOW_DAYS
        assert agent.state.seen_repos == {"new/repo"}

    def test_execute_tool_batch_check_growth_requires_search_results(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        result = agent._execute_tool("batch_check_growth", {})
        assert "没有搜索结果" in result["error"]

    def test_execute_tool_returns_error_for_unknown_tool(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        result = agent._execute_tool("unknown_tool", {})
        assert "未知 Tool" in result["error"]


class TestAgentStateHelpers:
    def test_serialize_result_truncates_large_lists(self):
        from github_hot_projects.agent import HotProjectAgent

        payload = {"repos": [{"repo": f"org/repo-{i}"} for i in range(100)]}
        result = HotProjectAgent._serialize_result(payload, max_len=400)

        assert "_repos_note" in result
        assert "结果已截断" in result or "已截取前" in result

    def test_build_confirmation_message_disables_thinking(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        agent.state.conversation.append({"role": "user", "content": "给我实时综合热榜前一百名，增长窗口10天"})

        with patch.object(
            agent,
            "_request_llm",
            return_value={"choices": [{"message": {"content": "收到！我理解为：综合热榜，统计近10天的增长，返回前100名。确认请回复“开始”，或直接告诉我需要修改的地方。"}}]},
        ) as mock_request:
            reply = agent._build_confirmation_message()

        assert "确认请回复“开始”" in reply
        assert mock_request.call_args.kwargs["enable_thinking"] is False
        assert mock_request.call_args.kwargs["max_tokens"] == 256

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
        assert len(system_messages) == 1
        assert "对话历史摘要" in (system_messages[0].get("content") or "")
        recent_contents = [msg.get("content") for msg in agent.state.conversation[-KEEP_RECENT_MESSAGES:]]
        assert any("助手回复" in (content or "") for content in recent_contents)


class TestValidateToolArgs:
    """直接测试 validate_tool_args 函数。"""

    def test_preserves_llm_new_project_days(self):
        from github_hot_projects.parsing.arg_validator import validate_tool_args

        result = validate_tool_args("search_hot_projects", {"new_project_days": 20})
        assert result["new_project_days"] == 20

    def test_defaults_missing_params(self):
        from github_hot_projects.parsing.arg_validator import validate_tool_args
        from github_hot_projects.common.config import MIN_STAR_FILTER

        result = validate_tool_args("search_hot_projects", {})
        assert result["project_min_star"] == MIN_STAR_FILTER
        assert result["max_pages"] == 3
        assert "new_project_days" not in result

    def test_coerces_out_of_range_int(self):
        from github_hot_projects.parsing.arg_validator import validate_tool_args

        result = validate_tool_args("rank_candidates", {"top_n": 500})
        assert result["top_n"] == 200

    def test_coerces_invalid_enum(self):
        from github_hot_projects.parsing.arg_validator import validate_tool_args

        result = validate_tool_args("rank_candidates", {"mode": "invalid"})
        assert result["mode"] == "comprehensive"

    def test_strips_unknown_params(self):
        from github_hot_projects.parsing.arg_validator import validate_tool_args

        result = validate_tool_args("fetch_trending", {"since": "daily", "language": "python"})
        assert "language" not in result
        assert result["since"] == "daily"

    def test_default_by_mode_hot_new(self):
        from github_hot_projects.parsing.arg_validator import validate_tool_args
        from github_hot_projects.common.config import HOT_NEW_PROJECT_COUNT, NEW_PROJECT_DAYS

        result = validate_tool_args("rank_candidates", {"mode": "hot_new"})
        assert result["top_n"] == HOT_NEW_PROJECT_COUNT
        assert result["new_project_days"] == NEW_PROJECT_DAYS

    def test_default_by_mode_comprehensive(self):
        from github_hot_projects.parsing.arg_validator import validate_tool_args
        from github_hot_projects.common.config import HOT_PROJECT_COUNT

        result = validate_tool_args("rank_candidates", {"mode": "comprehensive"})
        assert result["top_n"] == HOT_PROJECT_COUNT
        assert "new_project_days" not in result

    def test_bool_coercion(self):
        from github_hot_projects.parsing.arg_validator import validate_tool_args

        result = validate_tool_args("batch_check_growth", {"force_refresh": 1})
        assert result["force_refresh"] is True

    def test_preserves_required_str(self):
        from github_hot_projects.parsing.arg_validator import validate_tool_args

        result = validate_tool_args("check_repo_growth", {"repo": "org/repo"})
        assert result["repo"] == "org/repo"
