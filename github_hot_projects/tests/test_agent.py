"""
测试 agent 模块
================
覆盖：Agent 状态机、Tool 路由、参数校验、对话压缩、状态缓存与异常兜底。
"""

import json
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

    def test_search_preserves_llm_days_since_created(self):
        """LLM 传了 days_since_created=20，不应被剥离。"""
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()

        with patch(
            "github_hot_projects.agent.tool_search_by_keywords",
            return_value={"repos": [], "total": 0, "_raw_repos": []},
        ) as mock_search:
            agent._execute_tool("search_by_keywords", {"days_since_created": 20})

        assert mock_search.call_args.kwargs["days_since_created"] == 20

    def test_search_defaults_when_no_days_since_created(self):
        """LLM 没传 days_since_created，默认不过滤。"""
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()

        with patch(
            "github_hot_projects.agent.tool_search_by_keywords",
            return_value={"repos": [], "total": 0, "_raw_repos": []},
        ) as mock_search:
            agent._execute_tool("search_by_keywords", {})

        assert mock_search.call_args.kwargs["days_since_created"] is None

    def test_batch_check_preserves_both_windows(self):
        """LLM 同时传 growth_calc_days 和 days_since_created，都应保留。"""
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        agent.state.last_search_repos = [{"full_name": "org/repo", "_raw": {}}]

        with patch(
            "github_hot_projects.agent.tool_batch_check_growth",
            return_value={"candidates": {}, "total": 0},
        ) as mock_batch:
            agent._execute_tool("batch_check_growth", {
                "growth_calc_days": 10,
                "days_since_created": 20,
            })

        assert mock_batch.call_args.kwargs["growth_calc_days"] == 10
        assert mock_batch.call_args.kwargs["days_since_created"] == 20

    def test_rank_candidates_preserves_llm_mode_and_top_n(self):
        """LLM 传了 mode=hot_new + top_n=10，都应保留。"""
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        agent.state.last_candidates = {"org/repo": {"growth": 100, "star": 1000}}

        with patch(
            "github_hot_projects.agent.tool_rank_candidates",
            return_value={"ranked_projects": [], "_ordered_tuples": [], "total_candidates": 1, "returned": 0, "mode": "hot_new"},
        ) as mock_rank:
            agent._execute_tool("rank_candidates", {"mode": "hot_new", "top_n": 10, "days_since_created": 20})

        assert mock_rank.call_args.kwargs["mode"] == "hot_new"
        assert mock_rank.call_args.kwargs["top_n"] == 10
        assert mock_rank.call_args.kwargs["days_since_created"] == 20

    def test_rank_candidates_returns_retryable_error_for_invalid_values(self):
        """LLM 显式传错参数时，应返回可重试错误而不是静默纠偏。"""
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        agent.state.last_candidates = {"org/repo": {"growth": 100, "star": 1000}}

        with patch(
            "github_hot_projects.agent.tool_rank_candidates",
            return_value={"ranked_projects": [], "_ordered_tuples": [], "total_candidates": 1, "returned": 0, "mode": "comprehensive"},
        ) as mock_rank:
            result = agent._execute_tool("rank_candidates", {"top_n": -1, "mode": "bad-mode"})

        assert result["error_code"] == "invalid_arguments"
        assert result["retryable"] is True
        invalid_fields = {item["param"] for item in result["invalid_arguments"]}
        assert "top_n" in invalid_fields
        assert "mode" in invalid_fields
        mock_rank.assert_not_called()

    def test_search_returns_retryable_error_for_invalid_min_star(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()

        with patch(
            "github_hot_projects.agent.tool_search_by_keywords",
            return_value={"repos": [], "total": 0, "_raw_repos": []},
        ) as mock_search:
            result = agent._execute_tool("search_by_keywords", {"min_star": -5})

        assert result["error_code"] == "invalid_arguments"
        assert result["retryable"] is True
        invalid_fields = {item["param"] for item in result["invalid_arguments"]}
        assert "min_star" in invalid_fields
        mock_search.assert_not_called()

    def test_fetch_trending_validates_trending_range(self):
        """fetch_trending 应接受 trending_range 参数。"""
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()

        with patch(
            "github_hot_projects.agent.tool_fetch_trending",
            return_value={"repos": [], "count": 0, "_raw_repos": []},
        ) as mock_trending:
            agent._execute_tool(
                "fetch_trending",
                {"trending_range": "monthly"},
            )

        assert mock_trending.call_args.kwargs == {
            "trending_range": "monthly",
        }

    def test_rank_defaults_top_n_by_mode(self):
        from github_hot_projects.agent import HotProjectAgent, HOT_NEW_PROJECT_COUNT, DAYS_SINCE_CREATED

        agent = HotProjectAgent()
        agent.state.last_candidates = {"org/repo": {"growth": 100, "star": 1000}}

        with patch(
            "github_hot_projects.agent.tool_rank_candidates",
            return_value={"ranked_projects": [], "_ordered_tuples": [], "total_candidates": 1, "returned": 0, "mode": "hot_new"},
        ) as mock_rank:
            agent._execute_tool("rank_candidates", {"mode": "hot_new"})

        assert mock_rank.call_args.kwargs["top_n"] == HOT_NEW_PROJECT_COUNT
        assert mock_rank.call_args.kwargs["days_since_created"] == DAYS_SINCE_CREATED

    def test_batch_check_growth_passes_time_window(self):
        """batch_check_growth 应正确传递 growth_calc_days 参数。"""
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        agent.state.last_search_repos = [{"full_name": "org/repo", "_raw": {}}]

        with patch("github_hot_projects.agent.tool_batch_check_growth", return_value={"candidates": {}, "total": 0}) as mock_batch:
            agent._execute_tool("batch_check_growth", {"growth_calc_days": 10})

        assert mock_batch.call_args.kwargs["growth_calc_days"] == 10

    def test_log_validated_params_marks_system_injected_params(self):
        from github_hot_projects.parsing.arg_validator import log_validated_params

        with patch("github_hot_projects.parsing.arg_validator.logger.info") as mock_info:
            log_validated_params(
                "fetch_trending",
                {"since": "weekly"},
                {"since": "weekly", "include_all_periods": True},
                {"since": "weekly", "include_all_periods": True},
            )

        merged_logs = "\n".join(
            " ".join(str(arg) for arg in call.args)
            for call in mock_info.call_args_list
        )
        assert "since=weekly(llm)" in merged_logs
        assert "include_all_periods=True(system)" in merged_logs


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
        from github_hot_projects.agent import HotProjectAgent, PendingRequest

        agent = HotProjectAgent()

        with patch.object(
            agent,
            "_build_route_pending_request",
            return_value=PendingRequest(
                intent_family="comprehensive_ranking",
                intent_label_zh="综合热榜",
                user_specified_params={"growth_calc_days": 10, "top_n": 100},
                should_execute_now=True,
                source_turn_id=1,
            ),
        ) as mock_route:
            with patch.object(
                agent,
                "_call_llm",
                return_value={"choices": [{"message": {"content": "这是最终结果", "tool_calls": []}}]},
            ) as mock_llm:
                reply = agent.chat("给我实时综合热榜前100名，窗口为10天")

        assert reply == "这是最终结果"
        mock_route.assert_called_once()
        mock_llm.assert_called_once()
        assert agent.state.awaiting_confirmation is False
        assert mock_llm.call_args.kwargs["execution_confirmed"] is True

    def test_chat_executes_after_confirmation_ack(self):
        from github_hot_projects.agent import HotProjectAgent, PendingRequest

        agent = HotProjectAgent()

        with patch.object(
            agent,
            "_build_route_pending_request",
            side_effect=[
                PendingRequest(
                    intent_family="unknown",
                    intent_label_zh="未确定请求",
                    ambiguous_fields=["请确认这里的 10 天是增长窗口还是创建窗口"],
                    should_execute_now=False,
                    source_turn_id=1,
                ),
                PendingRequest(
                    intent_family="comprehensive_ranking",
                    intent_label_zh="综合热榜",
                    user_specified_params={"growth_calc_days": 10, "top_n": 100},
                    should_execute_now=True,
                    source_turn_id=2,
                ),
            ],
        ) as mock_route:
            first_reply = agent.chat("给我实时综合热榜前100名，窗口为10天")

            assert "需要确认" in first_reply or "请确认" in first_reply

            with patch.object(
                agent,
                "_call_llm",
                return_value={"choices": [{"message": {"content": "这是最终结果", "tool_calls": []}}]},
            ) as mock_llm:
                reply = agent.chat("10天指增长窗口")

        assert reply == "这是最终结果"
        assert agent.state.awaiting_confirmation is False
        assert mock_llm.call_args.kwargs["execution_confirmed"] is True
        assert mock_route.call_count == 2

    def test_chat_executes_after_confirmation_ack_with_shi(self):
        from github_hot_projects.agent import HotProjectAgent, PendingRequest

        agent = HotProjectAgent()
        agent.state.awaiting_confirmation = True
        agent.state.pending_request = PendingRequest(
            intent_family="trending_only",
            intent_label_zh="Trending 热门",
            should_execute_now=True,
            source_turn_id=1,
        )

        with patch.object(
            agent,
            "_call_llm",
            return_value={"choices": [{"message": {"content": "这是最终结果", "tool_calls": []}}]},
        ) as mock_llm:
            reply = agent.chat("是")

        assert reply == "这是最终结果"
        assert agent.state.awaiting_confirmation is False
        assert mock_llm.call_args.kwargs["execution_confirmed"] is True

    def test_confirmation_gate_accepts_short_ack(self):
        from github_hot_projects.agent import HotProjectAgent, PendingRequest

        agent = HotProjectAgent()
        agent.state.awaiting_confirmation = True
        agent.state.pending_request = PendingRequest(
            intent_family="hot_new_ranking",
            intent_label_zh="新项目热榜",
            should_execute_now=True,
            source_turn_id=1,
        )

        intercept, confirmed = agent._maybe_handle_confirmation_gate("开始")

        assert intercept is None
        assert confirmed is True
        assert agent.state.awaiting_confirmation is False

    def test_parse_pending_request_fact_check_enables_contract(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        payload = json.dumps(
            {
                "turn_kind": "fact_check",
                "intent_family": "repo_growth",
                "intent_label_zh": "单仓库增长",
                "target_repo": "org/repo",
                "specified_params": {"repo": "org/repo"},
                "ambiguous_fields": [],
                "should_execute_now": True,
                "must_call_tool_before_reply": False,
            },
            ensure_ascii=False,
        )

        pending = agent._parse_pending_request_content(payload)
        assert pending.turn_kind == "fact_check"
        assert pending.must_call_tool_before_reply is True
        assert pending.target_repo == "org/repo"

    def test_parse_pending_request_keeps_only_canonical_params(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        payload = json.dumps(
            {
                "turn_kind": "new_request",
                "intent_family": "hot_new_ranking",
                "intent_label_zh": "新项目热榜",
                "specified_params": {
                    "min_stars_gain": 500,
                    "limit": 5,
                    "days_since_created": 10,
                },
                "ambiguous_fields": [],
                "should_execute_now": True,
            },
            ensure_ascii=False,
        )

        pending = agent._parse_pending_request_content(payload)

        assert pending.user_specified_params == {"days_since_created": 10}
        assert pending.should_execute_now is False
        assert any("min_stars_gain" in item for item in pending.unresolved_constraints)
        assert any("limit" in item for item in pending.unresolved_constraints)

    def test_parse_pending_request_unresolved_constraints_block_execution(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        payload = json.dumps(
            {
                "turn_kind": "new_request",
                "intent_family": "comprehensive_ranking",
                "intent_label_zh": "综合热榜",
                "specified_params": {
                    "foo_signal": 123,
                    "growth_calc_days": 7,
                },
                "unresolved_constraints": ["用户提到热度系数，但未给计算口径"],
                "ambiguous_fields": [],
                "should_execute_now": True,
            },
            ensure_ascii=False,
        )

        pending = agent._parse_pending_request_content(payload)

        assert pending.user_specified_params == {"growth_calc_days": 7}
        assert pending.should_execute_now is False
        assert "用户提到热度系数，但未给计算口径" in pending.unresolved_constraints
        assert any("foo_signal" in item for item in pending.unresolved_constraints)
        assert "我还需要你确认这些参数含义" in pending.confirmation_text_zh
        assert "foo_signal" in pending.confirmation_text_zh

    def test_confirmation_gate_rebuilds_pending_when_user_modifies_request(self):
        from github_hot_projects.agent import HotProjectAgent, PendingRequest

        agent = HotProjectAgent()
        agent.state.awaiting_confirmation = True
        agent.state.pending_request = PendingRequest(
            intent_family="trending_only",
            intent_label_zh="Trending 热门",
            ambiguous_fields=["请确认是周榜还是日榜"],
            should_execute_now=False,
            source_turn_id=1,
        )

        with patch.object(
            agent,
            "_build_route_pending_request",
            return_value=PendingRequest(
                intent_family="trending_only",
                intent_label_zh="Trending 热门",
                ambiguous_fields=["请确认是 weekly 还是 daily"],
                should_execute_now=False,
                source_turn_id=2,
            ),
        ) as mock_route:
            intercept, confirmed = agent._maybe_handle_confirmation_gate("是的，并且再加上今天的")

        assert "需要确认" in intercept or "请确认" in intercept
        assert confirmed is False
        assert agent.state.awaiting_confirmation is True
        mock_route.assert_called_once()

    def test_chat_reconfirms_when_user_reply_is_long_modification(self):
        from github_hot_projects.agent import HotProjectAgent, PendingRequest

        agent = HotProjectAgent()

        with patch.object(
            agent,
            "_build_route_pending_request",
            side_effect=[
                PendingRequest(
                    intent_family="trending_only",
                    intent_label_zh="Trending 热门",
                    ambiguous_fields=["请确认需要 weekly 还是 all"],
                    should_execute_now=False,
                    source_turn_id=1,
                ),
                PendingRequest(
                    intent_family="trending_only",
                    intent_label_zh="Trending 热门",
                    user_specified_params={"trending_range": "all"},
                    should_execute_now=True,
                    source_turn_id=2,
                ),
            ],
        ) as mock_route:
            first_reply = agent.chat("看看本周 GitHub Trending")
            with patch.object(
                agent,
                "_call_llm",
                return_value={"choices": [{"message": {"content": "这是最终结果", "tool_calls": []}}]},
            ):
                second_reply = agent.chat("是的，并且再加上今天的")

        assert "需要确认" in first_reply or "请确认" in first_reply
        assert second_reply == "这是最终结果"
        assert mock_route.call_count == 2
        assert agent.state.awaiting_confirmation is False

    def test_chat_reconfirms_when_user_modifies_unconfirmed_request(self):
        from github_hot_projects.agent import HotProjectAgent, PendingRequest

        agent = HotProjectAgent()

        with patch.object(
            agent,
            "_build_route_pending_request",
            side_effect=[
                PendingRequest(
                    intent_family="comprehensive_ranking",
                    intent_label_zh="综合热榜",
                    ambiguous_fields=["请确认 top_n 的具体值"],
                    should_execute_now=False,
                    source_turn_id=1,
                ),
                PendingRequest(
                    intent_family="comprehensive_ranking",
                    intent_label_zh="综合热榜",
                    ambiguous_fields=["请确认 top_n 的具体值"],
                    should_execute_now=False,
                    source_turn_id=2,
                ),
            ],
        ) as mock_route:
            first_reply = agent.chat("给我实时综合热榜前100名，窗口为10天")
            second_reply = agent.chat("改成前50名")

        assert "需要确认" in first_reply or "请确认" in first_reply
        assert "需要确认" in second_reply or "请确认" in second_reply
        assert mock_route.call_count == 2
        assert agent.state.awaiting_confirmation is True

    def test_confirmation_gate_reconfirms_when_reply_startswith_ack_but_has_modification(self):
        from github_hot_projects.agent import HotProjectAgent, PendingRequest

        agent = HotProjectAgent()
        agent.state.awaiting_confirmation = True
        agent.state.pending_request = PendingRequest(
            intent_family="comprehensive_ranking",
            intent_label_zh="综合热榜",
            ambiguous_fields=["请确认 top_n"],
            should_execute_now=False,
            source_turn_id=1,
        )

        with patch.object(
            agent,
            "_build_route_pending_request",
            return_value=PendingRequest(
                intent_family="comprehensive_ranking",
                intent_label_zh="综合热榜",
                ambiguous_fields=["请确认 top_n"],
                should_execute_now=False,
                source_turn_id=2,
            ),
        ) as mock_route:
            intercept, confirmed = agent._maybe_handle_confirmation_gate("可以改成前50名")

        assert "需要确认" in intercept or "请确认" in intercept
        assert confirmed is False
        assert agent.state.awaiting_confirmation is True
        mock_route.assert_called_once()

    def test_chat_routes_greeting_request(self):
        from github_hot_projects.agent import HotProjectAgent, PendingRequest

        agent = HotProjectAgent()

        with patch.object(
            agent,
            "_build_route_pending_request",
            return_value=PendingRequest(
                turn_kind="greeting",
                intent_family="freeform_answer",
                intent_label_zh="自由回答",
                should_execute_now=True,
                route_confidence="high",
                source_turn_id=1,
            ),
        ) as mock_route:
            with patch.object(
                agent,
                "_call_llm",
                return_value={"choices": [{"message": {"content": "我是 GitHub 热门项目助手", "tool_calls": []}}]},
            ) as mock_llm:
                reply = agent.chat("你好")

        assert "GitHub 热门项目助手" in reply
        mock_route.assert_called_once()
        mock_llm.assert_called_once()
        assert agent.state.awaiting_confirmation is False

    def test_chat_routes_capability_query_request(self):
        from github_hot_projects.agent import HotProjectAgent, PendingRequest

        agent = HotProjectAgent()

        with patch.object(
            agent,
            "_build_route_pending_request",
            return_value=PendingRequest(
                turn_kind="capability_query",
                intent_family="freeform_answer",
                intent_label_zh="自由回答",
                should_execute_now=True,
                route_confidence="high",
                source_turn_id=1,
            ),
        ) as mock_route:
            with patch.object(
                agent,
                "_call_llm",
                return_value={"choices": [{"message": {"content": "我是 GitHub 热门项目助手", "tool_calls": []}}]},
            ) as mock_llm:
                reply = agent.chat("你能做什么")

        assert "GitHub 热门项目助手" in reply
        mock_route.assert_called_once()
        mock_llm.assert_called_once()
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

        with patch("github_hot_projects.agent.tool_search_by_keywords", return_value={"repos": [], "total": 1, "_raw_repos": raw_repos}):
            result = agent._execute_tool("search_by_keywords", {})

        assert result["total"] == 1
        assert agent.state.last_search_repos == raw_repos
        assert "org/repo" in agent.state.seen_repos

    def test_execute_tool_search_resets_discovery_state_on_new_turn(self):
        from github_hot_projects.agent import HotProjectAgent, GROWTH_CALC_DAYS

        agent = HotProjectAgent()
        agent.state.current_user_turn = 2
        agent.state.discovery_turn_id = 1
        agent.state.last_search_repos = [{"full_name": "old/repo", "star": 1}]
        agent.state.last_candidates = {"old/repo": {"growth": 1, "star": 1}}
        agent.state.last_ranked = [("old/repo", {"growth": 1, "star": 1})]
        agent.state.seen_repos = {"old/repo"}
        raw_repos = [{"full_name": "new/repo", "star": 123}]

        with patch(
            "github_hot_projects.agent.tool_search_by_keywords",
            return_value={"repos": [], "total": 1, "_raw_repos": raw_repos},
        ):
            agent._execute_tool("search_by_keywords", {})

        assert agent.state.last_search_repos == raw_repos
        assert agent.state.last_candidates == {}
        assert agent.state.last_ranked == []
        assert agent.state.last_mode == "comprehensive"
        assert agent.state.last_growth_calc_days == GROWTH_CALC_DAYS
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

    def test_build_route_pending_request_disables_thinking(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        agent.state.conversation.append({"role": "user", "content": "给我实时综合热榜前一百名，增长窗口10天"})

        with patch.object(
            agent,
            "_request_llm",
            return_value={
                "choices": [{
                    "message": {
                        "content": json.dumps(
                            {
                                "turn_kind": "new_request",
                                "intent_family": "comprehensive_ranking",
                                "intent_label_zh": "综合热榜",
                                "specified_params": {"growth_calc_days": 10, "top_n": 100},
                                "ambiguous_fields": [],
                                "report_requested": False,
                                "should_execute_now": True,
                                "must_call_tool_before_reply": False,
                            },
                            ensure_ascii=False,
                        )
                    }
                }]
            },
        ) as mock_request:
            pending = agent._build_route_pending_request()

        assert pending.intent_family == "comprehensive_ranking"
        assert pending.should_execute_now is True
        assert mock_request.call_args.kwargs["enable_thinking"] is False
        assert mock_request.call_args.kwargs["max_tokens"] == 1024

    def test_build_route_pending_request_parses_structured_pending_request(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        agent.state.current_user_turn = 3
        agent.state.conversation.append({"role": "user", "content": "给我近10天综合热榜前20名，并生成报告"})

        with patch.object(
            agent,
            "_request_llm",
            return_value={
                "choices": [{
                    "message": {
                        "content": json.dumps(
                            {
                                "intent_family": "comprehensive_ranking",
                                "intent_label_zh": "综合热榜",
                                "specified_params": {"growth_calc_days": 10, "top_n": 20},
                                "ambiguous_fields": [],
                                "report_requested": True,
                                "should_execute_now": True,
                                "must_call_tool_before_reply": False,
                                "confirmation_text_zh": "收到！我理解为：综合热榜，统计近10天的增长，返回前20名，并在结果后生成报告。",
                            },
                            ensure_ascii=False,
                        )
                    }
                }]
            },
        ):
            pending = agent._build_route_pending_request()

        assert "返回前20名" in pending.confirmation_text_zh
        assert pending.intent_family == "comprehensive_ranking"
        assert pending.user_specified_params == {"growth_calc_days": 10, "top_n": 20}
        assert pending.report_requested is True

    def test_parse_pending_request_ignores_structured_confirmation_text(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        content = json.dumps(
            {
                "intent_family": "keyword_ranking",
                "intent_label_zh": "关键词热榜",
                "specified_params": {"categories": ["llm"]},
                "ambiguous_fields": [],
                "confirmation_text_zh": '{"intent_family":"keyword_ranking"}',
            },
            ensure_ascii=False,
        )

        pending = agent._parse_pending_request_content(content)

        assert pending.intent_family == "keyword_ranking"
        assert "收到！我理解为：" in pending.confirmation_text_zh
        assert "关键词热榜" in pending.confirmation_text_zh

    def test_call_llm_appends_confirmed_request_context(self):
        from github_hot_projects.agent import HotProjectAgent, ResolvedRequest

        agent = HotProjectAgent()
        agent.state.last_confirmed_request = ResolvedRequest(
            intent_family="hot_new_ranking",
            intent_label_zh="新项目热榜",
            resolved_params={"mode": "hot_new", "days_since_created": 30, "growth_calc_days": 10},
            user_specified_params={"days_since_created": 30},
            defaulted_params={"growth_calc_days": 10},
            report_requested=True,
        )

        with patch.object(
            agent,
            "_request_llm",
            return_value={"choices": [{"message": {"content": "ok", "tool_calls": []}}]},
        ) as mock_request:
            agent._call_llm(execution_confirmed=True)

        messages = mock_request.call_args.kwargs["messages"]
        assert "[执行上下文]" in messages[0]["content"]
        assert "[已确认请求]" in messages[0]["content"]
        # 新架构使用 Python 列表格式显示可用工具
        assert "search_by_keywords" in messages[0]["content"]
        assert "generate_report" in messages[0]["content"]

    def test_select_tools_for_llm_filters_for_constrained_intent(self):
        from github_hot_projects.agent import HotProjectAgent, ResolvedRequest

        agent = HotProjectAgent()
        agent.state.last_confirmed_request = ResolvedRequest(
            intent_family="keyword_ranking",
            intent_label_zh="关键词热榜",
            route_confidence="high",
        )

        selected = agent._select_tools_for_llm()
        names = {
            schema.get("function", {}).get("name")
            for schema in selected
            if schema.get("function", {}).get("name")
        }
        assert names == {"search_by_keywords", "batch_check_growth", "rank_candidates"}

    def test_select_tools_for_llm_keeps_all_for_open_intent(self):
        from github_hot_projects.agent import HotProjectAgent, ResolvedRequest
        from github_hot_projects.parsing.schema import TOOL_SCHEMAS

        agent = HotProjectAgent()
        agent.state.last_confirmed_request = ResolvedRequest(
            intent_family="repo_info",
            intent_label_zh="单仓库综合查询",
            route_confidence="high",
        )

        selected = agent._select_tools_for_llm()
        assert len(selected) == len(TOOL_SCHEMAS)

    def test_compress_conversation_preserves_recent_messages_and_adds_summary(self):
        from github_hot_projects.agent import HotProjectAgent, KEEP_RECENT_MESSAGES, MAX_CONVERSATION_MESSAGES

        agent = HotProjectAgent()
        agent.state.last_search_repos = [{"full_name": "org/repo", "star": 1}]
        agent.state.last_candidates = {"org/repo": {"growth": 1, "star": 1}}
        agent.state.last_ranked = [("org/repo", {"growth": 1, "star": 1})]
        agent.state.last_mode = "hot_new"
        agent.state.last_growth_calc_days = 10
        agent.state.last_candidate_days_since_created = 45
        for idx in range(MAX_CONVERSATION_MESSAGES + 5):
            agent.state.conversation.append({"role": "user", "content": f"用户消息 {idx}"})
            agent.state.conversation.append({"role": "assistant", "content": f"助手回复 {idx}"})

        agent._compress_conversation()

        system_messages = [msg for msg in agent.state.conversation if msg.get("role") == "system"]
        assert len(system_messages) == 1
        assert "对话历史摘要" in (system_messages[0].get("content") or "")
        recent_contents = [msg.get("content") for msg in agent.state.conversation[-KEEP_RECENT_MESSAGES:]]
        assert any("助手回复" in (content or "") for content in recent_contents)

    def test_compress_conversation_keeps_system_prompt_after_multiple_passes(self):
        from github_hot_projects.agent import HotProjectAgent, MAX_CONVERSATION_MESSAGES

        agent = HotProjectAgent()

        with patch.object(agent, "_generate_summary_with_llm", return_value="summary"):
            for idx in range(MAX_CONVERSATION_MESSAGES + 5):
                agent.state.conversation.append({"role": "user", "content": f"u{idx}"})
                agent.state.conversation.append({"role": "assistant", "content": f"a{idx}"})
            agent._compress_conversation()

            for idx in range(MAX_CONVERSATION_MESSAGES + 5):
                agent.state.conversation.append({"role": "user", "content": f"u2-{idx}"})
                agent.state.conversation.append({"role": "assistant", "content": f"a2-{idx}"})
            agent._compress_conversation()

        system_content = (agent.state.conversation[0].get("content") or "")
        assert "你是GitHub热门项目发现助手" in system_content
        assert "[对话历史摘要]" in system_content


class TestValidateToolArgs:
    """直接测试 validate_tool_args 函数。"""

    def test_preserves_llm_days_since_created(self):
        from github_hot_projects.parsing.arg_validator import validate_tool_args

        result = validate_tool_args("search_by_keywords", {"days_since_created": 20})
        assert result["days_since_created"] == 20

    def test_defaults_missing_params(self):
        from github_hot_projects.parsing.arg_validator import validate_tool_args
        from github_hot_projects.common.config import MIN_STAR

        result = validate_tool_args("search_by_keywords", {})
        assert result["min_star"] == MIN_STAR
        assert "days_since_created" not in result

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

        result = validate_tool_args("fetch_trending", {"trending_range": "daily", "language": "python"})
        assert "language" not in result
        assert result["trending_range"] == "daily"

    def test_default_by_mode_hot_new(self):
        from github_hot_projects.parsing.arg_validator import validate_tool_args
        from github_hot_projects.common.config import HOT_NEW_PROJECT_COUNT, DAYS_SINCE_CREATED

        result = validate_tool_args("rank_candidates", {"mode": "hot_new"})
        assert result["top_n"] == HOT_NEW_PROJECT_COUNT
        assert result["days_since_created"] == DAYS_SINCE_CREATED

    def test_default_by_mode_comprehensive(self):
        from github_hot_projects.parsing.arg_validator import validate_tool_args
        from github_hot_projects.common.config import HOT_PROJECT_COUNT

        result = validate_tool_args("rank_candidates", {"mode": "comprehensive"})
        assert result["top_n"] == HOT_PROJECT_COUNT
        assert "days_since_created" not in result

    def test_enum_coercion_trending_range(self):
        from github_hot_projects.parsing.arg_validator import validate_tool_args

        result = validate_tool_args("fetch_trending", {"trending_range": "invalid_value"})
        # Should fall back to default
        assert result["trending_range"] == "weekly"

    def test_preserves_required_str(self):
        from github_hot_projects.parsing.arg_validator import validate_tool_args

        result = validate_tool_args("check_repo_growth", {"repo": "org/repo"})
        assert result["repo"] == "org/repo"


class TestConfirmedRequestExecution:
    def test_batch_check_growth_suggests_missing_collection_tools_but_not_required(self):
        """LLM决策为主：缺失建议工具只打印警告，不强制阻断。"""
        from github_hot_projects.agent import HotProjectAgent, ResolvedRequest

        agent = HotProjectAgent()
        agent.state.last_confirmed_request = ResolvedRequest(
            intent_family="comprehensive_ranking",
            intent_label_zh="综合热榜",
            resolved_params={"mode": "comprehensive", "growth_calc_days": 10},
        )
        agent.state.current_turn_tools = {"search_by_keywords", "scan_star_range"}
        agent.state.last_search_repos = [{"full_name": "org/repo", "star": 1}]

        # 现在不强制返回错误，只是打印警告
        with patch(
            "github_hot_projects.agent.tool_batch_check_growth",
            return_value={"candidates": {"org/repo": {"growth": 500}}, "growth_calc_days": 10, "db_updated": False},
        ):
            result = agent._execute_tool("batch_check_growth", {})

        # 应该正常执行，不返回错误
        assert "candidates" in result

    def test_keyword_ranking_suggestion_does_not_require_scan_or_trending(self):
        from github_hot_projects.agent import HotProjectAgent, ResolvedRequest

        agent = HotProjectAgent()
        agent.state.last_confirmed_request = ResolvedRequest(
            intent_family="keyword_ranking",
            intent_label_zh="关键词热榜",
            resolved_params={"categories": ["llm"]},
        )
        agent.state.current_turn_tools = {"search_by_keywords"}

        missing = agent._check_suggested_collection_tools("batch_check_growth")
        assert missing == []

    def test_batch_check_growth_injects_confirmed_request_defaults(self):
        from github_hot_projects.agent import HotProjectAgent, ResolvedRequest

        agent = HotProjectAgent()
        agent.state.last_confirmed_request = ResolvedRequest(
            intent_family="hot_new_ranking",
            intent_label_zh="新项目热榜",
            resolved_params={
                "mode": "hot_new",
                "growth_calc_days": 10,
                "days_since_created": 30,
                "growth_threshold": 400,
            },
            user_specified_params={"days_since_created": 30},
        )
        agent.state.current_turn_tools = {"search_by_keywords", "scan_star_range", "fetch_trending"}
        agent.state.last_search_repos = [{"full_name": "org/repo", "star": 1}]

        with patch(
            "github_hot_projects.agent.tool_batch_check_growth",
            return_value={"candidates": {"org/repo": {"growth": 500}}, "growth_calc_days": 10, "db_updated": False},
        ) as mock_batch:
            result = agent._execute_tool("batch_check_growth", {})

        assert "candidates" in result
        assert mock_batch.call_args.kwargs["growth_calc_days"] == 10
        assert mock_batch.call_args.kwargs["days_since_created"] == 30
        assert mock_batch.call_args.kwargs["growth_threshold"] == 400

    def test_log_execution_overview_prints_summary_snapshot(self):
        """日志应包含单条参数总览，hot_new 模式应为 desc_only 策略。"""
        from github_hot_projects.agent import HotProjectAgent, ResolvedRequest

        agent = HotProjectAgent()
        agent.state.current_user_turn = 7
        agent.state.last_confirmed_request = ResolvedRequest(
            intent_family="hot_new_ranking",
            intent_label_zh="新项目热榜",
            resolved_params={
                "mode": "hot_new",
                "growth_calc_days": 10,
                "days_since_created": 30,
            },
            user_specified_params={"growth_calc_days": 10, "days_since_created": 30},
            defaulted_params={"mode": "hot_new"},
            report_requested=True,
        )

        with patch("github_hot_projects.agent.logger.info") as mock_info:
            agent._log_execution_overview()

        merged_logs = "\n".join(
            " ".join(str(arg) for arg in call.args)
            for call in mock_info.call_args_list
        )
        assert "运行参数总览" in merged_logs
        assert "desc_only" in merged_logs
        assert '"growth_calc_days": 10' in merged_logs

    def test_log_execution_overview_shows_trending_range_all(self):
        from github_hot_projects.agent import HotProjectAgent, ResolvedRequest

        agent = HotProjectAgent()
        agent.state.current_user_turn = 9
        agent.state.last_confirmed_request = ResolvedRequest(
            intent_family="comprehensive_ranking",
            intent_label_zh="综合热榜",
            resolved_params={
                "mode": "comprehensive",
                "trending_range": "all",
            },
            user_specified_params={"mode": "comprehensive"},
        )

        with patch("github_hot_projects.agent.logger.info") as mock_info:
            agent._log_execution_overview()

        merged_logs = "\n".join(
            " ".join(str(arg) for arg in call.args)
            for call in mock_info.call_args_list
        )
        assert '"trending_range": "all"' in merged_logs

    def test_batch_check_growth_hot_new_persists_desc_only(self):
        from github_hot_projects.agent import HotProjectAgent, ResolvedRequest

        agent = HotProjectAgent()
        agent.state.last_confirmed_request = ResolvedRequest(
            intent_family="hot_new_ranking",
            intent_label_zh="新项目热榜",
            resolved_params={"mode": "hot_new", "days_since_created": 30},
        )
        agent.state.current_turn_tools = {"search_by_keywords", "scan_star_range", "fetch_trending"}
        agent.state.last_search_repos = [{"full_name": "org/repo", "star": 1}]

        with patch(
            "github_hot_projects.agent.tool_batch_check_growth",
            return_value={"candidates": {"org/repo": {"growth": 500}}, "growth_calc_days": 10, "db_updated": True},
        ):
            with patch("github_hot_projects.agent.save_db_desc_only", return_value=1) as mock_desc_save:
                agent._execute_tool("batch_check_growth", {})

        mock_desc_save.assert_called_once_with(agent.state.db)

    def test_generate_report_hot_new_persists_desc_only(self):
        from github_hot_projects.agent import HotProjectAgent

        agent = HotProjectAgent()
        agent.state.last_mode = "hot_new"
        agent.state.last_ranked = [("org/repo", {"growth": 100, "star": 200})]

        with patch(
            "github_hot_projects.agent.tool_generate_report",
            return_value={"report_path": "reports/now.md"},
        ):
            with patch("github_hot_projects.agent.save_db_desc_only", return_value=1) as mock_desc_save:
                agent._execute_tool("generate_report", {})

        mock_desc_save.assert_called_once_with(agent.state.db)

    def test_generate_report_comprehensive_writes_desc_only(self):
        """comprehensive 模式应只写 desc_only（full DB 写入由定时脚本负责）。"""
        from github_hot_projects.agent import HotProjectAgent, ResolvedRequest

        agent = HotProjectAgent()
        agent.state.last_mode = "comprehensive"
        agent.state.last_ranked = [("org/repo", {"growth": 100, "star": 200})]
        agent.state.last_confirmed_request = ResolvedRequest(
            intent_family="comprehensive_ranking",
            intent_label_zh="综合热榜",
            resolved_params={"mode": "comprehensive"},
        )

        with patch(
            "github_hot_projects.agent.tool_generate_report",
            return_value={"report_path": "reports/now.md"},
        ):
            with patch("github_hot_projects.agent.save_db_desc_only", return_value=1) as mock_desc_save:
                agent._execute_tool("generate_report", {})

        mock_desc_save.assert_called_once()

    def test_batch_check_growth_comprehensive_writes_desc_only(self):
        """comprehensive 模式应只写 desc_only（full DB 写入由定时脚本负责）。"""
        from github_hot_projects.agent import HotProjectAgent, ResolvedRequest

        agent = HotProjectAgent()
        agent.state.last_confirmed_request = ResolvedRequest(
            intent_family="comprehensive_ranking",
            intent_label_zh="综合热榜",
            resolved_params={"mode": "comprehensive", "growth_calc_days": 7},
        )
        agent.state.current_turn_tools = {"search_by_keywords", "scan_star_range", "fetch_trending"}
        agent.state.last_search_repos = [{"full_name": "org/repo", "star": 1}]

        with patch(
            "github_hot_projects.agent.tool_batch_check_growth",
            return_value={"candidates": {"org/repo": {"growth": 500}}, "growth_calc_days": 7, "db_updated": True},
        ):
            with patch("github_hot_projects.agent.save_db_desc_only", return_value=1) as mock_desc_save:
                agent._execute_tool("batch_check_growth", {})

        mock_desc_save.assert_called_once()

    def test_batch_check_growth_hot_new_writes_desc_only(self):
        """hot_new 模式应只写 desc_only。"""
        from github_hot_projects.agent import HotProjectAgent, ResolvedRequest

        agent = HotProjectAgent()
        agent.state.last_confirmed_request = ResolvedRequest(
            intent_family="hot_new_ranking",
            intent_label_zh="新项目热榜",
            resolved_params={"mode": "hot_new", "days_since_created": 10},
        )
        agent.state.current_turn_tools = {"search_by_keywords", "scan_star_range", "fetch_trending"}
        agent.state.last_search_repos = [{"full_name": "org/repo", "star": 1}]

        with patch(
            "github_hot_projects.agent.tool_batch_check_growth",
            return_value={"candidates": {"org/repo": {"growth": 500}}, "growth_calc_days": 7, "db_updated": True},
        ):
            with patch("github_hot_projects.agent.save_db_desc_only", return_value=1) as mock_desc_save:
                agent._execute_tool("batch_check_growth", {})

        mock_desc_save.assert_called_once()
