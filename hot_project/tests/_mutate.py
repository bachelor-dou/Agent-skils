"""变异检查:把每处守卫故意改坏,确认对应测试真的变红。

不是测试,是个一次性核对脚本(`python -m hot_project.tests._mutate`)。
留在仓库里是为了以后改数据层时能再跑一遍 —— 一条永远绿的测试和没有测试是一回事。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "hot_project" / "infra" / "store"
TOKENS = ROOT / "hot_project" / "provider" / "github" / "tokens.py"
TASKS = ROOT / "hot_project" / "infra" / "tasks" / "pool.py"
CRON = ROOT / "hot_project" / "cron_daily_snapshot.py"
GH_TASKS = ROOT / "hot_project" / "provider" / "github" / "tasks.py"
GH_CLIENT = ROOT / "hot_project" / "provider" / "github" / "client.py"
GH_TRENDING = ROOT / "hot_project" / "provider" / "github" / "trending.py"
GROWTH = ROOT / "hot_project" / "core" / "growth.py"
LLM_WIRE = ROOT / "hot_project" / "infra" / "llm" / "wire.py"
LLM_CLIENT = ROOT / "hot_project" / "infra" / "llm" / "client.py"
LLM_SCHEMES = ROOT / "hot_project" / "infra" / "llm" / "schemes.py"
ENV = ROOT / "hot_project" / "common" / "env.py"
SCORING = ROOT / "hot_project" / "core" / "scoring.py"
REPORT_PARSE = ROOT / "hot_project" / "core" / "report_parse.py"
RANKING = ROOT / "hot_project" / "tools" / "ranking.py"
REPORT = ROOT / "hot_project" / "tools" / "report.py"
SPEC = ROOT / "hot_project" / "tools" / "spec.py"
REPO_TOOLS = ROOT / "hot_project" / "tools" / "repo_tools.py"
RANK_TOOLS = ROOT / "hot_project" / "tools" / "rank_tools.py"
HISTORY = ROOT / "hot_project" / "agent" / "history.py"
AGENT_LOOP = ROOT / "hot_project" / "agent" / "loop.py"
SECURITY = ROOT / "hot_project" / "web" / "security.py"
SESSIONS = ROOT / "hot_project" / "web" / "sessions.py"
REPORTS_STORE = ROOT / "hot_project" / "infra" / "store" / "reports.py"
RENDER = ROOT / "hot_project" / "web" / "render.py"

STORE_TESTS = "hot_project/tests/test_store.py"
TOKEN_TESTS = "hot_project/tests/test_tokens.py"
TASK_TESTS = "hot_project/tests/test_tasks.py"
SNAP_TESTS = "hot_project/tests/test_snapshot.py"
GROWTH_TESTS = "hot_project/tests/test_growth.py"
LAYER_TESTS = "hot_project/tests/test_layering.py"
LLM_TESTS = "hot_project/tests/test_llm.py"
RANKING_TESTS = "hot_project/tests/test_ranking.py"
REPORT_TESTS = "hot_project/tests/test_report.py"
TOOL_TESTS = "hot_project/tests/test_tools.py"
AGENT_TESTS = "hot_project/tests/test_agent.py"
WEB_TESTS = "hot_project/tests/test_web.py"

# (说明, 文件, 原文片段, 改坏后的片段, 测试文件, 应当变红的测试)
MUTATIONS: list[tuple[str, Path, str, str, str, str]] = [
    (
        "读盘失败当空字典继续写(旧 save_db 的真实事故)",
        STORE / "atomic.py",
        'raise StoreReadError(f"{path} 读取失败({e})—— 放弃本次操作,盘上数据保持原样") from e',
        "return default if default is not None else {}",
        STORE_TESTS,
        "test_corrupt_db_is_never_overwritten",
    ),
    (
        "发现任务不校验字段归属",
        STORE / "universe.py",
        '_check_fields(records, DISCOVER_FIELDS, "每日发现")',
        "pass",
        STORE_TESTS,
        "test_discover_rejects_foreign_fields",
    ),
    (
        "发现任务附带覆盖已有条目",
        STORE / "universe.py",
        """            if name in projects:
                continue
            projects[name] = dict(info)""",
        "            projects[name] = dict(info)",
        STORE_TESTS,
        "test_discover_only_inserts_new",
    ),
    (
        "展示字段无条件覆写(抹掉已有 language / gh_desc)",
        STORE / "universe.py",
        "                if not value or target.get(key):\n                    continue",
        "                if False:\n                    continue",
        STORE_TESTS,
        "test_display_refresh_never_overwrites_existing",
    ),
    (
        "star 没变也重写整库",
        STORE / "universe.py",
        "    if not changed:\n        tx.abort()\n        return 0",
        "    if not changed:\n        return 0",
        STORE_TESTS,
        "test_no_change_writes_nothing",
    ),
    (
        "不检查快照覆盖率",
        STORE / "snapshots.py",
        "    if coverage < MIN_COVERAGE:",
        "    if False:",
        STORE_TESTS,
        "test_snapshot_rejects_low_coverage",
    ),
    (
        "prune 连非快照文件一起删",
        STORE / "snapshots.py",
        """    for path in sorted(directory.glob(f"*{_SUFFIX}")):
        day = parse_day(path.name[: -len(_SUFFIX)])
        if day is None or day >= cutoff:""",
        """    for path in sorted(directory.glob("*")):
        day = parse_day(path.name[: -len(_SUFFIX)]) or shift_days(cutoff, -1)
        if day >= cutoff:""",
        STORE_TESTS,
        "test_prune_only_deletes_snapshots",
    ),
    (
        "今天那份也拿来当基线(窗口 0 天,增长恒为 0,速率计算还会除零)",
        STORE / "snapshots.py",
        "        if span < 1 or day < floor:",
        "        if day < floor:",
        STORE_TESTS,
        "test_the_baseline_takes_each_repos_earliest_measurement_in_the_window",
    ),
    (
        "窗口外的老快照也当基线(等于偷偷把窗口拉长,增长阈值形同虚设)",
        STORE / "snapshots.py",
        "        if span < 1 or day < floor:",
        "        if span < 1:",
        STORE_TESTS,
        "test_a_snapshot_older_than_the_window_is_not_a_baseline",
    ),
    (
        "后来的快照覆盖最早的(基线越用越新,增长越算越小)",
        STORE / "snapshots.py",
        "            if name not in stars:",
        "            if True:",
        STORE_TESTS,
        "test_the_baseline_takes_each_repos_earliest_measurement_in_the_window",
    ),
    (
        "基线天数报全局窗口而非逐仓实际(晚进库的仓库速率虚高,爆发加成凭空多一档)",
        STORE / "snapshots.py",
        "                spans[name] = span",
        "                spans[name] = days",
        STORE_TESTS,
        "test_the_baseline_takes_each_repos_earliest_measurement_in_the_window",
    ),
    (
        "读取层丢掉未知字段",
        STORE / "universe.py",
        "    return projects if isinstance(projects, dict) else {}",
        "    return {n: {'star': p.get('star')} for n, p in (projects or {}).items()}",
        STORE_TESTS,
        "test_reading_preserves_fields_this_version_knows_nothing_about",
    ),
    (
        "收藏读完就放锁再写(旧实现的丢更新竞态)",
        STORE / "favorites.py",
        "    with transaction(config.FAVORITES_PATH, default=_empty()) as tx:",
        """    import time
    from .atomic import read_json as _rj, write_whole as _ww
    import json as _json

    class _FakeTx:
        def __init__(self, d): self.data = d
        def abort(self): pass

    _loaded = _rj(config.FAVORITES_PATH, default=_empty())
    time.sleep(0.02)          # 放大旧实现的竞态窗口

    class _Ctx:
        def __enter__(self): return _FakeTx(_loaded)
        def __exit__(self, *a):
            if a[0] is None:
                _ww(config.FAVORITES_PATH,
                    lambda t: t.write_text(_json.dumps(_loaded), encoding="utf-8"))
            return False
    with _Ctx() as tx:""",
        STORE_TESTS,
        "test_concurrent_favorites_do_not_lose_updates",
    ),
    # ── token 池 ──
    (
        "401 直接永久失效(cron_daily_star_snapshot.py:168 与 api.py:400 的真实 bug)",
        TOKENS,
        """            if token.auth_fails >= self._strikes:""",
        "            if True:",
        TOKEN_TESTS,
        "test_single_401_does_not_burn_the_token",
    ),
    (
        "成功归还不清零 401 连续计数",
        TOKENS,
        "            if healthy:\n                token.auth_fails = 0",
        "            if False:\n                token.auth_fails = 0",
        TOKEN_TESTS,
        "test_a_success_resets_the_strike_counter",
    ),
    (
        "租约吞掉异常(重试与否就不再由任务池决定了)",
        TOKENS,
        "        except TokenInvalidError as e:\n            await self._on_auth_failed(index, str(e))\n            raise",
        "        except TokenInvalidError as e:\n            await self._on_auth_failed(index, str(e))",
        TOKEN_TESTS,
        "test_exception_propagates",
    ),
    (
        "不记录配速时刻(退回「靠调用点自己 sleep」)",
        TOKENS,
        "                    if pace.interval > 0:\n                        token.next_at[pace.key] = now + pace.interval",
        "                    if False:\n                        token.next_at[pace.key] = now + pace.interval",
        TOKEN_TESTS,
        "test_same_token_respects_the_pace_interval",
    ),
    (
        "配速不分类(GraphQL 被迫等 Search 的 2.1 秒)",
        TOKENS,
        '        return max(self.available_at, self.next_at.get(pace.key, 0.0))',
        '        return max(self.available_at, max(self.next_at.values(), default=0.0))',
        TOKEN_TESTS,
        "test_pace_is_per_kind",
    ),
    (
        "_pick 不看 in_use(同一 token 被同时借出两次)",
        TOKENS,
        "            if token.invalid or token.in_use or token.ready_at(pace) > now:",
        "            if token.invalid or token.ready_at(pace) > now:",
        TOKEN_TESTS,
        "test_concurrent_leases_are_bounded_by_token_count",
    ),
    (
        "_pick 按 ready_at 排序(限流过的 token 冷却后被永久冷落)",
        TOKENS,
        "            if best is None or token.used_seq < self._tokens[best].used_seq:",
        "            if best is None or token.ready_at(pace) < self._tokens[best].ready_at(pace):",
        TOKEN_TESTS,
        "test_rate_limited_token_is_skipped_until_reset",
    ),
    (
        "等待没有超时(限流后所有 worker 永久挂死)",
        TOKENS,
        "                try:\n                    # 等冷却/配速到期,但也接受被提前叫醒(有人归还了、或新增了 token)。\n                    await asyncio.wait_for(self._cond.wait(), timeout=wait)\n                except TimeoutError:\n                    pass",
        "                await self._cond.wait()",
        TOKEN_TESTS,
        "test_waiter_wakes_itself_on_real_clock",
    ),
    (
        "capacity 不扣除已失效的 token",
        TOKENS,
        "        return sum(1 for t in self._tokens if not t.invalid)",
        "        return len(self._tokens)",
        TOKEN_TESTS,
        "test_capacity_tracks_tokens_without_a_configured_number",
    ),
    (
        "新增 token 不唤醒等待者(白等到冷却结束)",
        TOKENS,
        "            self._cond.notify_all()      # 可能有 worker 正因「全都在冷却」而挂着",
        "            pass",
        TOKEN_TESTS,
        "test_adding_a_token_wakes_a_blocked_waiter",
    ),

    # ── 任务池 ────────────────────────────────────────────
    (
        "所有任务挤进一条道(退回单队列,搜索被 GraphQL 堵死)",
        TASKS,
        "        queue = self._queues.get(task.lane)",
        "        queue = self._queues.get(next(iter(self._queues)))",
        TASK_TESTS,
        "test_lanes_do_not_block_each_other",
    ),
    (
        "每条道只开一个 worker(并发度打回 1)",
        TASKS,
        "            for i in range(size):",
        "            for i in range(1):",
        TASK_TESTS,
        "test_concurrency_equals_worker_count",
    ),
    (
        "提交时不加 pending(join 在派生任务跑完前就返回)",
        TASKS,
        "        self._pending += 1\n        self._idle.clear()",
        "        pass",
        TASK_TESTS,
        "test_join_waits_for_derived_tasks",
    ),
    (
        "限流也计入重试次数(一轮限流高峰烧光所有重试额度)",
        TASKS,
        "            task.attempts -= 1\n            task.rate_limits += 1",
        "            task.rate_limits += 1",
        TASK_TESTS,
        "test_rate_limit_does_not_consume_the_retry_budget",
    ),
    (
        "重试没有上限(网络长期不通时无限自旋)",
        TASKS,
        "            if task.attempts > task.max_retries:\n                self._finish(task, err=e)",
        "            if False:\n                self._finish(task, err=e)",
        TASK_TESTS,
        "test_retries_are_capped",
    ),
    (
        "程序 bug 也重排(刷屏且永不收敛)",
        TASKS,
        "        except Exception as e:                      # noqa: BLE001 - 兜底,不重排\n            self._finish(task, err=e)",
        "        except Exception:                           # noqa: BLE001\n            queue.put_nowait(task)",
        TASK_TESTS,
        "test_programming_error_fails_immediately",
    ),
    (
        "回调抛异常时漏减 pending(join 永远不返回)",
        TASKS,
        '            logger.exception("%r 的回调抛了异常", task)\n        finally:',
        '            logger.exception("%r 的回调抛了异常", task)\n        else:',
        TASK_TESTS,
        "test_callback_blowing_up_does_not_hang_join",
    ),
    (
        "不吃 token 的任务也去借租约(白占一张)",
        TASKS,
        "        if not task.needs_token:\n            return _NoToken()",
        "        if False:\n            return _NoToken()",
        TASK_TESTS,
        "test_task_without_token_never_touches_the_leaser",
    ),

    # ── 每日快照:淘汰、采集、收集 ──────────────────────────
    (
        "把「没问到」当成「查不到」(一次限流能删掉上万个活仓库)",
        CRON,
        "        missing=sorted(confirmed_missing & tracked),",
        "        missing=sorted(tracked - stars.keys()),",
        SNAP_TESTS,
        "test_a_whole_batch_failing_evicts_nobody",
    ),
    (
        "淘汰把等于门槛的也算进去(和发现阶段天天打架)",
        CRON,
        "            if name in tracked and star < star_floor",
        "            if name in tracked and star <= star_floor",
        SNAP_TESTS,
        "test_exactly_at_the_floor_stays",
    ),
    (
        "采集失败的批次记进 missing 而不是 failed",
        GH_TASKS,
        "        self.sink.failed.update(self.names)",
        "        self.sink.missing.update(self.names)",
        SNAP_TESTS,
        "test_a_batch_that_never_answers_goes_to_failed_not_missing",
    ),
    (
        "整批 null 的退化响应被当成「这批仓库都没了」",
        GH_CLIENT,
        "        if len(names) > 1:\n            return None",
        "        if False:\n            return None",
        SNAP_TESTS,
        "test_a_degenerate_all_null_batch_splits_instead_of_evicting",
    ),
    (
        "限流无上限重排(CI 一直转到六小时超时,不落盘也不报错)",
        TASKS,
        "            if task.rate_limits > task.max_rate_limits:",
        "            if False:",
        SNAP_TESTS,
        "test_a_batch_that_never_answers_goes_to_failed_not_missing",
    ),
    (
        "不满一页也去翻下一页(每个关键词白发一次请求)",
        GH_TASKS,
        "        if len(items) == PER_PAGE and self.page < MAX_PAGES:",
        "        if self.page < MAX_PAGES:",
        SNAP_TESTS,
        "test_keyword_search_follows_full_pages_and_stops_on_a_short_one",
    ),
    (
        "翻页不管 1000 条上限(第 11 页起全是 422)",
        GH_TASKS,
        "        if len(items) == PER_PAGE and self.page < MAX_PAGES:",
        "        if len(items) == PER_PAGE:",
        SNAP_TESTS,
        "test_search_stops_at_the_thousand_result_ceiling",
    ),
    (
        "星段装不下也直接开扫(超出 1000 的部分永远拿不到)",
        GH_TASKS,
        "        if count <= SEARCH_CAP or self.lo >= self.hi:",
        "        if True:",
        SNAP_TESTS,
        "test_a_fat_star_range_splits_before_it_is_scanned",
    ),
    (
        "星段对半拆出重叠区间(同一批仓库扫两遍)",
        GH_TASKS,
        "        ctx.submit(SegmentProbe(self.sink, self.client, mid + 1, self.hi))",
        "        ctx.submit(SegmentProbe(self.sink, self.client, mid, self.hi))",
        SNAP_TESTS,
        "test_a_fat_star_range_splits_before_it_is_scanned",
    ),
    (
        "空段也照样拆下去(白发一整棵子树的请求)",
        GH_TASKS,
        "        if count == 0:\n            return",
        "        if False:\n            return",
        SNAP_TESTS,
        "test_an_empty_star_range_costs_one_request",
    ),
    (
        "403 不认成限流(当场判死,不换 token 重试)",
        GH_CLIENT,
        "    if code in (403, 429):\n        raise RateLimitError(_reset_at(resp.headers), _limit_reason(resp))",
        '    if False:\n        raise RateLimitError(_reset_at(resp.headers), _limit_reason(resp))',
        SNAP_TESTS,
        "test_http_status_maps_to_the_right_exception",
    ),
    (
        "搜索 422(翻过 1000 条上限)被当成故障",
        GH_CLIENT,
        "    if resp.status_code == 422:\n        # 分页越界",
        "    if False:\n        # 分页越界",
        SNAP_TESTS,
        "test_a_422_page_means_no_more_results_not_an_error",
    ),
    (
        "GraphQL 藏在 200 里的限流被当成普通故障(不冷却 token,原地打转)",
        GH_CLIENT,
        '        if "RATE_LIMITED" in errors:',
        "        if False:",
        SNAP_TESTS,
        "test_graphql_rate_limit_hides_inside_a_200",
    ),
    (
        "Trending 解析器坏了不报警(静默返回 0 个仓库)",
        GH_TRENDING,
        "        return self.articles > 0 and len(self.repos) < self.articles * 0.75",
        "        return False",
        SNAP_TESTS,
        "test_a_short_list_is_not_a_broken_parser",
    ),
    (
        "算不出增长被当成涨了 0(仓库带着假的零增长进排名,永远出不了榜)",
        GROWTH,
        "    return None",
        "    return Growth(0, ANCHOR, window_days)",
        GROWTH_TESTS,
        "test_an_old_repo_without_a_baseline_is_left_out_not_zeroed",
    ),
    (
        "基线天数被丢掉,一律报全局窗口(晚进库的仓库速率虚高)",
        GROWTH,
        "        return Growth(current_star - anchor_star, ANCHOR, anchor_days or window_days)",
        "        return Growth(current_star - anchor_star, ANCHOR, window_days)",
        GROWTH_TESTS,
        "test_each_repo_reports_the_days_its_own_baseline_covers",
    ),
    (
        "窗口内新建判定漏掉边界(窗口第一天创建的仓库被剔出排名)",
        GROWTH,
        "    if age_days is not None and age_days <= window_days:",
        "    if age_days is not None and age_days < window_days:",
        GROWTH_TESTS,
        "test_a_repo_exactly_as_old_as_the_window_still_counts",
    ),
    (
        "元数据压过实测快照(created_at 不对时算出整个 star 数当增长)",
        GROWTH,
        """    if anchor_star is not None:
        return Growth(current_star - anchor_star, ANCHOR, anchor_days or window_days)
    if age_days is not None and age_days <= window_days:""",
        """    if age_days is not None and age_days <= window_days:
        return Growth(current_star, NEW_IN_WINDOW, max(1, round(age_days)))
    if anchor_star is not None:""",
        GROWTH_TESTS,
        "test_measured_snapshot_beats_declared_creation_date",
    ),
    (
        "掉星被夹到零(「掉了 300 星」和「一点没涨」再也分不开)",
        GROWTH,
        "        return Growth(current_star - anchor_star, ANCHOR, anchor_days or window_days)",
        "        return Growth(max(0, current_star - anchor_star), ANCHOR, anchor_days or window_days)",
        GROWTH_TESTS,
        "test_losing_stars_stays_negative",
    ),
    (
        "有人往 core 里 import 了文件系统(纯算法层的全部价值就此失效)",
        GROWTH,
        "from typing import NamedTuple",
        "import pathlib\nfrom typing import NamedTuple",
        LAYER_TESTS,
        "test_core_stays_pure",
    ),
    (
        "给 azure 发它不认的 temperature(整个请求 400,不是被忽略)",
        LLM_WIRE,
        """    if backend == AZURE:
        if max_tokens is not None:
            body["max_completion_tokens"] = max_tokens
    else:""",
        """    if backend == AZURE:
        if max_tokens is not None:
            body["max_completion_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
    else:""",
        LLM_TESTS,
        "test_azure_gets_max_completion_tokens_and_no_temperature",
    ),
    (
        "网页硬切换悄悄回退到别家(用户选了 A,拿到 B 的答案还不知道)",
        LLM_CLIENT,
        "            return [(sel, sel.model)]                           # 硬切换,不回退",
        "            return [(sel, sel.model)] + [(s, s.model) for s in usable if s is not sel]",
        LLM_TESTS,
        "test_a_hard_switch_never_falls_back",
    ),
    (
        "200 但正文为空当成功(整批描述静默变空,最难查的一类失败)",
        LLM_WIRE,
        '        if (msg.get("content") or "").strip() or msg.get("tool_calls"):\n            return data',
        "        if True:\n            return data",
        LLM_TESTS,
        "test_an_empty_reply_counts_as_failure",
    ),
    (
        "只回 tool_calls 被当成空响应(每次都白重试三遍再换平台)",
        LLM_WIRE,
        '        if (msg.get("content") or "").strip() or msg.get("tool_calls"):',
        '        if (msg.get("content") or "").strip():',
        LLM_TESTS,
        "test_a_reply_with_only_tool_calls_is_not_empty",
    ),
    (
        "流式下只回 tool_calls 被当成空响应(agent 用工具的常态,会重复调工具)",
        LLM_WIRE,
        "    if not (content.strip() or tool_calls):",
        "    if not content.strip():",
        LLM_TESTS,
        "test_a_stream_of_only_tool_calls_is_not_empty_either",
    ),
    (
        "已经外发过内容的流式照样重试(前端看到重复文字)",
        LLM_WIRE,
        "            if emitted:\n                return None                     # 已经外发过,不能重试",
        "            if False:\n                return None",
        LLM_TESTS,
        "test_a_stream_that_already_emitted_is_never_retried",
    ),
    (
        "外发过内容还换平台(前端看到「半句 + 另一个完整答案」)",
        LLM_CLIENT,
        "            if emitted:\n                logger.warning(",
        "            if False:\n                logger.warning(",
        LLM_TESTS,
        "test_a_stream_that_already_emitted_is_never_retried",
    ),
    (
        "工具调用泄漏的 JSON 直接吐给用户看",
        LLM_WIRE,
        "    return text if not any(k in blob for k in _LEAK_KEYS) else s[end:].lstrip()",
        "    return text",
        LLM_TESTS,
        "test_a_leaked_toolcall_blob_never_reaches_the_user",
    ),
    (
        "闸门把正常散文也拦住(每个回答的首字延迟都为罕见情况买单)",
        LLM_WIRE,
        "    s = text.lstrip()\n    if not s.startswith(\"{\"):\n        return text",
        "    s = text.lstrip()\n    if False:\n        return text",
        LLM_TESTS,
        "test_prose_is_emitted_without_waiting_for_the_gate",
    ),
    (
        "非 200 早退这条路上不归还连接(连接池几小时后耗干)",
        LLM_WIRE,
        "        if resp is not None:\n            resp.close()",
        "        if False:\n            resp.close()",
        LLM_TESTS,
        "test_the_connection_is_returned_on_every_path",
    ),
    (
        'enabled 写成字符串 "0" 被当真(关掉的平台照样调用,只体现在账单上)',
        ENV,
        "    if isinstance(value, str):\n        return value.strip().lower() in _TRUE_VALUES",
        "    if False:\n        return False",
        LLM_TESTS,
        "test_enabled_as_the_string_zero_still_means_off",
    ),
    (
        "重复的平台 id 放行(选 A 却调到 B,静默发生)",
        LLM_SCHEMES,
        "        if mid in seen:",
        "        if False:",
        LLM_TESTS,
        "test_duplicate_ids_are_a_hard_error",
    ),
    # ── 打分 ────────────────────────────────────────────────
    (
        "两项系数配比被改坏(热度 : 增长率 从 6.5:3.5 滑向增长率主导,榜单塌给低基数)",
        SCORING,
        "math.log2(1 + rate) * 1200",
        "math.log2(1 + rate) * 3600",
        RANKING_TESTS,
        "test_an_absurd_growth_rate_buys_much_less_than_it_looks",
    ),
    (
        "爆发加速比不封顶(一个异常值把整张榜掀翻)",
        SCORING,
        "min(max(acceleration - 1.0, 0.0), w.cap)",
        "max(acceleration - 1.0, 0.0)",
        RANKING_TESTS,
        "test_one_freak_number_cannot_flip_the_board",
    ),
    (
        "减速被反向惩罚(同一件事收两遍税,分数可能变负)",
        SCORING,
        "min(max(acceleration - 1.0, 0.0), w.cap)",
        "min(acceleration - 1.0, w.cap)",
        RANKING_TESTS,
        "test_slowing_down_is_not_punished",
    ),
    (
        "探针缺数据不特判(缺快照的日子整轮排名崩在 TypeError 上)",
        SCORING,
        "    if recent_growth is None or recent_growth < 0 or growth <= 0:",
        "    if growth <= 0:",
        RANKING_TESTS,
        "test_no_probe_data_means_no_boost_not_a_penalty",
    ),
    # ── 榜单流水线 ──────────────────────────────────────────
    (
        "没有可用 token 也照样出榜(拿一份残缺的 star 表排,榜单看着正常其实是错的)",
        RANKING,
        '    if gh is None or not getattr(gh, "usable", False):',
        "    if False:",
        RANKING_TESTS,
        "test_no_github_token_means_no_ranking_rather_than_a_wrong_one",
    ),
    (
        "算不出增长的记成零增长进池(带着假的「一点没涨」参与排名,且漏斗看不出原因)",
        RANKING,
        "        if result is None:\n            unresolved += 1\n            continue",
        "        if result is None:\n            result = growth_calc.Growth(0, growth_calc.ANCHOR, base.span)",
        RANKING_TESTS,
        "test_a_repo_with_no_baseline_and_no_creation_date_is_counted_but_dropped",
    ),
    (
        "低于阈值的也留在内存里(7.8 万条候选全量进池,之后每一步再全表遍历一遍)",
        RANKING,
        "        if result.value < threshold:\n            continue",
        "        if False:\n            continue",
        RANKING_TESTS,
        "test_below_threshold_candidates_never_enter_the_pool",
    ),
    (
        "窗口按请求值而非最早快照的实际跨度(5 天的增量当 7 天用,新项目被误判出局)",
        RANKING,
        "        window = base.span",
        "        window = growth_days",
        RANKING_TESTS,
        "test_the_window_follows_the_oldest_snapshot_not_the_request",
    ),
    (
        "探针不过滤掉星的仓库(负的最近增长混进加成)",
        RANKING,
        '        if anchor is None or info["star"] < anchor:',
        "        if anchor is None:",
        RANKING_TESTS,
        "test_a_repo_that_lost_stars_is_skipped_by_the_probe",
    ),
    (
        "探针天数不逐仓写回(3 天名义值除 5 天的增量,凭空造出一场爆发)",
        RANKING,
        '        info["recent_days"] = base.days.get(name, base.span)',
        '        info["recent_days"] = days',
        RANKING_TESTS,
        "test_the_probe_writes_back_each_repos_own_span",
    ),
    (
        "打分不看逐仓天数(全部按全局窗口折算速率,晚进库的仓库虚高一档)",
        SCORING,
        '                              window_days=item.get("window_days"),',
        "                              window_days=None,",
        RANKING_TESTS,
        "test_each_repo_uses_its_own_baseline_span_not_the_global_one",
    ),
    (
        "发现来源按先到先得记账(同一个词今天 0 个明天 50 个,砍关键词的依据全废)",
        GH_TASKS,
        "        names = {n for item in items if (n := item.get(\"full_name\"))}",
        "        names = {n for item in items\n"
        "                 if (n := item.get(\"full_name\")) and n not in self.repos}",
        SNAP_TESTS,
        "test_every_keyword_gets_credited_for_a_repo_it_returned",
    ),
    # ── 报告解析 ────────────────────────────────────────────
    (
        "增长字段按写死的键名找(窗口一改就静默返回空)",
        REPORT_PARSE,
        'label = next((k for k in metadata if "增长" in k), "")',
        'label = "近7天增长" if "近7天增长" in metadata else ""',
        REPORT_TESTS,
        "test_the_growth_field_is_found_no_matter_the_window",
    ),
    (
        "缺必需元数据的 md 也当榜单解析(说明文档被渲染成空榜)",
        REPORT_PARSE,
        "    if not any(all(e.metadata.get(k) for k in _REQUIRED_META) for e in entries):\n        return None",
        "    pass",
        REPORT_TESTS,
        "test_a_report_without_the_required_metadata_is_not_a_ranking",
    ),
    (
        "描述字段只认英文冒号(模型写全角的那几段整段丢失,报告静默退回兜底文案)",
        REPORT,
        'if line.startswith(f"{s}:") or line.startswith(f"{s}：")), "")',
        'if line.startswith(f"{s}:")), "")',
        REPORT_TESTS,
        "test_both_kinds_of_colon_are_accepted",
    ),
    # ── 报告目录 ────────────────────────────────────────────
    (
        "报告名不挡路径穿越(工具参数来自模型,模型输入来自用户)",
        REPORTS_STORE,
        'if not name or "/" in name or "\\\\" in name or ".." in name:\n        return None',
        "if not name:\n        return None",
        REPORT_TESTS,
        "test_a_report_name_can_never_escape_the_directory",
    ),
    (
        "报告名不核对目录里真有这份(拼出来的路径直接交给下游去读)",
        REPORTS_STORE,
        "    return name if any(item.name == name for item in items) else None",
        "    return name",
        REPORT_TESTS,
        "test_an_unknown_report_name_resolves_to_nothing",
    ),
    (
        "关键词方向名不清洗就进文件名(写报告这一步有权限建目录)",
        REPORTS_STORE,
        'return _UNSAFE.sub("", (text or "").strip())[:limit]',
        'return (text or "").strip()[:limit]',
        REPORT_TESTS,
        "test_a_topic_can_never_escape_the_report_directory",
    ),
    (
        "同一天多份报告都进时间序列(一周出现两个矛盾的 star 值)",
        REPORTS_STORE,
        "        if item.day is None or item.day in seen:\n            continue",
        "        if item.day is None:\n            continue",
        REPORT_TESTS,
        "test_several_reports_on_one_day_contribute_a_single_point",
    ),
    # ── 工具契约 ────────────────────────────────────────────
    (
        "参数重名放行(模型看到的定义和实际生效的规则不是同一条)",
        SPEC,
        "        if len(names) != len(set(names)):",
        "        if False:",
        TOOL_TESTS,
        "test_duplicate_param_names_are_caught_at_construction",
    ),
    (
        "bool 当整数收(`top_n=true` 静默变成 top_n=1)",
        SPEC,
        "if isinstance(value, bool) or not isinstance(value, (int, float)):",
        "if not isinstance(value, (int, float)):",
        TOOL_TESTS,
        "test_true_is_not_an_integer",
    ),
    (
        "越界值静默裁到边界(模型永远学不会自己传错了)",
        SPEC,
        '            if self.max is not None and number > self.max:\n                return None, f"must_be_lte_{self.max}"',
        "            if self.max is not None and number > self.max:\n                number = self.max",
        TOOL_TESTS,
        "test_out_of_range_is_rejected_rather_than_clamped",
    ),
    (
        "幻觉参数被静默吞掉(模型以为生效了,一直用同样的错法调)",
        SPEC,
        '        errors += [{"param": name, "reason": "unknown_parameter", "received": args[name]}\n'
        "                   for name in sorted(set(args) - known)]",
        "        pass",
        TOOL_TESTS,
        "test_a_hallucinated_param_is_rejected_not_swallowed",
    ),
    (
        "None 默认值也塞进参数(下游每处都要判 None)",
        SPEC,
        "                elif param.default is not None:\n                    clean[param.name] = param.default",
        "                else:\n                    clean[param.name] = param.default",
        TOOL_TESTS,
        "test_a_none_default_does_not_become_a_literal_parameter",
    ),
    (
        "昂贵工具不问就跑(一句话触发几十分钟的全量出榜)",
        RANK_TOOLS,
        'if not (pending and stored.get("mode") == mode and (confirm or pending == signature)):',
        "if False:",
        TOOL_TESTS,
        "test_an_expensive_tool_asks_before_it_runs",
    ),
    (
        "确认后用模型复述的参数而非屏幕上那份(用户确认的和实际跑的不是一回事)",
        RANK_TOOLS,
        '        if "params" in stored:              # mode 已在上面比过\n'
        '            params = stored["params"]',
        "        pass",
        TOOL_TESTS,
        "test_the_second_call_runs_the_parameters_that_were_shown",
    ),
    (
        "多个候选时不看仓库名精确匹配(明确的输入也被打回去问用户)",
        REPO_TOOLS,
        "    if len(exact) == 1:\n        return exact[0][\"full_name\"], None",
        "    pass",
        TOOL_TESTS,
        "test_one_exact_name_match_wins_over_the_other_candidates",
    ),
    # ── 会话历史 ────────────────────────────────────────────
    (
        "压缩切在工具调用中间(留下孤儿 tool 消息,接口一律 400)",
        HISTORY,
        "    old, recent = split_at_safe_boundary(rest, KEEP_RECENT)",
        "    old, recent = rest[:-KEEP_RECENT], rest[-KEEP_RECENT:]",
        AGENT_TESTS,
        "test_compression_never_leaves_an_orphan_tool_message",
    ),
    (
        "摘要拼进 system(每次压缩后前缀缓存全部落空)",
        HISTORY,
        '        rebuilt.append({"role": "user", "content": f"[对话历史摘要]\\n{self.summary}"})',
        '        rebuilt[0] = {"role": "system",\n'
        '                      "content": system["content"] + f"\\n[摘要]{self.summary}"}',
        AGENT_TESTS,
        "test_the_system_message_stays_byte_identical_after_compression",
    ),
    (
        "总结失败就丢掉上一份摘要(压一次少一段记忆)",
        HISTORY,
        "        if text := summarize(old):\n            self.summary = text",
        "        self.summary = summarize(old)",
        AGENT_TESTS,
        "test_a_failed_summary_keeps_the_previous_one_rather_than_losing_it",
    ),
    (
        "大结果留在历史里(之后每一轮都重发一遍,只体现在账单上)",
        HISTORY,
        'if message.get("role") != "tool" or len(content) <= OFFLOAD_THRESHOLD:',
        'if message.get("role") != "tool":',
        AGENT_TESTS,
        "test_a_small_result_is_left_intact",
    ),
    (
        "超长工具结果原样发给模型(一条几万字符把上下文挤爆)",
        HISTORY,
        "    if len(text) <= max_chars:\n        return text",
        "    return text",
        AGENT_TESTS,
        "test_an_enormous_result_is_truncated_before_it_reaches_the_model",
    ),
    # ── 网页端 ──────────────────────────────────────────────
    (
        "扫描器路径回 403(等于告诉对方「这里有东西」)",
        SECURITY,
        'return Verdict(404, "Not Found", "敏感路径")',
        'return Verdict(403, "Forbidden", "敏感路径")',
        WEB_TESTS,
        "test_scanner_paths_get_404_not_403",
    ),
    (
        "限速窗口不滑动(慢速客户端攒够次数后被永久封死)",
        SECURITY,
        "        while window and window[0] < now - RATE_WINDOW:\n            window.popleft()",
        "        pass",
        WEB_TESTS,
        "test_the_window_slides_so_a_slow_client_is_never_blocked",
    ),
    (
        "限速表不分 IP(一个爬虫把所有用户一起限掉)",
        SECURITY,
        'window = _hits.setdefault(ip, collections.deque())',
        'window = _hits.setdefault("all", collections.deque())',
        WEB_TESTS,
        "test_one_noisy_ip_does_not_throttle_everyone_else",
    ),
    (
        "只看 socket 地址不认反代头(全站流量都算在网关那一个 IP 上)",
        SECURITY,
        '    if forwarded := request.headers.get("x-forwarded-for"):\n'
        '        return forwarded.split(",")[0].strip()',
        "    pass",
        WEB_TESTS,
        "test_the_real_ip_is_taken_from_the_proxy_header",
    ),
    (
        "允许 origins=* 配 credentials=true(任何网站都能带着用户 cookie 调接口)",
        SECURITY,
        '    if credentials and "*" in config.CORS_ALLOWED_ORIGINS:',
        "    if False:",
        WEB_TESTS,
        "test_wildcard_origins_and_credentials_cannot_both_be_on",
    ),
    (
        "会话数没上限(随机 session_id 打几万次就能把内存撑爆)",
        SESSIONS,
        "        if len(_agents) >= MAX_SESSIONS:",
        "        if False:",
        WEB_TESTS,
        "test_the_oldest_session_is_evicted_at_the_cap",
    ),
    (
        "取用会话不刷新时间戳(活跃会话被当成最旧的淘汰掉)",
        SESSIONS,
        "            _agents[session_id] = (entry[0], now)\n            return entry[0]",
        "            return entry[0]",
        WEB_TESTS,
        "test_touching_a_session_keeps_it_from_being_the_eviction_victim",
    ),
    (
        "过期会话不清理(一小时前的对话历史一直占着内存)",
        SESSIONS,
        "        for stale in [sid for sid, (_, seen) in _agents.items() if now - seen > TTL_SECONDS]:",
        "        for stale in []:",
        WEB_TESTS,
        "test_an_expired_session_is_swept_away",
    ),
    (
        "删会话不清待发回复(下一个同名会话收到上一个的残留消息)",
        SESSIONS,
        "    _agents.pop(session_id, None)\n    with _pending_lock:\n        _pending.pop(session_id, None)",
        "    _agents.pop(session_id, None)",
        WEB_TESTS,
        "test_dropping_a_session_also_drops_its_stashed_replies",
    ),
    (
        "web 资源名不挡穿越(任意文件读取)",
        RENDER,
        "    if path != root and root not in path.parents:\n        raise FileNotFoundError",
        "    if False:\n        raise FileNotFoundError",
        WEB_TESTS,
        "test_a_web_asset_name_cannot_escape_the_web_directory",
    ),
    (
        "渲染后不过链接白名单(报告里的 javascript: 链接直接可点)",
        RENDER,
        "    article_html = _sanitize_urls(md.convert(text))",
        "    article_html = md.convert(text)",
        WEB_TESTS,
        "test_a_javascript_url_in_a_report_is_defused",
    ),

    # ── 第二轮审查(open-code-review)查出来的那批 ──────────────
    (
        "标签剥离只替换一次(拼接绕过:<scr<script>ipt src=...> 会重新拼成活标签)",
        RENDER,
        "    for _ in range(_MAX_CLEAN_PASSES):",
        "    for _ in range(1):",
        WEB_TESTS,
        "test_no_known_payload_survives_into_the_rendered_report",
    ),
    (
        "URL 白名单只认带引号的属性值(裸写的 href=javascript: 直接穿过)",
        RENDER,
        # 等价于"无引号那一支不进白名单":裸值一律当安全,原样穿过去。
        'value = match.group("quoted") if quote else match.group("bare")',
        'value = match.group("quoted") if quote else "#"',
        WEB_TESTS,
        "test_no_known_payload_survives_into_the_rendered_report",
    ),
    (
        "int 参数不拦非有限浮点(模型传 1e400 → OverflowError 逃出去,会话永久 400)",
        SPEC,
        "            if isinstance(value, float) and not math.isfinite(value):\n"
        "                return None, \"expected_integer\"",
        "            pass",
        TOOL_TESTS,
        "test_a_json_legal_infinity_is_an_error_not_a_crash",
    ),
    (
        "工具异常逃出 run_tool 时不兜底(tool_calls 配不上 tool 回复 → 会话报废)",
        AGENT_LOOP,
        "            try:\n                result = self.run_tool(",
        "            if True:\n                result = self.run_tool(",
        AGENT_TESTS,
        "test_even_a_crash_in_the_validator_still_leaves_a_reply_for_every_call",
    ),
    (
        "快照读取漏掉 zlib.error(压缩体损坏 → 掀翻整轮排名而不是当缺失)",
        STORE / "snapshots.py",
        "    except (OSError, zlib.error, json.JSONDecodeError, EOFError, UnicodeDecodeError) as e:",
        "    except (OSError, json.JSONDecodeError, EOFError) as e:",
        STORE_TESTS,
        "test_a_snapshot_with_a_mangled_body_reads_as_missing_not_as_a_crash",
    ),
    (
        "允许低覆盖快照盖掉高覆盖的(那天的基线永久缺一批仓库)",
        STORE / "snapshots.py",
        "    if (existing := _coverage_of(day)) is not None and existing > coverage:",
        "    if False:",
        STORE_TESTS,
        "test_a_better_snapshot_is_never_overwritten_by_a_worse_one",
    ),
    (
        "prune 不设下限(keep_days=0 一次删光全部快照,且重算不回来)",
        STORE / "snapshots.py",
        "    if keep_days < 1:",
        "    if False:",
        STORE_TESTS,
        "test_pruning_everything_is_refused_rather_than_obeyed",
    ),
    (
        "淘汰不设量级闸门(一次采集事故就能把整个库删空)",
        CRON,
        "    if len(plan.missing) > ceiling:",
        "    if False:",
        SNAP_TESTS,
        "test_an_implausible_number_of_missing_repos_aborts_the_eviction",
    ),
    (
        "单名批的 null 一律当「确认查不到」(清库那条路的入口)",
        GH_CLIENT,
        '        if types != {"NOT_FOUND"}:',
        "        if False:",
        SNAP_TESTS,
        "test_a_lone_null_without_not_found_counts_as_unanswered_not_as_deleted",
    ),
    (
        "token 池不重绑事件循环(client 第二次调用只要有人等锁就永久报废)",
        TOKENS,
        "        self._cond = asyncio.Condition()\n        self._cond_loop = loop",
        "        pass",
        TOKEN_TESTS,
        "test_a_pool_survives_being_used_by_a_second_event_loop",
    ),
    (
        "确认签名不认工具(先请求关键词榜、再拿 confirm=true 调综合榜就能绕过回显)",
        RANK_TOOLS,
        'if not (pending and stored.get("mode") == mode and (confirm or pending == signature)):',
        "if not (pending and (confirm or pending == signature)):",
        TOOL_TESTS,
        "test_confirming_one_ranking_does_not_authorize_a_different_one",
    ),
    (
        "空 keywords 退化成全库排名(用户要关键词榜,拿到的是综合榜)",
        RANK_TOOLS,
        "        logger.warning(\"关键词榜收到空的 keywords,候选池按空处理(不退化成全库排名)。\")\n"
        "        return {}",
        "        return None",
        TOOL_TESTS,
        "test_an_empty_keyword_list_never_degrades_into_a_whole_database_ranking",
    ),
    (
        "上一期按 listing 顺序挑(CI 里 mtime 全一样 → 随机挑一期,推送数字全错)",
        ROOT / "hot_project" / "cron_weekly_report.py",
        "        key=lambda item: item.day, reverse=True,",
        "        key=lambda item: 0,",
        REPORT_TESTS,
        "test_the_previous_issue_is_picked_by_date_not_by_file_mtime",
    ),
    (
        "WebSocket 不过安全检查(唯一真会驱动 agent 的入口反而没有黑名单和限速)",
        ROOT / "hot_project" / "api_server.py",
        "    if verdict := security.check(ip, websocket.url.path):",
        "    if False:",
        WEB_TESTS,
        "test_the_websocket_is_guarded_too_not_just_the_http_routes",
    ),
    (
        "限速表只增不减(伪造 X-Forwarded-For 就能把它撑到几百万条)",
        SECURITY,
        "        if len(_hits) > _SWEEP_THRESHOLD:",
        "        if False:",
        WEB_TESTS,
        "test_spoofed_forwarded_headers_do_not_grow_the_table_forever",
    ),
]


def _clear_pycache() -> None:
    for cache in ROOT.joinpath("hot_project").rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


TEST_TIMEOUT = 30.0


def _run(test_file: str, test_name: str) -> bool:
    """跑一个测试,返回 True 表示它通过了。

    超时算**不通过**:有些变异(比如把 `wait_for` 的 timeout 去掉)的后果就是永久挂死,
    那正是测试要证明的事,只不过表现为「不返回」而不是「断言失败」。少了这个超时,
    整个脚本会卡在那一条上。
    """
    _clear_pycache()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest",
             f"{test_file}::{test_name}", "-q", "--no-header"],
            cwd=ROOT, capture_output=True, text=True, timeout=TEST_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return False
    return proc.returncode == 0


def _baseline_is_green() -> bool:
    """先确认干净的树本来是绿的。

    这一步是被真事教出来的:某次跑到一半被 kill,一处 `if False:` 留在了 `pool.py` 里
    (限流预算被整个关掉),之后两轮变异检查照样报「全部有效」—— 因为每处变异都是拿
    「当时盘上的内容」当原文的,坏掉的那行成了新基准。这次只是碰巧有个变异瞄准同一行,
    才以「对不上原文」暴露出来;泄漏到没人瞄准的行上就永远没人管了。
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "hot_project/tests", "-q", "--no-header", "-x"],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    if proc.returncode == 0:
        return True
    print("工作区本来就是红的,先修好再跑变异检查(常见原因:上次变异没还原干净):")
    print(proc.stdout[-2000:])
    return False


def main() -> int:
    failures: list[str] = []

    if not _baseline_is_green():
        return 1

    # 先把所有要改的文件原文存下来。哪怕脚本被 Ctrl-C 或 kill 掉,`_restore_all` 也能
    # 在 finally 里把它们全部复原 —— 只靠单条的 finally 不够:kill -9 一来,
    # 变异就留在工作区里了(这是实际发生过的)。
    originals = {path: path.read_text(encoding="utf-8") for _, path, *_ in MUTATIONS}

    def _restore_all() -> None:
        for path, text in originals.items():
            if path.read_text(encoding="utf-8") != text:
                path.write_text(text, encoding="utf-8")
        _clear_pycache()

    try:
        failures = _sweep(originals)
    finally:
        _restore_all()

    if failures:
        print("\n变异检查失败:")
        for line in failures:
            print(" ", line)
        return 1

    print(f"\n{len(MUTATIONS)} 处守卫全部确认有效。")
    return 0


def _sweep(originals: dict[Path, str]) -> list[str]:
    failures: list[str] = []

    for label, path, original, broken, test_file, test_name in MUTATIONS:
        source = originals[path]
        if original not in source:
            failures.append(f"[对不上原文] {label} —— {path.name} 里找不到要改的片段")
            continue

        path.write_text(source.replace(original, broken, 1), encoding="utf-8")
        try:
            still_green = _run(test_file, test_name)
        finally:
            path.write_text(source, encoding="utf-8")
            _clear_pycache()

        if still_green:
            failures.append(f"[没拦住] {label} —— 改坏后 {test_name} 仍然通过")
        else:
            print(f"  ok   {test_name}  ←  {label}", flush=True)

    return failures


if __name__ == "__main__":
    raise SystemExit(main())
