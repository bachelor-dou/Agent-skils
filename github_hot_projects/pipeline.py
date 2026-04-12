"""
核心流程编排模块
================
将搜索、增长计算、排序、报告生成串联为完整 pipeline。

流程架构：
  Phase 1 - 统一收集：关键词搜索 + Star 范围扫描 + GitHub Trending → 汇入同一 raw_repos map（去重 + star >= 1000）
  Phase 2 - 批量增长计算：对 raw_repos 中全部仓库并行计算窗口期增长 → 筛出候选
  Phase 3 - 评分排序 + 截取 Top N
  Phase 4 - LLM 描述 + 报告生成

主要函数：
  - collect_from_keyword_search()  — 关键词搜索收集
  - collect_from_star_range()      — Star 范围扫描收集
  - collect_from_trending()        — GitHub Trending 收集（周范围直接判断,月范围走增长计算）
  - batch_growth_calc()            — 统一批量增长计算
  - step2_rank_and_select()        — 评分排序 + 截取 Top N
  - step3_generate_report()        — LLM 描述 + Markdown 报告
  - main()                         — 入口编排
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from .config import (
    GITHUB_TOKENS,
    HOT_PROJECT_COUNT,
    LLM_API_URL,
    LLM_MODEL,
    MIN_STAR_FILTER,
    REPORT_DIR,
    DEFAULT_SCORE_MODE,
    SEARCH_REQUEST_INTERVAL,
    STAR_GROWTH_THRESHOLD,
    STAR_RANGE_MAX,
    STAR_RANGE_MIN,
    TIME_WINDOW_DAYS,
    DATA_EXPIRE_DAYS,
    RATE_LIMIT_WAIT_INTERVAL,
    SEARCH_KEYWORDS,
)
from .db import load_db, save_db, update_db_project
from .github_api import auto_split_star_range, search_github_repos
from .github_trending import fetch_trending
from .growth_estimator import estimate_star_growth_binary
from .llm import call_llm_describe
from .token_manager import TokenManager

logger = logging.getLogger("discover_hot")


# ══════════════════════════════════════════════════════════════
# 辅助：单仓库增长计算 worker
# ══════════════════════════════════════════════════════════════


def _process_repo_growth(
    token_mgr: TokenManager,
    full_name: str,
    current_star: int,
    repo_item: dict,
) -> tuple[str, int, int]:
    """
    计算单个仓库的窗口期 star 增长（线程安全 worker）。

    仅负责二分法/采样外推路径。DB 差值法由 batch_growth_calc 主线程处理。
    DB 更新也在主线程完成，此函数不修改 DB（线程安全）。

    Returns:
        (full_name, growth, current_star)
    """
    parts = full_name.split("/", 1)
    if len(parts) != 2:
        return full_name, -1, current_star
    owner, repo_name = parts
    logger.info(f"  [SEARCH] stargazers 查询: {full_name} (star={current_star})")
    growth = estimate_star_growth_binary(token_mgr, owner, repo_name, current_star)

    # 钳位：窗口期增长不可能超过总 star 数
    if growth > current_star:
        growth = current_star

    return full_name, growth, current_star


# ══════════════════════════════════════════════════════════════
# Phase 1: 统一收集阶段（只做搜索 + 去重 + star >= 1000 过滤）
# ══════════════════════════════════════════════════════════════


def collect_from_keyword_search(
    token_mgr: TokenManager,
    raw_repos: dict[str, dict],
) -> None:
    """
    关键词搜索收集：遍历 SEARCH_KEYWORDS，逐关键词搜索，汇入 raw_repos。

    只做搜索和过滤，不计算增长。
    raw_repos: {full_name: {"star": int, "repo_item": dict, "created_at": str}}
    """
    total_keywords = sum(len(kws) for kws in SEARCH_KEYWORDS.values())
    keyword_idx = 0

    for category, keywords in SEARCH_KEYWORDS.items():
        for keyword in keywords:
            keyword_idx += 1
            logger.info(f"[{keyword_idx}/{total_keywords}] 搜索: '{keyword}' (类别: {category})")

            for page in range(1, 4):
                items = search_github_repos(token_mgr, keyword, page=page)
                if not items:
                    break
                for repo_item in items:
                    full_name = repo_item.get("full_name", "")
                    if not full_name or full_name in raw_repos:
                        continue
                    current_star = repo_item.get("stargazers_count", 0)
                    if current_star < MIN_STAR_FILTER:
                        continue
                    raw_repos[full_name] = {
                        "star": current_star,
                        "repo_item": repo_item,
                        "created_at": repo_item.get("created_at", ""),
                    }
                time.sleep(SEARCH_REQUEST_INTERVAL)
            time.sleep(SEARCH_REQUEST_INTERVAL)

    logger.info(f"关键词搜索完成: raw_repos 累计 {len(raw_repos)} 个。")


def collect_from_star_range(
    token_mgr: TokenManager,
    raw_repos: dict[str, dict],
) -> None:
    """
    Star 范围扫描收集：自动分段 + 逐子区间搜索，汇入 raw_repos。
    """
    logger.info(f"Star 范围扫描 ({STAR_RANGE_MIN}..{STAR_RANGE_MAX}) 开始")

    segments = auto_split_star_range(token_mgr, STAR_RANGE_MIN, STAR_RANGE_MAX)
    logger.info(
        f"自动分段完成，共 {len(segments)} 个子区间: "
        + ", ".join(f"[{lo}..{hi}]" for lo, hi in segments)
    )

    for seg_idx, (low, high) in enumerate(segments, 1):
        query = f"stars:{low}..{high}"
        logger.info(f"  子区间 {seg_idx}/{len(segments)}: {query}")

        for page in range(1, 11):
            items = search_github_repos(
                token_mgr, query, page=page, sort="updated", auto_star_filter=False
            )
            if not items:
                break
            for repo_item in items:
                full_name = repo_item.get("full_name", "")
                if not full_name or full_name in raw_repos:
                    continue
                current_star = repo_item.get("stargazers_count", 0)
                if current_star < MIN_STAR_FILTER:
                    continue
                raw_repos[full_name] = {
                    "star": current_star,
                    "repo_item": repo_item,
                    "created_at": repo_item.get("created_at", ""),
                }
            time.sleep(SEARCH_REQUEST_INTERVAL)

    logger.info(f"Star 范围扫描完成: raw_repos 累计 {len(raw_repos)} 个。")


def collect_from_trending(
    raw_repos: dict[str, dict],
    candidate_map: dict[str, dict],
) -> None:
    """
    GitHub Trending 收集。

    策略：
      - daily/weekly 范围：爬取 Trending 页面获取 stars_today（实际为本日/本周增长星数）。
        对于 weekly 范围，stars_today 可直接作为增长指标 ——
        若 stars_today >= STAR_GROWTH_THRESHOLD，直接加入候选（无需查窗口期增长）。
      - monthly 范围：stars_today 为本月增长，粒度太粗，仍需走正常增长计算。

    Trending 仓库如果不满足直接入选条件，则加入 raw_repos 走后续批量增长计算。
    """
    logger.info("GitHub Trending 收集开始")

    trending_repos: list[dict] = []
    for since in ("daily", "weekly", "monthly"):
        repos = fetch_trending(since=since)
        trending_repos.extend(repos)

    # 去重（保留 stars_today 更大的）
    unique: dict[str, dict] = {}
    for r in trending_repos:
        fn = r["full_name"]
        if fn not in unique:
            unique[fn] = r
        else:
            if r.get("stars_today", 0) > unique[fn].get("stars_today", 0):
                unique[fn] = r

    direct_count = 0
    raw_count = 0

    for fn, r in unique.items():
        star = r.get("star", 0)
        if star < MIN_STAR_FILTER:
            continue

        stars_today = r.get("stars_today", 0)
        since = r.get("since", "")

        # weekly 范围：stars_today 是一周增长星数，与 10 天窗口期接近，可直接判断
        # daily 范围的 stars_today 只代表当天增长，粒度太细，不直接入选
        if since == "weekly" and stars_today >= STAR_GROWTH_THRESHOLD:
            if fn not in candidate_map:
                candidate_map[fn] = {
                    "growth": stars_today,
                    "star": star,
                    "created_at": "",
                    "source": f"trending-{since}",
                }
                direct_count += 1
                logger.info(
                    f"  [OK] Trending 直接入选({since}): {fn} | "
                    f"stars_today={stars_today} | star={star}"
                )
            continue

        # 其他（monthly 或 stars_today 不够）→ 加入 raw_repos 走增长计算
        if fn not in raw_repos and fn not in candidate_map:
            repo_item = {
                "full_name": fn,
                "stargazers_count": star,
                "forks_count": r.get("forks", 0),
                "description": r.get("description", ""),
                "language": r.get("language", ""),
                "topics": [],
            }
            raw_repos[fn] = {
                "star": star,
                "repo_item": repo_item,
                "created_at": "",
            }
            raw_count += 1

    logger.info(
        f"Trending 收集完成: {len(unique)} 个仓库, "
        f"直接入选 {direct_count} 个, 加入待计算 {raw_count} 个, "
        f"候选 {len(candidate_map)} 个, raw_repos 累计 {len(raw_repos)} 个。"
    )


# ══════════════════════════════════════════════════════════════
# Phase 2: 统一批量增长计算
# ══════════════════════════════════════════════════════════════


def batch_growth_calc(
    token_mgr: TokenManager,
    raw_repos: dict[str, dict],
    db: dict,
    candidate_map: dict[str, dict],
) -> None:
    """
    对 raw_repos 中全部仓库并行计算窗口期增长，满足阈值的加入 candidate_map。

    DB 差值法仓库在主线程处理（无 IO），其余提交线程池。
    已在 candidate_map 中的仓库跳过（如 Trending 直接入选的）。
    """
    db_valid = db.get("valid", False)
    db_projects = db.get("projects", {})

    # 过滤掉已在候选中的仓库
    pending = {
        fn: info for fn, info in raw_repos.items()
        if fn not in candidate_map
    }

    logger.info(f"批量增长计算: {len(pending)} 个仓库待计算（跳过已入选 {len(raw_repos) - len(pending)} 个）。")

    with ThreadPoolExecutor(max_workers=len(GITHUB_TOKENS)) as executor:
        futures = {}
        for full_name, info in pending.items():
            current_star = info["star"]
            repo_item = info["repo_item"]
            created_at = info.get("created_at", "")

            if full_name in db_projects and db_valid:
                # DB 差值法：主线程直接计算
                saved_star = db_projects[full_name].get("star", 0)
                growth = current_star - saved_star
                update_db_project(db_projects, full_name, current_star, repo_item)
                if growth >= STAR_GROWTH_THRESHOLD:
                    _upsert_candidate(candidate_map, full_name, growth, current_star, created_at, "DB")
            else:
                f = executor.submit(
                    _process_repo_growth, token_mgr, full_name, current_star, repo_item
                )
                futures[f] = (full_name, created_at, repo_item)

        for f in as_completed(futures):
            full_name, created_at, repo_item = futures[f]
            try:
                _, growth, current_star = f.result()
            except Exception as e:
                logger.error(f"  增长计算异常: {full_name}, {e}")
                continue
            if growth < 0:
                continue
            # DB 更新在主线程完成（线程安全）
            update_db_project(db_projects, full_name, current_star, repo_item)
            if growth >= STAR_GROWTH_THRESHOLD:
                _upsert_candidate(candidate_map, full_name, growth, current_star, created_at)

    logger.info(f"批量增长计算完成: 候选总数 {len(candidate_map)} 个。")


# ══════════════════════════════════════════════════════════════
# Step 2: 评分排序
# ══════════════════════════════════════════════════════════════


def step2_rank_and_select(
    candidate_map: dict[str, dict],
    mode: str = DEFAULT_SCORE_MODE,
) -> list[tuple[str, dict]]:
    """
    Step 2: 评分排序 + 截取 Top N。

    评分模式：
      comprehensive — 综合排名：log(增长量) + log(增长率)，新项目平滑折扣
      hot_new       — 新项目专榜：仅创建时间 <= NEW_PROJECT_DAYS 天的项目，按增长量排序

    Returns:
        [(full_name, {"growth": int, "star": int, ...}), ...] 按 score 降序，最多 HOT_PROJECT_COUNT 个。
    """
    import math
    from .config import NEW_PROJECT_DAYS

    def _calc_score(item: dict) -> float:
        g = item["growth"]
        s = item["star"]

        if s <= 0:
            return float(g)

        # 增长量（对数压缩极端值）
        growth_score = math.log(1 + g) * 1000
        # 增长率（对数衰减，避免新项目固定加分）
        rate = g / s
        rate_score = math.log(1 + rate) / math.log(2) * 3000

        if mode == "comprehensive":
            # 新项目平滑折扣：rate > 0.5 开始线性衰减，rate=1.0 时 0.85 折
            if rate > 0.5:
                discount = 1.0 - 0.15 * min((rate - 0.5) / 0.5, 1.0)
            else:
                discount = 1.0
            return (growth_score + rate_score) * discount
        else:
            # 兜底：纯增长量
            return float(g)

    def _is_new_project(info: dict) -> bool:
        """判定是否为新项目：创建时间距今 <= NEW_PROJECT_DAYS 天。"""
        created_at = info.get("created_at", "")
        if not created_at:
            return False
        try:
            created_date = datetime.strptime(
                created_at[:10], "%Y-%m-%d"
            ).replace(tzinfo=timezone.utc)
            days_since = (datetime.now(timezone.utc) - created_date).days
            return days_since <= NEW_PROJECT_DAYS
        except (ValueError, TypeError):
            return False

    if mode == "hot_new":
        # 新项目专榜：仅筛选创建时间 <= 45 天的新项目
        new_projects = {
            name: info for name, info in candidate_map.items()
            if _is_new_project(info)
        }
        # 新项目按增长量排序
        sorted_candidates = sorted(
            new_projects.items(),
            key=lambda x: x[1]["growth"],
            reverse=True,
        )
        top_n = sorted_candidates[:HOT_PROJECT_COUNT]
        logger.info(
            f"Step 2 (hot_new): 新项目(<={NEW_PROJECT_DAYS}天) {len(new_projects)} 个 → 取前 {len(top_n)} 个。"
        )
    else:
        sorted_candidates = sorted(
            candidate_map.items(), key=lambda x: _calc_score(x[1]), reverse=True
        )
        top_n = sorted_candidates[:HOT_PROJECT_COUNT]
        logger.info(
            f"Step 2 (comprehensive): 候选 {len(candidate_map)} → 取前 {len(top_n)} 个。"
        )

    logger.info("  Top 10 预览:")
    for i, (name, info) in enumerate(top_n[:10], 1):
        score = _calc_score(info)
        logger.info(
            f"    {i}. {name} (+{info['growth']}, star={info['star']}, score={score:.0f})"
        )

    return top_n


# ══════════════════════════════════════════════════════════════
# Step 3: LLM 描述 + 报告生成
# ══════════════════════════════════════════════════════════════


def step3_generate_report(
    top_projects: list[tuple[str, dict]], db: dict
) -> str:
    """
    Step 3: 为 Top N 项目生成 LLM 描述 + 输出 Markdown 报告。

    流程：
      1. 筛选 desc 为空的项目
      2. ThreadPoolExecutor 并行调用 LLM
      3. LLM 描述写回 DB projects[repo].desc
      4. 写入 report/YYYY-MM-DD.md

    Returns:
        报告文件路径。
    """
    os.makedirs(REPORT_DIR, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    report_path = os.path.join(REPORT_DIR, f"{today}.md")
    db_projects = db.get("projects", {})

    # ── 并行 LLM 调用 ──
    need_llm: list[tuple[int, str, str, dict]] = []
    desc_results: dict[str, str] = {}

    for idx, (full_name, info) in enumerate(top_projects):
        saved = db_projects.get(full_name, {})
        existing_desc = saved.get("desc", "")
        html_url = f"https://github.com/{full_name}"
        if existing_desc:
            desc_results[full_name] = existing_desc
        else:
            need_llm.append((idx + 1, full_name, html_url, saved))

    if need_llm:
        logger.info(f"Step 3: 需要生成描述 {len(need_llm)} 个项目，并行调用 LLM...")

        def _llm_worker(task: tuple) -> tuple[str, str]:
            idx, full_name, html_url, saved = task
            logger.info(f"[{idx}/{len(top_projects)}] LLM 生成描述: {full_name}")
            desc = call_llm_describe(full_name, saved, html_url)
            return full_name, desc

        workers = min(len(GITHUB_TOKENS), len(need_llm))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for full_name, desc in executor.map(_llm_worker, need_llm):
                if desc:
                    desc_results[full_name] = desc
                    if full_name in db_projects:
                        db_projects[full_name]["desc"] = desc
                else:
                    desc_results.setdefault(full_name, "")

    # ── 生成报告 ──
    lines: list[str] = [f"# GitHub 热门项目 — {today}\n"]
    lines.append(
        f"> 共 {len(top_projects)} 个项目 | "
        f"窗口期: {TIME_WINDOW_DAYS} 天 | "
        f"增长阈值: >={STAR_GROWTH_THRESHOLD} stars | "
        f"最低 star: >={MIN_STAR_FILTER}\n"
    )

    for idx, (full_name, info) in enumerate(top_projects, 1):
        growth = info["growth"]
        star = info["star"]
        html_url = f"https://github.com/{full_name}"
        detailed_desc = desc_results.get(full_name, "")

        lines.append(f"## {idx}. {full_name}（+{growth}，total {star}）\n")
        lines.append(f"链接: {html_url}\n")
        lines.append(f"{detailed_desc}\n")
        lines.append("---\n")

    report_content = "\n".join(lines)
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
    except IOError as e:
        logger.error(f"报告写入失败: {report_path}, {e}")
        return ""

    logger.info(f"Step 3 完成: 报告已写入 {report_path}")
    return report_path


# ══════════════════════════════════════════════════════════════
# 内部辅助
# ══════════════════════════════════════════════════════════════


def _upsert_candidate(
    candidate_map: dict[str, dict],
    full_name: str,
    growth: int,
    current_star: int,
    created_at: str = "",
    source: str = "",
) -> None:
    """更新或插入候选（取更大的 growth 值），保留 created_at。"""
    existing = candidate_map.get(full_name)
    if existing:
        if growth > existing["growth"]:
            existing["growth"] = growth
            existing["star"] = current_star
        if created_at and not existing.get("created_at"):
            existing["created_at"] = created_at
    else:
        candidate_map[full_name] = {
            "growth": growth,
            "star": current_star,
            "created_at": created_at,
        }
        tag = f"({source})" if source else ""
        logger.info(f"  [OK] 候选{tag}: {full_name} | growth={growth} | star={current_star}")


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════


def main() -> None:
    """完整执行流程的入口函数。"""
    token_mgr = TokenManager()
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("GitHub 热门项目发现 — 开始执行")
    logger.info(f"  Token 数量: {len(token_mgr.tokens)}")
    logger.info(f"  LLM: {LLM_API_URL} / {LLM_MODEL}")
    logger.info(f"  时间窗口: {TIME_WINDOW_DAYS} 天")
    logger.info(f"  增长阈值: >={STAR_GROWTH_THRESHOLD}")
    logger.info(f"  最低 star: >={MIN_STAR_FILTER}")
    logger.info(f"  热门数量: {HOT_PROJECT_COUNT}")
    logger.info(f"  DB 过期天数: {DATA_EXPIRE_DAYS}")
    logger.info(f"  搜索间隔: {SEARCH_REQUEST_INTERVAL}s")
    logger.info(f"  限流等待间隔: {RATE_LIMIT_WAIT_INTERVAL}s")
    logger.info(f"  Star 范围扫描: {STAR_RANGE_MIN}..{STAR_RANGE_MAX}")
    logger.info(f"  评分模式: {DEFAULT_SCORE_MODE}")
    logger.info(f"  并行 workers: {len(GITHUB_TOKENS)}")
    logger.info("=" * 60)

    # ── 1. 加载 DB ──
    db = load_db()
    logger.info(
        f"DB 状态: valid={db['valid']}, "
        f"已有 {len(db.get('projects', {}))} 个项目, "
        f"上次更新: {db.get('date', '从未')}"
    )

    # ── Phase 1: 统一收集（去重 + star >= 1000，不计算增长） ──
    raw_repos: dict[str, dict] = {}  # {full_name: {"star": int, "repo_item": dict, "created_at": str}}
    candidate_map: dict[str, dict] = {}  # Trending 直接入选的在此

    collect_from_keyword_search(token_mgr, raw_repos)
    collect_from_star_range(token_mgr, raw_repos)
    collect_from_trending(raw_repos, candidate_map)

    logger.info(
        f"Phase 1 收集完成: raw_repos {len(raw_repos)} 个, "
        f"Trending 直接入选 {len(candidate_map)} 个。"
    )

    # ── Phase 2: 统一批量增长计算 ──
    batch_growth_calc(token_mgr, raw_repos, db, candidate_map)

    # ── 中间落盘 ──
    save_db(db)
    logger.info("Phase 2 后中间落盘完成。")

    if not candidate_map:
        logger.warning("未找到任何满足增长阈值的候选项目。")
        db["valid"] = True
        save_db(db)
        elapsed = time.time() - start_time
        logger.info(f"无候选项目，DB 基线已更新。耗时: {elapsed:.1f}s")
        return

    # ── Phase 3: 排序取 Top N ──
    top_projects = step2_rank_and_select(candidate_map)

    # ── Phase 4: 生成报告 ──
    report_path = step3_generate_report(top_projects, db)

    # ── 5. 最终落盘 ──
    db["valid"] = True
    save_db(db)

    # ── 6. 统计摘要 ──
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("执行完成！")
    logger.info(f"  候选仓库数: {len(candidate_map)}")
    logger.info(f"  热门项目数: {len(top_projects)}")
    logger.info(f"  报告路径: {report_path}")
    logger.info(f"  DB 项目总数: {len(db.get('projects', {}))}")
    logger.info(f"  总耗时: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    logger.info("=" * 60)
