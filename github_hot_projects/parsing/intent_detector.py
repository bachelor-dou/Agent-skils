"""
意图识别
========
纯函数：从用户消息文本中判断工作流意图。

所有函数接收小写化的用户消息文本，返回布尔值。
"""


def is_new_project_workflow(text: str) -> bool:
    """用户消息表达新项目榜意图时返回 True。

    触发条件：
      1. 直接命中标记词（"新项目榜"、"新仓库热榜"等）
      2. 新项目语义词 + 排名语义词 组合命中
    """
    if not text:
        return False

    markers = (
        "新项目榜",
        "新项目热榜",
        "新项目排名",
        "新仓库榜",
        "新仓库热榜",
        "新创建",
        "新创建的仓库",
        "刚创建",
        " new project",
        " new repo",
        "hot_new",
    )
    if any(marker in text for marker in markers):
        return True

    has_new_project_intent = any(
        marker in text
        for marker in [
            "新项目", "新仓库", "新创建", "刚创建", "new project", "new repo",
        ]
    )
    has_hot_ranking_intent = any(
        marker in text
        for marker in [
            "热榜", "榜单", "排名", "top", "前", "热门", "比较火",
        ]
    )
    return has_new_project_intent and has_hot_ranking_intent


def is_comprehensive_ranking(text: str) -> bool:
    """用户消息明确表达综合榜意图时返回 True。"""
    if not text:
        return False

    direct_markers = (
        "综合热榜",
        "综合榜",
        "综合排名",
        "综合热门",
        "comprehensive",
    )
    if any(marker in text for marker in direct_markers):
        return True

    has_comprehensive_intent = "综合" in text
    has_hot_ranking_intent = any(
        marker in text
        for marker in [
            "热榜", "榜单", "排名", "top", "前", "热门",
        ]
    )
    return has_comprehensive_intent and has_hot_ranking_intent


def is_realtime_refresh(text: str) -> bool:
    """用户消息包含实时强刷语义时返回 True。"""
    if not text:
        return False

    # 强触发词：出现即强制刷新
    hard_triggers = [
        "强制刷新", "立即刷新", "重新跑", "实时热榜",
        "realtime", "force refresh",
    ]
    if any(k in text for k in hard_triggers):
        return True

    # 语义组合触发：明确的当前时态词 + 排名词 + GitHub/项目语境
    has_current_time_intent = any(k in text for k in ["实时", "当前", "现在", "此刻", "最新"])
    has_ranking_intent = any(k in text for k in ["热榜", "榜单", "排名", "top", "前"])
    has_github_intent = any(k in text for k in ["github", "社区", "项目"])
    return has_current_time_intent and has_ranking_intent and has_github_intent
