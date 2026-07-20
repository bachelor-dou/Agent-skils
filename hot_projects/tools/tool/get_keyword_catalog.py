"""get_keyword_catalog 工具：返回预设搜索关键词分组全表（本地读配置，零成本）。

设计动机：分组词表约 4k 字符，若常驻 system 提示词，对不做关键词榜的对话是纯 token
浪费；改为模型判断需要挑词时按需调用（Claude Code 式的"内容用工具取"）。
"""

from ...config import SEARCH_KEYWORDS


def get_keyword_catalog_handler(ctx, args: dict) -> dict:
    return {
        "categories": SEARCH_KEYWORDS,
        "usage": "从相关分组挑选关键词，并补充分组未覆盖的英文搜索词，一起传给 keyword_ranking 的 keywords 参数。",
    }
