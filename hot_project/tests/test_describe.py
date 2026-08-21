"""service.describe 的接口级测试:注入 stub client,不碰真实 LLM、不 monkeypatch 模块属性。

这些规则(素材进 prompt、序号对齐解析、解析不足整批回退)以前只能靠生产日志发现回归 ——
describe 硬连 llm.get() 时没法从接口喂假响应。client 参数就是为这份文件开的接缝。
"""

from __future__ import annotations

from hot_project.infra import llm
from hot_project.service import describe


class _StubLLM:
    """最小 LLM 替身:记录收到的 prompt,回放固定文本。"""

    def __init__(self, reply: str = "", ok: bool = True) -> None:
        self.reply, self._ok = reply, ok
        self.prompts: list[str] = []
        self.kwargs: list[dict] = []

    def configured(self) -> bool:
        return self._ok

    def text(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        return self.reply


# ──────────────────────────────────────────────────────────
# describe
# ──────────────────────────────────────────────────────────

def test_describe_feeds_facts_into_the_prompt_and_returns_the_reply():
    stub = _StubLLM(reply="项目定位与用途:一个测试项目。")
    facts = {"gh_desc": "vector db", "topics": ["rag", "embedding"],
             "readme_excerpt": "Usage: pip install thing"}

    text = describe.describe("acme/thing", facts, client=stub)

    assert text == "项目定位与用途:一个测试项目。"
    (prompt,) = stub.prompts
    assert "acme/thing" in prompt
    assert "vector db" in prompt
    assert "rag, embedding" in prompt
    assert "pip install thing" in prompt
    assert describe.SECTIONS[0] in prompt, "格式要求必须点名字段,解析端才有的对齐"


def test_prompt_templates_name_exactly_the_sections_the_parser_knows():
    """提示词里的字段名和 `SECTIONS` 必须是同一套:标题在元组里声明一遍、又在
    `_FORMAT` 字符串里手写一遍,只改其中一处,LLM 输出的段名解析端就静默认不出。
    STANDARD 档有意只要前三段;退役的旧段名两档都不许再点名。"""
    for level, wanted in ((describe.DETAILED, describe.SECTIONS),
                          (describe.STANDARD, describe.SECTIONS[:3])):
        text = describe._FORMAT[level]
        for title in wanted:
            assert f"{title}:" in text, f"{level} 提示词丢了字段「{title}」"
        for legacy in describe.LEGACY_SECTIONS:
            assert legacy not in text, f"{level} 提示词还在要求退役段名「{legacy}」"
    assert f"{describe.SECTIONS[3]}:" not in describe._FORMAT[describe.STANDARD], \
        "STANDARD 档只要前三段,第四段冒出来说明两处改岔了"


def test_both_batch_calls_ask_the_model_to_think_at_the_batch_depth():
    """这两处曾经显式关掉思考,总结明显更泛;但它们是批量的(一份报告几十个仓库),
    也不能用对话那一档。中间档是这个取舍的落点,而它只在调用点上,漏改不会有任何报错。"""
    stub = _StubLLM(reply="1. 一个测试项目")
    describe.describe("a/b", {}, client=stub)
    describe.condense([{"full_name": "a/b", "description": "thing"}], client=stub)
    assert [k["effort"] for k in stub.kwargs] == [llm.EFFORT_MEDIUM] * 2


def test_describe_without_a_configured_llm_returns_empty_and_never_calls_it():
    stub = _StubLLM(ok=False)
    assert describe.describe("a/b", {}, client=stub) == ""
    assert stub.prompts == [], "没配 LLM 还发请求,说明守卫失效"


# ──────────────────────────────────────────────────────────
# condense
# ──────────────────────────────────────────────────────────

REPOS = [
    {"full_name": "a/one", "description": "first repo with a long description"},
    {"full_name": "a/two", "description": "second repo"},
]


def test_condense_aligns_replies_by_index_and_caps_length():
    stub = _StubLLM(reply="1. 第一个仓库的浓缩\n2. 第二个仓库的浓缩")

    out = describe.condense(REPOS, max_chars=6, client=stub)

    assert out == ["第一个仓库的", "第二个仓库的"], "按序号对齐并截到 max_chars"
    (prompt,) = stub.prompts
    assert "a/one" in prompt and "a/two" in prompt


def test_condense_falls_back_to_truncation_when_parsing_collapses():
    stub = _StubLLM(reply="抱歉,我无法完成这个任务。")

    out = describe.condense(REPOS, max_chars=10, client=stub)

    assert out == ["first repo", "second rep"], "解析不出一半以上必须整批回退截断原文"


def test_condense_without_a_configured_llm_falls_back_silently():
    stub = _StubLLM(ok=False)
    out = describe.condense(REPOS, max_chars=10, client=stub)
    assert out == ["first repo", "second rep"]
    assert stub.prompts == []
