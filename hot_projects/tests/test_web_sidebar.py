"""侧栏折叠/拖拽的前后端契约自检（纯文本断言，不需要浏览器）。

这里守的是三条改起来很容易静默失效的约定：
1. chat.js 里 getElementById 的目标 id 必须在 chat.html 中存在（改名/删元素会静默变成 no-op）；
2. 收起态必须覆盖 grid-template-columns，而不是改 --sidebar-w —— 拖拽宽度是写在元素
   style 上的内联变量，优先级高于类规则，改变量会被内联值盖掉导致收不起来；
3. 收起必须用 visibility 而非 display:none —— display 无法参与宽度过渡动画。
"""

import re
from html.parser import HTMLParser
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "web"
HTML = (WEB / "chat.html").read_text(encoding="utf-8")
CSS = (WEB / "chat.css").read_text(encoding="utf-8")
JS = (WEB / "chat.js").read_text(encoding="utf-8")

SIDEBAR_IDS = [
    "sidebar-toggle", "sidebar-collapse", "sidebar-sash", "sidebar-fav-badge",
]


def test_sidebar_element_ids_exist_in_html():
    for el_id in SIDEBAR_IDS:
        assert f'id="{el_id}"' in HTML, f"chat.html 缺少 id={el_id}，chat.js 会拿到 null"


def test_every_getelementbyid_in_js_has_a_target():
    """JS 取的每个 id 都应在 HTML 里有对应元素（防止改名后静默失效）。

    排除 JS 自己按需创建的元素（如 app-toast 走 `if (!el) create` 的懒创建）。
    """
    created = set(re.findall(r'\.id = "([^"]+)"', JS))
    for el_id in set(re.findall(r'getElementById\("([^"]+)"\)', JS)) - created:
        assert f'id="{el_id}"' in HTML, f"chat.js 引用了不存在的 id={el_id}"


def test_collapsed_overrides_grid_columns_not_the_width_var():
    block = re.search(
        r"\.shell\.sidebar-collapsed\s*\{([^}]*)\}", CSS, re.S
    )
    assert block, "找不到 .shell.sidebar-collapsed 规则"
    body = block.group(1)
    assert "grid-template-columns" in body, "收起态必须直接覆盖 grid-template-columns"
    assert "--sidebar-w" not in body, (
        "收起态不能改 --sidebar-w：拖拽保存的内联变量优先级更高，会盖掉它导致收不起来"
    )


def test_collapse_uses_visibility_so_width_can_animate():
    block = re.search(
        r"\.shell\.sidebar-collapsed\s+\.report-panel\s*\{([^}]*)\}", CSS, re.S
    )
    assert block, "找不到收起态的 .report-panel 规则"
    body = block.group(1)
    assert "visibility: hidden" in body, "需用 visibility:hidden 以便宽度动画可播放"
    assert "display: none" not in body, "display:none 会让宽度过渡失效"


def test_panel_collapse_button_is_visible_on_desktop():
    """标题栏收起按钮在桌面端必须真的显示出来。

    媒体查询不增加优先级，同权重下后写的规则胜出：桌面端的 display:flex 若写在
    基础规则 display:none 之前，按钮会被静默隐藏（曾经就是这样）。
    这里取最后一条声明了 display 的 .panel-collapse 规则，它才是生效的那条。
    """
    blocks = [
        m.group(1)
        for m in re.finditer(r"^\s*\.panel-collapse\s*\{([^}]*)\}", CSS, re.S | re.M)
        if "display:" in m.group(1)
    ]
    assert blocks, "找不到 .panel-collapse 的 display 声明"
    assert "display: flex" in blocks[-1], (
        "最后生效的 .panel-collapse display 不是 flex —— 桌面端收起按钮会不可见"
    )


def test_sash_tracks_sidebar_width():
    """分隔条必须锚在侧栏左沿，否则拖拽时手柄和边界会脱开。"""
    block = re.search(r"^\.sidebar-sash\s*\{([^}]*)\}", CSS, re.S | re.M)
    assert block, "找不到 .sidebar-sash 规则"
    assert "right: var(--sidebar-w)" in block.group(1)


def test_both_toggle_buttons_share_the_accessible_fill():
    """收起/展开是同一个控件的两态，必须同款；且填充色必须是对比度达标的那个。

    --accent (#d7883a) 配白字只有 2.81:1，连大字号 3:1 都不到，
    所以实心按钮一律用 --accent-deep (#b9611d, 4.38:1)。
    """
    assert "--accent-deep:" in CSS, "缺少 --accent-deep 变量"
    for selector in (r"\.sidebar-handle", r"\.panel-collapse"):
        block = re.search(rf"^{selector}\s*\{{([^}}]*)\}}", CSS, re.S | re.M)
        assert block, f"找不到 {selector} 规则"
        body = block.group(1)
        assert "background: var(--accent-deep)" in body, (
            f"{selector} 的填充色应为 --accent-deep（两态同款且对比度达标）"
        )
        assert "color: #fff" in body, f"{selector} 应配白色图标"


def test_absolute_children_live_inside_the_positioned_shell():
    """展开按钮与拖拽条都是 absolute 定位，必须待在 .shell（唯一的定位上下文）内。

    若被挪到 .shell 外面，它们会改以视口为基准定位 —— 页面不报错，只是静默跑偏。
    """
    main = re.search(r'<main class="shell">(.*?)</main>', HTML, re.S)
    assert main, '找不到 <main class="shell"> ... </main>（标签可能没闭合）'
    for el_id in ("sidebar-toggle", "sidebar-sash"):
        assert f'id="{el_id}"' in main.group(1), f"{el_id} 不在 .shell 内，absolute 定位会失效"

    shell = re.search(r"^\.shell\s*\{([^}]*)\}", CSS, re.S | re.M)
    assert shell and "position: relative" in shell.group(1), (
        ".shell 必须是 position: relative，否则上面两个元素会以视口为定位基准"
    )


def test_chat_html_tags_are_balanced():
    """标签配对自检：手工搬动 DOM 时漏掉一个闭合标签是最容易犯又最难发现的错。"""
    void = {"meta", "link", "br", "img", "input", "hr", "source", "path", "circle"}

    class Checker(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack: list[str] = []
            self.errors: list[str] = []

        def handle_starttag(self, tag, attrs):
            if tag not in void:
                self.stack.append(tag)

        def handle_endtag(self, tag):
            if tag in void:
                return
            if not self.stack:
                self.errors.append(f"多余的 </{tag}>")
            elif self.stack[-1] != tag:
                self.errors.append(f"闭合错位：期望 </{self.stack[-1]}>，实际 </{tag}>")
                if tag in self.stack:
                    while self.stack and self.stack.pop() != tag:
                        pass
            else:
                self.stack.pop()

    checker = Checker()
    checker.feed(HTML)
    assert not checker.errors, f"chat.html 标签配对有问题: {checker.errors}"
    assert not checker.stack, f"chat.html 有未闭合标签: {checker.stack}"
