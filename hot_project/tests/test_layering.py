"""分层守卫 —— 用 AST 扫全包的 import,断言没有反向依赖。

这是整个重构里最重要的一个测试:旧包的耦合不是一次写坏的,是几十次「就近 import」
累积出来的(`tools/basic/core.py` 最终 import 了配置、三个存储、token 池、出站 API、
LLM、参数校验、任务池)。人工 review 挡不住这种渐变,所以让它每次跑测试都被挡一次。

规则:上层可依赖下层,下层永不 import 上层;同层互相 import 允许。
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = PACKAGE_ROOT.name

SCRIPTS = "<顶层入口脚本>"

_LAYERS = ["common", "config", "infra", "provider", "service", "tools", "agent", "web"]

ALLOWED: dict[str, set[str]] = {
    name: set(_LAYERS[:i]) for i, name in enumerate(_LAYERS)
}
ALLOWED[SCRIPTS] = set(_LAYERS)
ALLOWED["tests"] = set(_LAYERS) | {SCRIPTS}

CORE_FORBIDDEN = {
    "os", "pathlib", "requests", "aiohttp", "httpx", "urllib", "socket",
    "gzip", "shutil", "tempfile", "sqlite3", "subprocess", "asyncio",
}

PROJECT_WORDS = {
    "star", "stars", "repo", "repos", "repository", "snapshot", "snapshots",
    "ranking", "rank", "favorite", "favorites", "token", "tokens", "github",
    "readme", "trending", "growth", "universe", "db", "llm", "report",
}


def _module_parts(path: Path) -> tuple[list[str], bool]:
    """文件路径 → (模块路径片段, 是否是包的 __init__)。

    两者都要返回,因为相对 import 的解析基准不同:普通模块 `pkg.mod` 里 `from . import x`
    指 `pkg`(去掉自己这一级),而包的 `__init__` 里 `from . import x` 指**这个包自己**
    (不去掉)。混为一谈会让规则整体错位一级,而错位后的守卫仍然「能跑」,只是在拦错东西。
    """
    rel = path.relative_to(PACKAGE_ROOT.parent).with_suffix("")
    parts = list(rel.parts)
    is_pkg = parts[-1] == "__init__"
    if is_pkg:
        parts.pop()
    return parts, is_pkg


def _layer_of(parts: list[str]) -> str | None:
    """模块片段 → 所属层。不在本包内或就是包根 → None。

    `hot_project/api_server.py` 这类顶层脚本不属于任何层,归 SCRIPTS 伪层。
    判据是「第二段不是已知的层名」—— 层都是目录,脚本都是文件,不会重名。
    """
    if len(parts) < 2 or parts[0] != PACKAGE_NAME:
        return None
    name = parts[1]
    return name if name in ALLOWED else SCRIPTS


def _imported_targets(
    tree: ast.Module, parts: list[str], is_pkg: bool
) -> list[tuple[str, int]]:
    """收集本文件 import 到的**包内**模块,返回 [(层名, 行号)]。

    `ImportFrom` 要连 `names` 一起解析,不能只看 `node.module`:`from .. import config`
    里被导入的层名在 alias 上,`node.module` 是空的。漏了这一支等于给守卫开了个后门 ——
    而且是最常用的那个写法(包里几处读配置都是 `from ... import config`)。
    """
    found: set[tuple[str, int]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if target := _layer_of(alias.name.split(".")):
                    found.add((target, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level:                     # 相对 import:按当前模块位置解析
                keep = len(parts) - node.level + (1 if is_pkg else 0)
                base = parts[: max(keep, 0)]
            else:                              # 绝对 import
                base = []
            full = base + (node.module.split(".") if node.module else [])
            candidates = [full] + [full + [alias.name] for alias in node.names]
            for candidate in candidates:
                if target := _layer_of(candidate):
                    found.add((target, node.lineno))
    return sorted(found, key=lambda item: item[1])


def _external_roots(tree: ast.Module) -> list[tuple[str, int]]:
    """收集本文件 import 的**包外**顶层模块名,返回 [(模块名, 行号)]。"""
    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root != PACKAGE_NAME:
                    out.append((root, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if not node.level and node.module:
                root = node.module.split(".")[0]
                if root != PACKAGE_NAME:
                    out.append((root, node.lineno))
    return out


def _docstring_nodes(tree: ast.Module) -> set[int]:
    """收集文档字符串常量节点的 id,便于把它们从字面量检查里排除。"""
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        first = node.body[0] if node.body else None
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                and isinstance(first.value.value, str):
            out.add(id(first.value))
    return out


def _words(text: str) -> set[str]:
    """把标识符或字面量切成小写词。

    切分要按词而非子串比:`startswith` 含子串 "star" 但它显然不是项目词汇,
    而 `min_star` 切出来的 "star" 是。camelCase 也要断开,否则 `minStar` 会漏。
    """
    spaced = "".join(
        f" {ch}" if ch.isupper() else ch for ch in text
    ).replace("_", " ")
    return {w for w in "".join(
        ch if ch.isalnum() or ch == " " else " " for ch in spaced
    ).lower().split() if w}


def _vocabulary(tree: ast.Module) -> list[tuple[str, int]]:
    """收集文件里出现的标识符与非文档字符串字面量,返回 [(词, 行号)]。"""
    skip = _docstring_nodes(tree)
    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        raw: str | None = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            raw = node.name
        elif isinstance(node, ast.Name):
            raw = node.id
        elif isinstance(node, ast.Attribute):
            raw = node.attr
        elif isinstance(node, ast.arg):
            raw = node.arg
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in skip:
            raw = node.value
        if raw:
            out.extend((w, node.lineno) for w in _words(raw))
    return out


def _source_files() -> list[Path]:
    return sorted(p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _parsed() -> list[tuple[Path, list[str], bool, ast.Module]]:
    out = []
    for path in _source_files():
        parts, is_pkg = _module_parts(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        out.append((path, parts, is_pkg, tree))
    return out


def test_every_layer_has_a_rule():
    """新增顶层包必须同时在 ALLOWED 里登记 —— 否则它会静默地不受任何约束。

    这条是守卫的守卫:没有它,以后加个 `hot_project/whatever/` 就绕过了全部规则,
    而测试照样全绿。

    只认**带 `__init__.py` 的目录**:`logs/`、`shadow/`、`data/`、`report/`、`web/`(前端
    静态文件)这些非代码目录会在包里出现,它们不参与分层,按目录名一律计入会让这条测试
    在第一次写日志时就变红。
    """
    actual = {p.name for p in PACKAGE_ROOT.iterdir()
              if p.is_dir() and (p / "__init__.py").exists()}
    unregistered = actual - ALLOWED.keys()
    assert not unregistered, (
        f"这些包没有分层规则:{sorted(unregistered)}。"
        f"在 ALLOWED 里登记它能 import 哪些层,再决定它的位置。"
    )


def test_the_cron_entries_do_not_drag_in_web_only_dependencies():
    """CI 只装 httpx + requests(`requirements-ci.txt`),web 那套一个都不装。

    所以 cron 的导入树里混进 fastapi / markdown 的后果是生产环境启动即崩,而本地什么都
    装了、永远复现不出来。开子进程是必须的:测试进程里这些模块早被别的用例导进来了。
    """
    probe = (
        "import sys, hot_project.cron_weekly_report, hot_project.cron_daily_snapshot;"
        "print(','.join(m for m in ('fastapi','markdown','uvicorn','starlette','pydantic')"
        " if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", probe], cwd=PACKAGE_ROOT.parent,
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"cron 导入树里混进了只有本地才装的包:{out.stdout}"


def test_no_reverse_dependency():
    """下层不许 import 上层。"""
    violations = []
    for path, parts, is_pkg, tree in _parsed():
        layer = _layer_of(parts)
        if layer is None or layer not in ALLOWED:
            continue
        allowed = ALLOWED[layer] | {layer}          # 同层允许
        for target, lineno in _imported_targets(tree, parts, is_pkg):
            if target not in allowed:
                violations.append(
                    f"{path.relative_to(PACKAGE_ROOT)}:{lineno} — "
                    f"{layer} 不该 import {target}"
                )
    assert not violations, "反向依赖:\n" + "\n".join(violations)


def test_growth_stays_pure():
    """service/growth.py 零 I/O:不许 import 网络库、文件系统、子进程、asyncio。

    它的全部价值就是「不用 token、不联网、不碰盘就能跑完整的增长回归测试」。
    一旦有人在里面 `open()` 一次,这个保证就没了,而且没有任何测试会因此变红 ——
    所以得有这一条。
    """
    violations = []
    for path, parts, _is_pkg, tree in _parsed():
        if parts[1:3] != ["service", "growth"]:
            continue
        for root, lineno in _external_roots(tree):
            if root in CORE_FORBIDDEN:
                violations.append(
                    f"{path.relative_to(PACKAGE_ROOT)}:{lineno} — growth 不该 import {root}"
                )
    assert not violations, "growth 被污染:\n" + "\n".join(violations)


def test_common_knows_nothing_about_the_project():
    """common 里不许出现项目词汇 —— 这是它唯一的成员判据,也是它不变成杂物间的唯一保障。

    「common」这种名字没有说清什么**不该**进来,所以判据必须写成可执行的:
    换个完全不同的项目,common 下每个文件都能原样拷走。提到 star 就不行。

    (「不许 import 上层」由 `test_no_reverse_dependency` 顺带保证:common 在 `_LAYERS`
    第一位,允许集合为空,所以它连 config 都不能 import。)
    """
    violations = []
    for path, parts, _is_pkg, tree in _parsed():
        if _layer_of(parts) != "common":
            continue
        for word, lineno in _vocabulary(tree):
            if word in PROJECT_WORDS:
                violations.append(
                    f"{path.relative_to(PACKAGE_ROOT)}:{lineno} — "
                    f"common 出现项目词汇 {word!r}"
                )
    assert not violations, (
        "common 认识这个项目了,说明这段代码该去 core(有产品规则、零 I/O)"
        "或 infra(有状态的机制):\n" + "\n".join(sorted(set(violations)))
    )


@pytest.mark.parametrize("layer", sorted(ALLOWED))
def test_rules_reference_real_layers(layer):
    """ALLOWED 里写的层必须真实存在 —— 防止改名后规则悄悄失效。

    层可以是目录(core/ infra/ ...)也可以是单个文件(config.py),两种都认。
    """
    for target in ALLOWED[layer]:
        if target == SCRIPTS:      # 伪层,对应一组顶层脚本而非某个具体路径
            continue
        exists = (PACKAGE_ROOT / target).is_dir() or (PACKAGE_ROOT / f"{target}.py").is_file()
        assert exists, f"{layer} 的规则引用了不存在的层 {target}"
