"""JSON 文件的唯一读写原语 —— 数据层其余模块都只经由这里碰盘。

**读不出来就抛,绝不拿空数据覆盖盘上的东西。** `transaction` 全程持排他锁,读改写在同一
把锁里,没有丢更新的窗口。
不 fsync:os.replace 保证读者要么看到旧文件、要么看到新文件,不会看到半截 —— 半截快照
被当成锚点读走才是这里要防的,掉电丢最后一次写不在威胁模型内。
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_locks: dict[Path, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(path, threading.Lock())


@contextmanager
def _flocked(path: Path, mode: int) -> Iterator[None]:
    """在 `<path>.lock` 上加 fcntl 锁。锁文件必须独立于数据文件 —— 锁跟着 inode 走,
    而 os.replace 会换掉 inode。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock_for(path):
        fd = open(f"{path}.lock", "w")
        try:
            fcntl.flock(fd, mode)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            fd.close()


class StoreReadError(RuntimeError):
    """盘上的数据读不出来(不存在以外的任何原因)。

    单独一个类型,好让调用方区分「文件还没有」(可从默认值起步)和「文件坏了」(绝不能当空数据)。
    """


def _load(path: Path, default: Any | None) -> Any:
    if not path.exists():
        if default is None:
            raise StoreReadError(f"{path} 不存在,且调用方没给默认值")
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        raise StoreReadError(f"{path} 读取失败({e})—— 放弃本次操作,盘上数据保持原样") from e


def _dump(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_json(path: Path, *, default: Any | None = None) -> Any:
    """共享锁读一份 JSON。文件坏了抛 `StoreReadError`;不存在时返回 `default`。"""
    with _flocked(path, fcntl.LOCK_SH):
        return _load(path, default)


class Tx:
    """一次事务。改 `tx.data`,退出上下文时写回;调 `tx.abort()` 则不写。"""

    __slots__ = ("data", "_write")

    def __init__(self, data: Any) -> None:
        self.data = data
        self._write = True

    def abort(self) -> None:
        """本次不写盘。用于「算完发现没有变化」—— 省掉一次整库序列化,也不在 git 里留
        无差异的改动。"""
        self._write = False


@contextmanager
def transaction(path: Path, *, default: Any | None = None) -> Iterator[Tx]:
    """排他锁下的读-改-写,全程不放锁,所以没有丢更新的窗口。

    调用方抛异常 = 放弃写入,盘上原样(异常继续向上传)。
    """
    with _flocked(path, fcntl.LOCK_EX):
        tx = Tx(_load(path, default))
        yield tx
        if tx._write:
            _dump(path, tx.data)


def write_whole(path: Path, write_fn: Callable[[Path], None]) -> None:
    """整份覆盖写(不需要先读),由 `write_fn` 往给它的临时路径里写内容。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with _flocked(path, fcntl.LOCK_EX):
        try:
            write_fn(tmp)
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)   # 半截 .tmp 留着会在下次被当成正常文件读
            raise
