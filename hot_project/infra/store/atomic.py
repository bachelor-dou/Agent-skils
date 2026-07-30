"""JSON 文件的唯一读写原语 —— 数据层其余模块都只经由这里碰盘。

## 为什么要有这一层

旧包把「开锁文件 → flock → 读 → 改 → 写 .tmp → os.replace → 解锁」这套手写了六遍
(`db.py` 四遍、`favorites_store.py` 一遍、`snapshots.py` 一遍),于是同一件事有了三种
互不相同的失败处置,其中两种是数据事故:

- `save_db` / `save_db_desc_only` / `favorites._read_all`:读盘失败 → 当成 `{}` → **照写**。
  一次 JSON 截断就把 5 万条记录清空,而且日志只是一行 warning,看起来是成功的。
- `insert_new_projects` / `evict_stale_projects`:读盘失败 → 放弃写入。这个是对的。

本模块只留后一种:**读不出来就抛,绝不拿空数据覆盖盘上的东西。**

## 还修掉了一个丢更新的竞态

旧 `set_favorite` 是 `_read_all()`(拿共享锁,读完**放锁**)→ 改 → `_write_all()`(再拿排他锁)。
两个并发的收藏请求会各读到同一份旧数据,后写的那个把前一个的收藏抹掉。`transaction`
全程持排他锁,读改写在同一把锁里。

## 不做的事

不 fsync。os.replace 保证的是「读者要么看到旧文件、要么看到新文件,不会看到半截」,
这正是我们要防的(半截快照被当成锚点读走)。掉电丢最后一次写不是这里的威胁模型 ——
数据每天由 CI 重新产出并进 git。
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

# 每个路径一把进程内锁。fcntl 是**进程间**互斥,同进程内多线程拿同一个 fd 不会互相阻塞,
# 所以两层都需要:threading.Lock 挡住本进程的线程,flock 挡住 CI 里并行的另一个进程。
_locks: dict[Path, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(path, threading.Lock())


@contextmanager
def _flocked(path: Path, mode: int) -> Iterator[None]:
    """在 `<path>.lock` 上加 fcntl 锁。锁文件独立于数据文件,这样 os.replace 换掉数据文件
    时锁不会跟着失效(替换会换掉 inode,锁跟着 inode 走)。"""
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

    单独一个类型,是为了让调用方能区分「文件还没有」和「文件坏了」:前者可以从默认值起步,
    后者必须停下来让人看一眼,绝不能当空数据继续走。
    """


def _load(path: Path, default: Any | None) -> Any:
    if not path.exists():
        if default is None:
            raise StoreReadError(f"{path} 不存在,且调用方没给默认值")
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise StoreReadError(f"{path} 读取失败({e})—— 放弃本次操作,盘上数据保持原样") from e


def _dump(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


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
        """本次不写盘。用于「算完发现没有变化」—— 主库 30MB,白写一次序列化要 0.8 秒,
        而且会在 git 里留下一个无差异的改动。"""
        self._write = False


@contextmanager
def transaction(path: Path, *, default: Any | None = None) -> Iterator[Tx]:
    """排他锁下的读-改-写。全程不放锁,所以没有丢更新的窗口。

    调用方抛异常 = 放弃写入,盘上原样(异常继续向上传)。
    """
    with _flocked(path, fcntl.LOCK_EX):
        tx = Tx(_load(path, default))
        yield tx
        if tx._write:
            _dump(path, tx.data)


def write_whole(path: Path, write_fn: Callable[[Path], None]) -> None:
    """整份覆盖写(不需要先读),由 `write_fn` 往给它的临时路径里写内容。

    快照走这条:它是 gzip 而非 JSON,而且每天是全新一份、没有读-改-写。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with _flocked(path, fcntl.LOCK_EX):
        try:
            write_fn(tmp)
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)   # 半截 .tmp 留着会在下次被当成正常文件读
            raise
