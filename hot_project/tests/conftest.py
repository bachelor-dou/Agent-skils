"""让 `async def test_...` 直接可跑,不引入 pytest-asyncio。

旧包的写法是每个异步测试里手写一个内层 `async def run()` 再 `asyncio.run(run())`,
样板压过了内容。这个钩子做同一件事,只是挪到了一处。

不装 pytest-asyncio 的理由:它要装依赖、要配 `asyncio_mode`,而我们需要的全部功能就是
「用 asyncio.run 跑这个协程」六行。
"""

from __future__ import annotations

import asyncio
import inspect


def pytest_pyfunc_call(pyfuncitem):
    func = pyfuncitem.obj
    if not inspect.iscoroutinefunction(func):
        return None                 # 同步测试交还给 pytest 默认处理
    kwargs = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
    asyncio.run(func(**kwargs))
    return True
