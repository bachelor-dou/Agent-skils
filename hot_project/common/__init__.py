"""common —— 公共小工具,**零项目知识**。

    timeutil.py   UTC 时钟、GitHub 时间戳与日期串的解析/格式化、年龄计算
    env.py        环境变量解析(逗号分隔、布尔、字符串)

判据:它必须完全不认识这个项目(不知道什么是 star、仓库、快照、榜单),换个项目整个文件能
原样拷走。所以**只能 import 标准库**,连 `config` 都不许;有状态的机制归 `infra`。
由 `tests/test_layering.py` 强制,既查 import,也查有没有出现项目词汇。
"""
