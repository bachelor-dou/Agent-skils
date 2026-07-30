"""common —— 公共小工具,**零项目知识**。

    timeutil.py   UTC 时钟、GitHub 时间戳与日期串的解析/格式化、年龄计算
    env.py        环境变量解析(逗号分隔、布尔、字符串)

## 成员判据(这条比目录本身重要)

「common」这种名字天生会变成杂物间 —— 因为它没有说清什么东西**不该**进来。这里的判据是:

    它必须完全不认识这个项目。不知道什么是 star、仓库、快照、榜单、报告。
    换成另一个完全不同的项目,整个文件能原样拷走。

所以它**只能 import 标准库**:不许 import `config`,更不许 import 上面任何一层。
由 `tests/test_layering.py::test_common_knows_nothing_about_the_project` 强制 ——
既查 import,也查有没有出现项目词汇。

## 和 infra 的分界

两者都是「被很多地方用」,但判据不同,不重叠:

    common   零项目知识,换个项目能原样拷走       —— 时间、环境变量
    infra    有状态的机制,不知道产品规则          —— 文件锁、任务池、LLM 客户端

拿不准就问:这段代码提到 star 了吗?提到了就不是 common。
"""
