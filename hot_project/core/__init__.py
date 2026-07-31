"""core —— 纯算法层,零 I/O。

进来的门槛是「隔这一层能买到东西」:多个调用方共用同一套算术(`growth.py`、`report_parse.py`),
或这段逻辑历史上被 I/O 污染过、需要守卫顶住(`scoring.py`)。其余一律写在调用方里。

**契约**:只能 import `config`、`common` 和标准库里的纯计算部分。不许 import
`infra` / `provider` / `tools`,也不许联网或做文件读写。由 `tests/test_layering.py` 守卫。
"""
