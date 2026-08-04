"""配置纯净性。

1. **import 期不碰磁盘** —— 旧 `config.py:238` 在模块级 `os.makedirs(DATA_DIR)`,
   于是光 import 配置就建目录:测试无法在临时环境里干净导入,CI 里也会在意外位置留下目录。
2. **路径指向真数据** —— 快照漏一天永久补不回,是全项目唯一不可恢复的事故;
   路径算错时读到的是空目录,而下游一律把「空」当「今天还没跑」,于是静默重采/静默零差异。
"""

import subprocess
import sys

from hot_project import config

_PROBE = """
import os, pathlib
def _boom(*a, **k):
    raise AssertionError("import 期创建了目录")
os.mkdir = _boom
os.makedirs = _boom
pathlib.Path.mkdir = _boom

import hot_project.config
print("clean")
"""


def test_import_creates_no_directories():
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=config.PACKAGE_DIR.parent, capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"import 期有磁盘副作用:\n{proc.stderr}"
    assert "clean" in proc.stdout


def test_paths_point_at_the_live_data():
    """路径必须指向本包正在用的真数据 —— 这条同时验证 PACKAGE_DIR 推导没错位。

    路径算错不会报错,只会读到空:每日脚本以为「今天还没写快照」于是重跑全量采集,
    周报以为「没有锚点」于是算不出增长。假绿比崩溃难查得多。
    """
    assert config.DB_PATH.is_file(), f"读不到 DB:{config.DB_PATH}"
    assert config.SNAPSHOT_DIR.is_dir(), f"读不到快照目录:{config.SNAPSHOT_DIR}"
    assert config.DATA_DIR.is_relative_to(config.PACKAGE_DIR), "数据跑到包外面去了"


def test_secrets_are_not_baked_into_the_model_catalog():
    """LLM 平台目录必须只声明 key 所在的环境变量名,不含 key 本身。

    旧代码把 key 塞进每条记录的 `key` 字段,该列表一旦被日志/接口/异常回溯带出去就泄密。
    """
    for entry in config.LLM_MODELS:
        assert "key" not in entry, f"{entry['id']} 的目录条目里出现了 key 字段"
        assert entry["key_env"].endswith("_KEY"), f"{entry['id']} 的 key_env 命名可疑"
