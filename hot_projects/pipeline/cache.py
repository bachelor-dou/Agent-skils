"""榜单分阶段缓存：按 (阶段, 参数签名) 缓存输出；上游变化使下游失效。

阶段顺序与依赖（昂贵的在前，廉价的在后）：
  collect       ← categories, min_star, max_star, days_since_created, mode/sources
  growth_calc   ← collect 输出 + growth_calc_days, days_since_created      (昂贵: API)
  threshold     ← growth_calc 输出 + growth_threshold                       (廉价: 纯过滤)
  rank          ← threshold 输出 + mode, top_n, days_since_created          (廉价)
  report        ← rank 输出 + 展示参数
"""

import json

STAGE_ORDER = ["collect", "growth_calc", "threshold", "rank", "report"]


def _sig(params: dict) -> str:
    return json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)


class RankingCache:
    """会话级榜单缓存。get 命中需阶段名 + 参数签名一致；set 会失效所有下游阶段。"""

    def __init__(self) -> None:
        self._store: dict[str, tuple[str, object]] = {}
        # 会话级旁路缓存：不随阶段失效。用于缓存"按项目"的稳定事实（如各候选最近 K 天增长），
        # 使阈值/top_n 等下游参数变化时无需对已算过的项目重复发 API。
        self.aux: dict[str, dict] = {}

    def get(self, stage: str, params: dict):
        entry = self._store.get(stage)
        if entry is None or entry[0] != _sig(params):
            return None
        return entry[1]

    def set(self, stage: str, params: dict, payload) -> None:
        self._store[stage] = (_sig(params), payload)
        idx = STAGE_ORDER.index(stage)
        for downstream in STAGE_ORDER[idx + 1:]:
            self._store.pop(downstream, None)
