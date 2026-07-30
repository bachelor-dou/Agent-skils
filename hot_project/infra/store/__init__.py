"""store —— 数据层,三份数据共用一个写原语。

    atomic.py      唯一读写原语:排他锁 → 读 → 改 → 原子替换。读不出来就抛,绝不覆盖
    universe.py    Github_DB.json:观测宇宙 + 每个仓库的元信息
    snapshots.py   data/snapshots/*.json.gz:每日 star,全部增长计算的唯一来源
    favorites.py   data/favorites.json:用户收藏

三条贯穿整层的规则:

1. **读不出来就抛,绝不写。** 旧包六处写路径有三种不同的失败处置,其中 `save_db` 和
   `save_db_desc_only` 把读失败当成 `{}` 然后照写 —— 一次 JSON 截断清空 5 万条记录。
   现在只有 `atomic` 一处决定这件事,答案是抛 `StoreReadError`。
2. **每个字段只有一个写入者。** 见 `universe` 里的字段归属表,它是代码里的强制依据而非
   注释:写不属于自己的字段直接抛。旧包只有一个什么都能写的 `save_db`,抹过一次全库 fork 数。
3. **DB 损坏要抛,快照损坏按缺失处理。** 不是不一致,是因为可替代性不同:锚点顺延到邻近
   那天就行,而 DB 无可替代。

`snapshots` 与 `universe` 的分工:快照是「某一天 × 全部仓库的 star」,DB 是「全部仓库 ×
元信息」。前者按天一个 gz 文件,取 T−7 只读 0.8MB;把 `{日期: star}` 挂到项目下是转置
布局,实测会让主库 30MB→83MB、每次保存从 0.8s 涨到 2.7s(全程持锁)。
"""
