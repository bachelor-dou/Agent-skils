"""recall_tool_result 工具：取回之前因体积过大而卸载暂存的工具结果。

历史里体积超阈值的 tool 结果会被 agent 替换为 {offloaded:true, ref:...} 存根，
完整内容暂存在会话 tool_state。模型需要旧结果细节时按 ref 取回（本地读取，零成本）。
"""

import json


def recall_tool_result_handler(ctx, args: dict) -> dict:
    ref = str(args.get("ref", "")).strip()
    store = ctx.state.tool_state.get("offloaded", {})
    raw = store.get(ref)
    if raw is None:
        return {"error": f"未找到暂存结果 {ref}。可用 ref: {sorted(store.keys())}"}
    try:
        result = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        result = raw
    return {"ref": ref, "result": result}
