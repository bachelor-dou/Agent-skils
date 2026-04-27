"""
全局配置模块
============
集中管理所有可调参数：GitHub Token、LLM 接口、阈值、评分权重、路径等。
修改配置只需编辑此文件或通过环境变量覆盖。
"""

import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]

# ──────────────────────────────────────────────────────────────
# GitHub Token（最多 N 个，轮换使用，用完一个自动切下一个）
#   从环境变量 GITHUB_TOKENS（逗号分隔）读取；
#   未设置时保持为空，由运行入口决定是否退出。
# ──────────────────────────────────────────────────────────────
_env_tokens = os.environ.get("GITHUB_TOKENS", "")
GITHUB_TOKENS: list[str] = (
    [t.strip() for t in _env_tokens.split(",") if t.strip()]
    if _env_tokens
    else []
)

# ──────────────────────────────────────────────────────────────
# LLM 双模型配置（兼容 OpenAI /v1/chat/completions 格式）
#   主模型：高智能推理（Agent 核心）    辅助模型：低成本文本处理
#   两者可配置不同平台、账号、模型；辅助模型未设置时降级到主模型。
# ──────────────────────────────────────────────────────────────
# 主模型配置（Agent 推理 / 用户交互 / ReAct 工具调用）
#   用于理解用户意图、规划执行链、生成最终回复等高智能任务。
#   环境变量：LLM_API_URL / LLM_API_KEY / LLM_MODEL
# ──────────────────────────────────────────────────────────────
LLM_API_URL: str = os.environ.get(
    "LLM_API_URL", "https://api.siliconflow.cn/v1/chat/completions"
)
LLM_API_KEY: str = os.environ.get("LLM_API_KEY", "")
LLM_MODEL: str = os.environ.get(
    "LLM_MODEL", "Pro/zai-org/GLM-5"
)

# ──────────────────────────────────────────────────────────────
# 辅助模型配置（项目描述生成 / 文本压缩 / 报告摘要）
#   用于结构化信息改写、文本浓缩等简单任务，可使用低成本小模型。
#   支持配置不同平台或账号，与主模型完全独立。
#   未设置时自动降级到主模型，保持向后兼容。
#   环境变量：LLM_LITE_API_URL / LLM_LITE_API_KEY / LLM_LITE_MODEL
# ──────────────────────────────────────────────────────────────
LLM_LITE_API_URL: str = os.environ.get("LLM_LITE_API_URL", LLM_API_URL)
LLM_LITE_API_KEY: str = os.environ.get("LLM_LITE_API_KEY", LLM_API_KEY)
LLM_LITE_MODEL: str = os.environ.get("LLM_LITE_MODEL", "Qwen/Qwen3.5-35B-A3B")
    
# ──────────────────────────────────────────────────────────────
# 阈值与数量
# ──────────────────────────────────────────────────────────────
STAR_GROWTH_THRESHOLD: int = 800       # 窗口期 star 增长阈值
MIN_STAR: int = 1200                   # 项目最低 star 门槛（关键词搜索 + 范围扫描下界）
MAX_STAR: int = 45000                  # 范围扫描上限
HOT_PROJECT_COUNT: int = 100           # 综合热门项目默认输出数量（上限，有几个出几个）
HOT_NEW_PROJECT_COUNT: int = 20        # 新项目榜默认输出数量（未指定 top_n 时使用）
GROWTH_CALC_DAYS: int = 7              # 增长统计窗口（天）—— 计算 star 增长的时间范围
DAYS_SINCE_CREATED: int = 45           # 新项目判定窗口（天）—— 创建时间距今 <= 此值视为新项目
DATA_EXPIRE_DAYS: int = GROWTH_CALC_DAYS + 1  # DB 数据过期判定天数（必须 > GROWTH_CALC_DAYS）

# ──────────────────────────────────────────────────────────────
# 评分模式
#   comprehensive — 综合排名（增长量 + 增长率，新项目平滑折扣）
#   hot_new       — 新项目专榜（仅创建时间 <= DAYS_SINCE_CREATED 天的新项目，按增长量排序）
# ──────────────────────────────────────────────────────────────
DEFAULT_SCORE_MODE: str = "comprehensive"

# ──────────────────────────────────────────────────────────────
# 请求控制
# ──────────────────────────────────────────────────────────────
MAX_BINARY_SEARCH_DEPTH: int = 20      # 二分法查 stargazers 最大深度
SEARCH_REQUEST_INTERVAL: float = 2.5   # Search API 请求最小间隔（秒）
MAX_GRAPHQL_SAMPLING_BATCHES: int = 30  # GraphQL 采样外推最多翻页批次数（30×100≈3000 条）

# ──────────────────────────────────────────────────────────────
# 路径配置（基于包根目录 github_hot_projects/）
#   可通过环境变量覆盖：DATA_DIR
# ──────────────────────────────────────────────────────────────
DATA_DIR: str = os.environ.get("DATA_DIR", str(PACKAGE_DIR))
DB_FILE_PATH = os.path.join(DATA_DIR, "Github_DB.json")
CHECKPOINT_FILE_PATH = os.path.join(DATA_DIR, ".pipeline_checkpoint.json")
REPORT_DIR = os.path.join(DATA_DIR, "report")
LOG_DIR = os.path.join(DATA_DIR, "logs")

# ──────────────────────────────────────────────────────────────
# 搜索关键词词典（AI 重点 + 通用全覆盖）
#   键 = 类别名，值 = 关键词列表
#   每个关键词会独立搜索，stars:>=MIN_STAR 自动追加
# ──────────────────────────────────────────────────────────────
SEARCH_KEYWORDS: dict[str, list[str]] = {
    # ─── AI 重点方向（高密度查询，每子方向多个关键词）───
    "AI-Agent": [
        "ai agent", "agent framework", "multi-agent", "agent sdk",
        "coding agent", "browser-use", "computer-use", "web agent",
        "ai tools", "autonomous agent", "task agent",
        "agent orchestration", "ai assistant", "tool calling",
        "function calling llm",
    ],
    "AI-MCP": [
        "mcp server", "mcp client", "model context protocol", "mcp sdk",
        "mcp tools", "mcp bridge", "mcp registry", "mcp integration",
    ],
    "AI-Skill-Prompt-Workflow": [
        "ai skill", "agent skill", "ai plugin", "prompt engineering",
        "prompt library", "prompt tool", "ai workflow",
        "workflow automation", "langgraph", "ai orchestration",
        "ai automation", "llm chain", "ai pipeline",
    ],
    "AI-CLI-DevTool": [
        "ai cli", "ai terminal", "ai devtool", "coding assistant",
        "code review ai", "code generation", "ai ide", "ai copilot",
        "ai coding", "code completion ai",
    ],
    "AI-LLM-Core": [
        "large language model", "llm framework", "llm sdk",
        "transformer model", "open source llm", "llm api",
        "chat model", "language model", "foundation model", "llm runtime",
    ],
    "AI-RAG": [
        "rag", "retrieval augmented", "vector database",
        "embedding model", "semantic search", "document retrieval",
        "knowledge base", "chunking embedding",
    ],
    "AI-Inference-Serving": [
        "llm inference", "llm serving", "vllm", "sglang",
        "quantization", "kv-cache", "speculative decoding",
        "model serving", "inference engine", "tensor parallel",
    ],
    "AI-Training-Finetune": [
        "finetune llm", "lora", "rlhf", "post-training",
        "pretraining framework", "dpo", "distillation", "alignment", "sft",
    ],
    "AI-Infra": [
        "triton kernel", "cuda kernel", "ml compiler",
        "distributed training", "model gateway", "gpu scheduling",
        "ml platform", "ai infrastructure",
    ],
    "AI-Multimodal": [
        "multimodal llm", "vision language model", "text to image",
        "text to video", "text to speech", "speech to text",
        "image generation", "video generation", "diffusion model",
    ],
    "AI-Observability": [
        "llm observability", "ai guardrails", "llm evaluation",
        "ai monitoring", "llm tracing", "ai safety", "model evaluation",
    ],
    "AI-Data-Synthetic": [
        "synthetic data", "data augmentation ai", "ai dataset",
        "llm data", "rlhf data", "instruction tuning data",
    ],
    "AI-Edge-OnDevice": [
        "on-device llm", "edge ai", "mobile llm", "llm.js",
        "webgpu llm", "tinyml",
    ],
    # ─── 通用类别（保证覆盖面）───
    "Database": [
        "database", "sql database", "nosql", "time series database",
        "graph database", "document database",
        # 热门数据库引擎
        "postgresql", "mysql", "redis", "sqlite", "elasticsearch",
        "mongodb", "clickhouse", "vector search", "olap database",
    ],
    "Cloud-Native": [
        "kubernetes", "docker", "terraform", "serverless", "service mesh",
        # 云原生工具链
        "helm chart", "prometheus", "grafana", "argo workflow", "cilium",
        "container runtime", "istio", "envoy proxy", "knative",
    ],
    "Frontend": [
        "react", "vue", "svelte", "ui component", "nextjs", "tailwindcss",
        # 前端框架与工具
        "angular", "nuxt", "vite", "webpack", "electron app",
        "flutter", "react native", "typescript", "webgl",
    ],
    "Backend": [
        "web framework", "api framework", "microservice", "graphql server", "rpc framework",
        # 后端框架
        "fastapi", "django", "spring boot", "golang http", "nodejs framework",
        "gin", "express", "nestjs", "flask", "koa",
    ],
    "DevOps": [
        "ci cd pipeline", "monitoring", "infrastructure as code", "gitops",
        # DevOps工具
        "ansible", "pulumi", "github actions", "jenkins", "gitlab ci",
        "argo cd", "flux", "terraform provider",
    ],
    "Security": [
        "security tool", "authentication", "vulnerability scanner",
        # 安全工具
        "waf", "ids ips", "penetration testing", "security scanner",
        "cve scanner", "secret scanner", "sast dast", "dependency check",
    ],
    "Data-Engineering": [
        "data pipeline", "etl", "stream processing", "feature store",
        "data lake", "data warehouse",
        # 数据工程工具
        "apache spark", "kafka", "flink", "airflow", "dbt",
        "duckdb", "polars", "pandas",
    ],
    "System-Tool": [
        "terminal tool", "cli tool", "shell", "wasm runtime",
        # 系统工具
        "terminal emulator", "file sync", "backup tool", "text editor",
        "neovim", "helix editor", "zsh", "fish shell", "tmux",
    ],
    "Programming-Language": [
        "programming language", "compiler", "language server",
        # 语言与编译器
        "rust", "golang", "zig", "lua", "julia",
        "python tooling", "typescript compiler", "lisp",
    ],
    # ─── 新兴领域补充 ───
    "Web3-Blockchain": [
        "blockchain", "ethereum", "smart contract", "defi",
        "web3", "nft", "crypto", "solidity", "layer2",
        "bitcoin", "solana", "arbitrum", "optimism",
    ],
    "Game-Engine": [
        "game engine", "unity", "unreal engine", "godot",
        "game framework", "game dev", "3d engine", "physics engine",
    ],
    "Audio-Video": [
        "video processing", "audio processing", "ffmpeg",
        "video editor", "audio editor", "media player",
        "streaming media", "video codec", "audio codec",
    ],
}
