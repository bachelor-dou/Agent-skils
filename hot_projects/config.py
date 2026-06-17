"""
全局配置模块
============
集中管理所有可调参数：GitHub Token、LLM 接口、阈值、评分权重、路径等。
修改配置只需编辑此文件或通过环境变量覆盖。
"""

import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent

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


def _parse_csv_env(name: str) -> list[str]:
    """读取逗号分隔的环境变量，忽略空值。"""
    value = os.environ.get(name, "")
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_bool_env(name: str, default: bool = False) -> bool:
    """读取布尔环境变量。"""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# ──────────────────────────────────────────────────────────────
# API Server 安全配置
# ──────────────────────────────────────────────────────────────
# 这三项配合 api_server.py 的 SecurityMiddleware 工作，对每个进入的 HTTP 请求做：
#   IP 黑名单拦截 → 敏感路径拦截 → 速率限制 → 请求日志。
# 本节定义其中的「CORS 白名单」与「IP 黑名单」。

# 【CORS 白名单】允许哪些前端域名跨域调用本 API。
#   - 浏览器跨域请求才受此限制；非浏览器客户端（curl/脚本）不受影响。
#   - 生产环境务必通过环境变量 CORS_ALLOWED_ORIGINS 指定明确域名，不要用 "*"。
#   - 例如：CORS_ALLOWED_ORIGINS="https://example.com,https://app.example.com"
_DEFAULT_CORS_ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
]
CORS_ALLOWED_ORIGINS: list[str] = _parse_csv_env("CORS_ALLOWED_ORIGINS") or _DEFAULT_CORS_ALLOWED_ORIGINS
CORS_ALLOW_CREDENTIALS: bool = _parse_bool_env("CORS_ALLOW_CREDENTIALS", default=False)

# 【IP 黑名单】名单内的来源 IP 一律禁止访问本服务，命中即直接返回 403 Forbidden。
#   - 用途：屏蔽已确认的恶意扫描器 / 攻击源 IP（下方内置的几个就是历史抓到的扫描 IP）。
#   - 匹配的是请求方真实 IP（支持反向代理的 X-Forwarded-For）。
#   - 可通过环境变量 SECURITY_IP_BLACKLIST 覆盖；未配置时使用内置默认值。
#   - 例如追加封禁：SECURITY_IP_BLACKLIST="1.2.3.4,5.6.7.8"
#   - 注意：这是「黑名单」（默认放行、命中才拒绝），不是「白名单」（默认拒绝、命中才放行）。
_DEFAULT_SECURITY_IP_BLACKLIST = [
    "104.243.32.126",
    "209.222.101.194",
    "172.232.209.215",
]
SECURITY_IP_BLACKLIST: list[str] = _parse_csv_env("SECURITY_IP_BLACKLIST") or _DEFAULT_SECURITY_IP_BLACKLIST

# ──────────────────────────────────────────────────────────────
# LLM A/B 双后端配置（兼容 OpenAI /v1/chat/completions 格式）
#   逐调用回退：每次先调方案 A，失败再回退方案 B。
#   两方案可配置不同平台、账号、模型与参数风格（azure / openai）。
# ──────────────────────────────────────────────────────────────
# URL、后端类型、模型名均固定在此；仅 KEY 从环境变量读取（export LLM_A_KEY / LLM_B_KEY）。

# ===== LLM 方案 A（主力）: Azure OpenAI =====
LLM_A_BACKEND = "azure"
LLM_A_URL = "https://ceshi-001.openai.azure.com/openai/v1/chat/completions?api-version=preview"
LLM_A_MODEL = "gpt-5.4"
LLM_A_LITE_MODEL = "gpt-5.4-mini"
LLM_A_KEY = os.environ.get("LLM_A_KEY", "")

# ===== LLM 方案 B（备选）: SiliconFlow =====
LLM_B_BACKEND = "openai"
LLM_B_URL = "https://api.siliconflow.cn/v1/chat/completions"
LLM_B_MODEL = "Pro/zai-org/GLM-5.1"
LLM_B_LITE_MODEL = "Qwen/Qwen3.5-35B-A3B"
LLM_B_KEY = os.environ.get("LLM_B_KEY", "")

# ──────────────────────────────────────────────────────────────
# 阈值与数量
# ──────────────────────────────────────────────────────────────
STAR_GROWTH_THRESHOLD: int = 1000       # 窗口期 star 增长阈值
MIN_STAR: int = 1200                   # 项目最低 star 门槛（关键词搜索 + 范围扫描下界）
MAX_STAR: int = 45000                  # 范围扫描上限
HOT_PROJECT_COUNT: int = 120           # 综合热门项目默认输出数量（上限，有几个出几个）
HOT_NEW_PROJECT_COUNT: int = 15        # 新项目榜默认输出数量（未指定 top_n 时使用）
GROWTH_CALC_DAYS: int = 7              # 增长统计窗口（天）—— 计算 star 增长的时间范围
DAYS_SINCE_CREATED: int = 45           # 新项目判定窗口（天）—— 创建时间距今 <= 此值视为新项目
# DB 差值法：项目 refreshed_at 年龄与计算窗口的最大允许偏差（小时）。
# 仅当 |项目年龄 − 计算窗口| ≤ 该值时，current_star − DB旧star 才被视为有效的窗口期增长。
DB_DIFF_TOLERANCE_HOURS: int = 5
# 关键词榜：LLM 动态补充的搜索关键词数量上限（控制 Search API 配额；预设类别不受此限）
MAX_DYNAMIC_SEARCH_KEYWORDS: int = 30

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
SEARCH_REQUEST_INTERVAL: float = 1.3  # Search API 请求最小间隔（秒）
MAX_GRAPHQL_SAMPLING_BATCHES: int = 45  # GraphQL 采样外推最多翻页批次数（35×100≈3500 条）

# ──────────────────────────────────────────────────────────────
# 路径配置（基于包根目录 hot_projects/）
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
        "autonomous agent", "agent orchestration", "ai assistant",
        "tool calling", "function calling llm", "deep research agent",
        "voice agent",
    ],
    "AI-MCP": [
        "mcp server", "mcp client", "model context protocol", "mcp sdk",
        "mcp tools", "mcp bridge", "mcp registry", "mcp integration",
    ],
    "AI-Skill-Prompt-Workflow": [
        "ai skill", "agent skill", "ai plugin", "prompt engineering",
        "prompt library", "prompt tool", "ai workflow",
        "workflow automation", "langgraph", "llm chain",
    ],
    "AI-CLI-DevTool": [
        "ai cli", "ai terminal", "ai devtool", "coding assistant",
        "code review ai", "code generation", "ai ide", "ai copilot",
        "ai coding", "code completion ai",
    ],
    "AI-LLM-Core": [
        "large language model", "llm framework", "llm sdk",
        "transformer model", "open source llm", "llm api",
        "foundation model",
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
        "fine-tuning", "instruction tuning", "lora", "qlora", "peft",
        "rlhf", "dpo", "sft", "reward model", "distillation",
        "alignment", "pretraining framework",
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
    # ─── AI 应用层补充 ───
    "AI-App-Chatbot": [
        "chatbot", "chatgpt", "ai chat", "llm app", "ai webui",
        "openai compatible", "ai gateway", "llm proxy",
    ],
    # ─── 经典 ML / 深度学习（非 LLM）───
    "ML-DeepLearning": [
        "pytorch", "tensorflow", "jax", "scikit-learn", "keras",
        "deep learning", "machine learning framework", "onnx",
        "reinforcement learning",
    ],
    "Computer-Vision": [
        "computer vision", "object detection", "ocr", "opencv",
        "image segmentation", "pose estimation", "face recognition",
    ],
    # ─── 移动 / 桌面 / 跨端 ───
    "Mobile": [
        "android", "ios", "swift", "kotlin", "jetpack compose", "swiftui",
    ],
    "Desktop": [
        "tauri", "qt framework", "gtk", "wails", "desktop app",
    ],
    # ─── 自托管 / Homelab ───
    "Self-Hosted": [
        "self-hosted", "home assistant", "media server", "dashboard",
        "home automation",
    ],
    # ─── 爬虫 / 自动化 ───
    "Scraping-Automation": [
        "web scraping", "crawler", "playwright", "browser automation", "rpa",
    ],
    # ─── 代理 / 网关 / 网络 ───
    "Proxy-Gateway": [
        "nginx", "caddy", "traefik", "api gateway", "reverse proxy",
        "load balancer",
    ],
    # ─── 文档 / 静态站点 ───
    "Docs-StaticSite": [
        "static site generator", "hugo", "docusaurus", "mkdocs", "documentation",
    ],
    # ─── 测试 / QA ───
    "Testing-QA": [
        "testing framework", "cypress", "selenium", "e2e testing", "test automation",
    ],
    # ─── 机器人 / IoT / 嵌入式 ───
    "Robotics-IoT": [
        "ros", "esp32", "arduino", "raspberry pi", "embedded", "firmware",
    ],
}
