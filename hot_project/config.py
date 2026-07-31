"""全局配置 —— 想改行为就来这一个文件。

第五节的 `LLM_MODELS` 是**原始声明**(含 enabled=0 的条目),归一化在 `infra/llm` 里做。
import 期不碰磁盘,建目录推到真要写之前(`ensure_dir`)。
实测定死的实现细节(批大小、并发上限、容差等)不进这里 —— 当旋钮调错会静默污染基线。
分层上本文件是最底座:任何层都能 import 它,它**不 import 任何层**(只用标准库)。
"""

from pathlib import Path

from .common.env import csv_list, flag, text

# ══════════════════════════════════════════════════════════════
# 一、策略旋钮 —— 用户真会去调的数,每个都注明调大调小会发生什么
# ══════════════════════════════════════════════════════════════

# ── 观测宽度与出榜闸门:两个独立旋钮,别把它们当成一回事 ──

# 「观测宇宙」的宽度:涨过它就收进 DB 开始记快照,掉到它以下就淘汰。同时是榜单候选池的
# 下界(候选池就是 DB,必须同值)和 Agent 工具 min_star 的默认值。
# 调低只是提前给仓库记快照,不放水 —— 谁能出榜由 STAR_GROWTH_THRESHOLD 独立决定。
MIN_STAR: int = 500

# 出榜的唯一闸门:窗口期涨够这么多 star 才入选。想收紧/放宽榜单只动这个数。
STAR_GROWTH_THRESHOLD: int = 1000

# 星段扫描的上限,只用于每日发现阶段;榜单侧读快照时不设 star 上限,否则超大仓库出不了榜。
# 调高几乎不费时间(分段按命中密度二分,96% 的仓库挤在 2 万以下),超出本值的约 91 个仓库
# 由无上限的关键词搜索兜底。
MAX_STAR: int = 120000

# ── 窗口与数量 ──

GROWTH_CALC_DAYS: int = 7        # 增长统计窗口(天);实际跨度以锚点快照为准,本值是请求值
DAYS_SINCE_CREATED: int = 45     # 新项目判定:创建距今 <= 此值算新项目
DESC_REFRESH_DAYS: int = 60      # LLM 描述刷新周期(天),超期则重新生成
HOT_PROJECT_COUNT: int = 100     # 综合/关键词榜默认输出上限(有几个出几个)
HOT_NEW_PROJECT_COUNT: int = 13  # 新项目榜默认输出数量
SNAPSHOT_KEEP_DAYS: int = 35     # 快照保留天数(按日期截断,够覆盖月度窗口)

# 关键词榜:LLM 动态补充的搜索关键词数量上限,也是 Search API 配额的闸门 ——
# 每个关键词是一次独立搜索,调大直接线性增加请求数。预设类别不受此限。
MAX_DYNAMIC_SEARCH_KEYWORDS: int = 30

# ── 最近爆发加成(仅综合/关键词榜打分)──
#   在「窗口总增长」基础分之上叠加「最近几天爆发强度」的乘法加成,
#   让最近突然爆火的项目排名更高,而不是只看窗口平均:
#     acceleration = (recent_growth / RECENT_GROWTH_DAYS) / (window_growth / window_days)
#     boost        = 1 + BURST_ALPHA * min(max(acceleration - 1, 0), BURST_CAP)
#   acceleration <= 1 不反向惩罚;缺快照自行跳过(不加成、不报错),所以没有开关。

RECENT_GROWTH_DAYS: int = 3   # 「最近几天」窗口
BURST_ALPHA: float = 0.2     # 加成强度,越大则最近爆发对排名影响越大
BURST_CAP: float = 2.0        # acceleration-1 的封顶(boost 最高 1 + ALPHA*CAP)

# ── 其他 ──

# 收藏默认标签:点 ★ 时可选的分类,用户仍可自定义新标签。
FAVORITE_DEFAULT_TAGS: list[str] = ["效率", "工具"]


# ══════════════════════════════════════════════════════════════
# 二、路径 —— 纯派生,import 时不碰磁盘
# ══════════════════════════════════════════════════════════════
#
# 数据跟着包走:`hot_project/data/`(DB、快照、收藏)与 `hot_project/report/`(周报 md),
# 由 CI 产出并提交到 tmp 分支,本地 `git pull` 拿到。

# 所有路径都从这里推,且必须是绝对的:相对路径会跟着 CWD 走,而 CI、`python -m hot_project`、
# agent CLI、pytest 四种入口的 CWD 各不相同,算错了不报错,只是读到空。
PACKAGE_DIR = Path(__file__).resolve().parent

DATA_DIR = PACKAGE_DIR / "data"
REPORT_DIR = PACKAGE_DIR / "report"

# 日志不入库,各包写各自的
LOG_DIR = PACKAGE_DIR / "logs"

# 前端静态资源(html/css/js):跟着包走而不进 data —— 它是代码,不是数据。
WEB_DIR = PACKAGE_DIR / "web"

DB_PATH = DATA_DIR / "Github_DB.json"
FAVORITES_PATH = DATA_DIR / "favorites.json"
SNAPSHOT_DIR = DATA_DIR / "snapshots"


def ensure_dir(path: Path) -> Path:
    """确保目录存在并返回它。**只在真正要写之前调用**,不在 import 期调用。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ══════════════════════════════════════════════════════════════
# 三、机密 —— 只从环境变量读,永不落文件,永不进可序列化的数据结构
# ══════════════════════════════════════════════════════════════
#
# 最后一条是这几个函数存在的理由:key 一旦进了数据结构,就会被日志、/api 返回、异常回溯
# 带出去。模型目录(第五节)只声明「我的 key 在哪个环境变量里」,真值在使用处才取。


def github_tokens() -> list[str]:
    """GitHub token 列表(逗号分隔)。未配置返回 [],由运行入口决定是否退出。

    做成函数而非常量:token 池要能在进程运行期间重读(用户中途补 token)。
    """
    return csv_list("GITHUB_TOKENS")


def llm_key(env_name: str) -> str:
    """按环境变量名取 LLM key。空串表示该平台未配置,调用方应跳过它而不是发空 key 请求。"""
    return text(env_name)


def serverchan_sendkey() -> str:
    """Server酱推送 key。空串 = 关闭推送(周报生成后不推),不是错误。"""
    return text("SERVERCHAN_SENDKEY")


# ══════════════════════════════════════════════════════════════
# 四、Web 安全 —— 只有 api_server.py 一个消费者
# ══════════════════════════════════════════════════════════════
#
# 这两项配合安全中间件:IP 黑名单 → 敏感路径 → 速率限制 → 请求日志。

# 【CORS 白名单】允许哪些前端域名跨域调用本 API。只约束浏览器跨域请求,curl / 脚本不受
#   影响,所以**它不是访问控制**。生产环境务必用环境变量指定明确域名,不要 "*"。
_DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
]
CORS_ALLOWED_ORIGINS: list[str] = csv_list("CORS_ALLOWED_ORIGINS") or _DEFAULT_ALLOWED_ORIGINS
CORS_ALLOW_CREDENTIALS: bool = flag("CORS_ALLOW_CREDENTIALS", default=False)

# 【IP 黑名单】名单内的来源 IP 一律 403,默认放行、命中才拒绝,匹配请求方真实 IP
#   (支持反向代理的 X-Forwarded-For)。环境变量 SECURITY_IP_BLACKLIST 会**覆盖**内置项。
_DEFAULT_IP_BLACKLIST = [
    "104.243.32.126",
    "209.222.101.194",
    "172.232.209.215",
]
SECURITY_IP_BLACKLIST: list[str] = csv_list("SECURITY_IP_BLACKLIST") or _DEFAULT_IP_BLACKLIST


# ══════════════════════════════════════════════════════════════
# 五、LLM 平台目录 —— 纯声明,不含任何 key
# ══════════════════════════════════════════════════════════════
#
# 每条只记 `key_env`(key 在哪个环境变量里),真值由第三节的 `llm_key()` 在使用处取,
# 所以这个列表可以安全地整体序列化。
#
# 归一化(补默认值、剔除 enabled=0、校验 id 唯一、拆 lite_model)在 `infra/llm` 里做 ——
# 配置反向依赖实现是循环 import 的温床。
#
# 字段:
#   id          唯一标识,前端按它选平台
#   label       前端展示名
#   backend     协议分支(azure / openai),决定请求头与 payload 形状
#   url         完整 endpoint
#   model       主模型名
#   lite_model  子模型,逗号分隔可多个;留空则 lite 调用回退用主模型
#   key_env     key 所在的环境变量名
#   enabled     1/0,关闭的条目全局不可用(前端不显示、内部回退也跳过)
#   desc        备注,仅人看

LLM_MODELS: list[dict] = [
    {
        "id": "azure01",
        "label": "GPT-5.4",
        "backend": "azure",
        "url": "https://ceshi-001.openai.azure.com/openai/v1/chat/completions?api-version=preview",
        "model": "gpt-5.4",
        "lite_model": "gpt-5.4-mini",
        "key_env": "LLM_A_KEY",
        "enabled": 1,
        "desc": "三组测试用",
    },
    {
        "id": "azure02",
        "label": "GPT-5.5",
        "backend": "azure",
        "url": "https://project003003.openai.azure.com/openai/v1/chat/completions?api-version=preview",
        "model": "gpt-5.5",
        "lite_model": "gpt-5.4-mini",
        "key_env": "LLM_B_KEY",
        "enabled": 1,
        "desc": "三组专用",
    },
    {
        "id": "siliconflow",
        "label": "GLM-5.1(雷达)",
        "backend": "openai",
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "model": "Pro/zai-org/GLM-5.1",
        "lite_model": "Qwen/Qwen3.6-35B-A3B,Qwen/Qwen3.5-35B-A3B",
        "key_env": "LLM_C_KEY",
        "enabled": 1,
        "desc": "开源雷达",
    },
    {
        # 阿里百炼兼容模式即 OpenAI 格式,字段与 openai 分支一致
        "id": "aliyun01",
        "label": "GLM-5.1(阿里A)",
        "backend": "openai",
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model": "glm-5.1",
        "lite_model": "qwen3.6-35b-a3b",
        "key_env": "LLM_D_KEY",
        "enabled": 1,
        "desc": "开源个人",
    },
    {
        "id": "aliyun02",
        "label": "Qwen3.7-Max(阿里B)",
        "backend": "openai",
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model": "qwen3.7-max",
        "lite_model": "qwen-flash",
        "key_env": "LLM_E_KEY",
        "enabled": 0,
        "desc": "阿里百炼三组",
    },
]


# ══════════════════════════════════════════════════════════════
# 六、搜索关键词词典 —— 放最后,因为它最长且最少改
# ══════════════════════════════════════════════════════════════
#
# 键 = 类别名,值 = 关键词列表。每个关键词独立搜索一次,`stars:>=MIN_STAR` 由调用方追加。
#
# 加一个关键词就是加一次 Search API 搜索,而发现阶段的耗时地板就是总请求数 ÷ 限流速率。

SEARCH_KEYWORDS: dict[str, list[str]] = {
    # ─── AI 重点方向 ───
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
    # ─── 通用类别(保证覆盖面)───
    "Database": [
        "database", "sql database", "nosql", "time series database",
        "graph database", "document database",
        "postgresql", "mysql", "redis", "sqlite", "elasticsearch",
        "mongodb", "clickhouse", "vector search", "olap database",
    ],
    "Cloud-Native": [
        "kubernetes", "docker", "terraform", "serverless", "service mesh",
        "helm chart", "prometheus", "grafana", "argo workflow", "cilium",
        "container runtime", "istio", "envoy proxy", "knative",
    ],
    "Frontend": [
        "react", "vue", "svelte", "ui component", "nextjs", "tailwindcss",
        "angular", "nuxt", "vite", "webpack", "electron app",
        "flutter", "react native", "typescript", "webgl",
    ],
    "Backend": [
        "web framework", "api framework", "microservice", "graphql server", "rpc framework",
        "fastapi", "django", "spring boot", "golang http", "nodejs framework",
        "gin", "express", "nestjs", "flask", "koa",
    ],
    "DevOps": [
        "ci cd pipeline", "monitoring", "infrastructure as code", "gitops",
        "ansible", "pulumi", "github actions", "jenkins", "gitlab ci",
        "argo cd", "flux", "terraform provider",
    ],
    "Security": [
        "security tool", "authentication", "vulnerability scanner",
        "waf", "ids ips", "penetration testing", "security scanner",
        "cve scanner", "secret scanner", "sast dast", "dependency check",
    ],
    "Data-Engineering": [
        "data pipeline", "etl", "stream processing", "feature store",
        "data lake", "data warehouse",
        "apache spark", "kafka", "flink", "airflow", "dbt",
        "duckdb", "polars", "pandas",
    ],
    "System-Tool": [
        "terminal tool", "cli tool", "shell", "wasm runtime",
        "terminal emulator", "file sync", "backup tool", "text editor",
        "neovim", "helix editor", "zsh", "fish shell", "tmux",
    ],
    "Programming-Language": [
        "programming language", "compiler", "language server",
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
    # ─── 经典 ML / 深度学习(非 LLM)───
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
