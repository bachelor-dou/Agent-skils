"""全局配置 —— 改行为只看这一个文件。

- `LLM_MODELS` 是原始声明,归一化在 `infra/llm`。
- import 期不碰磁盘(建目录靠 `ensure_dir`)。
- 实测定死的实现细节(批大小、并发等)不进这里 —— 调错会静默污染基线。
- 最底座:任何层都能 import 它,它不 import 任何层。
"""

from pathlib import Path

from .common.env import csv_list, flag, text

# ══════════════════════════════════════════════════════════════
# 一、策略旋钮 —— 用户真会去调的数,每个都注明调大调小会发生什么
# ══════════════════════════════════════════════════════════════

# ── 观测宽度与出榜闸门:两个独立参数,勿混为一谈 ──

# 观测宇宙宽度:涨过它收进 DB 记快照,掉下淘汰。也是榜单候选池下界和 Agent min_star 默认值。
# 调低只是提前记快照,不影响入选标准 —— 谁出榜由 STAR_GROWTH_THRESHOLD 独立决定。
MIN_STAR: int = 500

# 出榜的唯一闸门:窗口期涨够这么多 star 才入选。想收紧/放宽榜单只动这个数。
STAR_GROWTH_THRESHOLD: int = 1000

# 星段扫描上限,仅每日发现阶段用;榜单读快照不设上限,否则超大仓库出不了榜。
# 调高几乎不费时(按密度二分,96% 集中在 2 万以下),超出的约 91 个由无上限关键词搜索兜底。
MAX_STAR: int = 3000

# ── 窗口与数量 ──

GROWTH_CALC_DAYS: int = 7        # 增长统计窗口(天);实际跨度以锚点快照为准,本值是请求值
DAYS_SINCE_CREATED: int = 45     # 新项目判定:创建距今 <= 此值算新项目
DESC_REFRESH_DAYS: int = 60      # LLM 描述刷新周期(天),超期则重新生成
HOT_PROJECT_COUNT: int = 100     # 综合/关键词榜默认输出上限(有几个出几个)
HOT_NEW_PROJECT_COUNT: int = 13  # 新项目榜默认输出数量
SNAPSHOT_KEEP_DAYS: int = 35     # 快照保留天数(按日期截断,够覆盖月度窗口)

# 关键词榜:LLM 动态补充关键词的数量上限,也是 Search 配额闸门 —— 调大线性增请求。预设类别不受此限。
MAX_DYNAMIC_SEARCH_KEYWORDS: int = 30

# ── 最近爆发加成(仅综合/关键词榜):窗口总增长基础分之上乘一个「最近爆发强度」加成 ──
#     acceleration = (recent_growth / RECENT_GROWTH_DAYS) / (window_growth / window_days)
#     boost        = 1 + BURST_ALPHA * min(max(acceleration - 1, 0), BURST_CAP)
#   acceleration <= 1 不惩罚;缺快照自行跳过,故无开关。

RECENT_GROWTH_DAYS: int = 3   # 「最近几天」窗口
BURST_ALPHA: float = 0.2     # 加成强度,越大则最近爆发对排名影响越大
BURST_CAP: float = 2.0        # acceleration-1 的封顶(boost 最高 1 + ALPHA*CAP)

# ── 其他 ──

# 收藏默认标签:点 ★ 时可选的分类,用户仍可自定义新标签。
FAVORITE_DEFAULT_TAGS: list[str] = ["效率", "工具"]


# ══════════════════════════════════════════════════════════════
# 二、路径 —— 纯派生,import 时不碰磁盘
# ══════════════════════════════════════════════════════════════
# 数据跟着包走(data/、report/ 由 CI 产出提交,本地 git pull 拿到)。

# 必须是绝对路径:相对路径跟着 CWD 走,而 CI/python -m/agent CLI/pytest 四种入口 CWD 各异,
# 算错不报错、只读到空。
PACKAGE_DIR = Path(__file__).resolve().parent

DATA_DIR = PACKAGE_DIR / "data"
REPORT_DIR = PACKAGE_DIR / "report"

# 日志不入库,各包写各自的
LOG_DIR = PACKAGE_DIR / "logs"

# 前端静态资源(html/css/js):跟着包走不进 data —— 是代码不是数据。
WEB_DIR = PACKAGE_DIR / "web"

DB_PATH = DATA_DIR / "Github_DB.json"
FAVORITES_PATH = DATA_DIR / "favorites.json"
SNAPSHOT_DIR = DATA_DIR / "snapshots"


def ensure_dir(path: Path) -> Path:
    """确保目录存在并返回它。**只在真正要写之前调用**,不在 import 期调用。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ══════════════════════════════════════════════════════════════
# 三、机密 —— 只从环境变量读,永不落文件、永不进可序列化结构
# ══════════════════════════════════════════════════════════════
# key 一旦进数据结构就会被日志、/api、异常回溯带出去。模型目录只声明 key 在哪个环境变量,
# 真值在使用处才取。


def github_tokens() -> list[str]:
    """GitHub token 列表(逗号分隔)。未配置返回 []。做成函数是为了运行期能重读(中途补 token)。"""
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
# 配合安全中间件:IP 黑名单 → 敏感路径 → 速率限制 → 请求日志。

# 【CORS 白名单】只约束浏览器跨域,curl/脚本不受影响 —— 不是访问控制。生产用环境变量指定
#   明确域名,别用 "*"。
_DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
]
CORS_ALLOWED_ORIGINS: list[str] = csv_list("CORS_ALLOWED_ORIGINS") or _DEFAULT_ALLOWED_ORIGINS
CORS_ALLOW_CREDENTIALS: bool = flag("CORS_ALLOW_CREDENTIALS", default=False)

# 【IP 黑名单】名单内 IP 一律 403,默认放行、命中才拒(取真实 IP,支持 X-Forwarded-For)。
#   环境变量 SECURITY_IP_BLACKLIST 覆盖内置项。
_DEFAULT_IP_BLACKLIST = [
    "104.243.32.126",
    "209.222.101.194",
    "172.232.209.215",
]
SECURITY_IP_BLACKLIST: list[str] = csv_list("SECURITY_IP_BLACKLIST") or _DEFAULT_IP_BLACKLIST


# ══════════════════════════════════════════════════════════════
# 五、LLM 平台目录 —— 纯声明,不含 key
# ══════════════════════════════════════════════════════════════
# 每条只记 key_env,真值由 llm_key() 使用处取,故整表可安全序列化。
# 归一化(补默认、剔 enabled=0、校验 id 唯一、拆 lite_model)在 infra/llm,免循环 import。
# 字段:id 唯一标识 / label 展示名 / backend 协议(azure|openai) / url endpoint / model 主模型 /
#   lite_model 子模型(逗号分隔,空则回退主模型) / key_env 环境变量名 / enabled 1|0 / desc 备注

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
# 六、搜索关键词词典 —— 放最后,最长且最少改
# ══════════════════════════════════════════════════════════════
# 键=类别,值=关键词;每个关键词独立搜一次(`stars:>=MIN_STAR` 由调用方追加)。
# 加一个关键词=加一次 Search 请求,发现阶段耗时地板 = 总请求数 ÷ 限流速率。

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
