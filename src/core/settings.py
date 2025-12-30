"""
本文件在整个项目中的角色：全局配置中心（Settings / Configuration Hub）。

它解决的核心问题是什么？
- 将运行参数与密钥（API keys、数据库连接、Tracing、服务端口等）统一收敛到一个 `settings` 实例：
  - 读取来源：环境变量 + `.env`（通过 `python-dotenv` 的 `find_dotenv()` 自动定位）
  - 解析与校验：由 `pydantic-settings` 完成（类型安全、默认值、校验失败早抛错）
- 基于“当前提供了哪些 provider 的 key”，动态推导：
  - `DEFAULT_MODEL`：服务端 `/info` 默认模型、Agent 默认模型
  - `AVAILABLE_MODELS`：服务端 `/info` 暴露给客户端的可选模型列表

典型调用者：
- `src/service/service.py`：鉴权（AUTH_SECRET）、暴露 `/info`（DEFAULT_MODEL/AVAILABLE_MODELS）、日志级别、启动参数等
- `src/core/llm.py`：初始化各 provider 的 ChatModel 时读取对应 API key 与 endpoint/base_url
- `src/memory/*`：选择数据库类型与连接参数（sqlite/postgres/mongo）

理解本文件的关键点：
- 这是“配置即代码”的落点：把一堆松散的 env vars，变成一个可被 IDE/类型系统理解的对象模型。
- `settings = Settings()` 在 import 时就会执行：因此缺失/错误配置会在启动期立刻暴露（Fail Fast）。
"""

from enum import StrEnum
from json import loads
from typing import Annotated, Any

from dotenv import find_dotenv
from pydantic import (
    BeforeValidator,
    Field,
    HttpUrl,
    SecretStr,
    TypeAdapter,
    computed_field,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from schema.models import (
    AllModelEnum,
    AnthropicModelName,
    AWSModelName,
    AzureOpenAIModelName,
    DeepseekModelName,
    FakeModelName,
    GoogleModelName,
    GroqModelName,
    OllamaModelName,
    OpenAICompatibleName,
    OpenAIModelName,
    OpenRouterModelName,
    Provider,
    VertexAIModelName,
)


class DatabaseType(StrEnum):
    """数据库后端类型枚举：用于选择 checkpointer/store 的具体实现。"""

    SQLITE = "sqlite"
    POSTGRES = "postgres"
    MONGO = "mongo"


class LogLevel(StrEnum):
    """日志级别枚举（字符串形式），便于从环境变量直接解析。"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def to_logging_level(self) -> int:
        """转换为 Python `logging` 模块使用的整数级别常量。"""
        import logging

        mapping = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }
        return mapping[self]


def check_str_is_http(x: str) -> str:
    """
    轻量校验：把字符串校验为 HttpUrl 并返回标准化后的字符串。

    为什么存在？
    - `pydantic` 的 `HttpUrl` 在 schema 中是专用类型；
    - 这里用 `BeforeValidator` 把“字符串 -> HttpUrl -> 字符串”串起来，
      既保证输入是合法 URL，又保持字段类型为 `str` 便于后续使用/序列化。
    """
    http_url_adapter = TypeAdapter(HttpUrl)
    return str(http_url_adapter.validate_python(x))


class Settings(BaseSettings):
    """
    项目运行期配置模型（由环境变量/.env 驱动）。

    设计意图：
    - 用类型与默认值把配置“结构化”：减少散落的 `os.getenv(...)`，并让 IDE 能提示字段。
    - 在 `model_post_init()` 中基于 provider 配置推导出默认模型与可用模型集合，
      使 `/info` 可以直接暴露给客户端，Agent 也可以有一致的默认行为。
    """

    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        validate_default=False,
    )
    MODE: str | None = None

    HOST: str = "0.0.0.0"
    PORT: int = 8080
    GRACEFUL_SHUTDOWN_TIMEOUT: int = 30
    LOG_LEVEL: LogLevel = LogLevel.WARNING

    AUTH_SECRET: SecretStr | None = None

    OPENAI_API_KEY: SecretStr | None = None
    DEEPSEEK_API_KEY: SecretStr | None = None
    ANTHROPIC_API_KEY: SecretStr | None = None
    GOOGLE_API_KEY: SecretStr | None = None
    GOOGLE_APPLICATION_CREDENTIALS: SecretStr | None = None
    GROQ_API_KEY: SecretStr | None = None
    USE_AWS_BEDROCK: bool = False
    OLLAMA_MODEL: str | None = None
    OLLAMA_BASE_URL: str | None = None
    USE_FAKE_MODEL: bool = False
    OPENROUTER_API_KEY: str | None = None

    # 如果 DEFAULT_MODEL 未显式指定，会在 `model_post_init()` 中根据可用 provider 自动设置。
    DEFAULT_MODEL: AllModelEnum | None = None  # type: ignore[assignment]
    AVAILABLE_MODELS: set[AllModelEnum] = set()  # type: ignore[assignment]

    # OpenAI Compatible：用于对接任意“OpenAI 协议兼容”的服务（PoC/自托管/网关场景常用）。
    COMPATIBLE_MODEL: str | None = None
    COMPATIBLE_API_KEY: SecretStr | None = None
    COMPATIBLE_BASE_URL: str | None = None

    OPENWEATHERMAP_API_KEY: SecretStr | None = None

    # MCP（Model Context Protocol）相关配置：用于 GitHub MCP 等外部工具服务器接入。
    GITHUB_PAT: SecretStr | None = None
    MCP_GITHUB_SERVER_URL: str = "https://api.githubcopilot.com/mcp/"

    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_PROJECT: str = "default"
    LANGCHAIN_ENDPOINT: Annotated[str, BeforeValidator(check_str_is_http)] = (
        "https://api.smith.langchain.com"
    )
    LANGCHAIN_API_KEY: SecretStr | None = None

    LANGFUSE_TRACING: bool = False
    LANGFUSE_HOST: Annotated[str, BeforeValidator(check_str_is_http)] = "https://cloud.langfuse.com"
    LANGFUSE_PUBLIC_KEY: SecretStr | None = None
    LANGFUSE_SECRET_KEY: SecretStr | None = None

    # 数据库配置：用于 LangGraph 的短期记忆（checkpointer）与长期记忆（store）。
    DATABASE_TYPE: DatabaseType = (
        DatabaseType.SQLITE
    )  # 可选：sqlite / postgres / mongo（具体能力见 `src/memory/*`）
    SQLITE_DB_PATH: str = "checkpoints.db"

    # PostgreSQL 配置
    POSTGRES_USER: str | None = None
    POSTGRES_PASSWORD: SecretStr | None = None
    POSTGRES_HOST: str | None = None
    POSTGRES_PORT: int | None = None
    POSTGRES_DB: str | None = None
    POSTGRES_APPLICATION_NAME: str = "agent-service-toolkit"
    POSTGRES_MIN_CONNECTIONS_PER_POOL: int = 1
    POSTGRES_MAX_CONNECTIONS_PER_POOL: int = 1

    # MongoDB 配置
    MONGO_HOST: str | None = None
    MONGO_PORT: int | None = None
    MONGO_DB: str | None = None
    MONGO_USER: str | None = None
    MONGO_PASSWORD: SecretStr | None = None
    MONGO_AUTH_SOURCE: str | None = None

    # Azure OpenAI 配置
    AZURE_OPENAI_API_KEY: SecretStr | None = None
    AZURE_OPENAI_ENDPOINT: str | None = None
    AZURE_OPENAI_API_VERSION: str = "2024-02-15-preview"
    AZURE_OPENAI_DEPLOYMENT_MAP: dict[str, str] = Field(
        default_factory=dict, description="Map of model names to Azure deployment IDs"
    )

    def model_post_init(self, __context: Any) -> None:
        """
        Pydantic v2 的“实例创建后钩子”：在这里做派生字段计算与跨字段校验。

        本项目在这里做两件事：
        1) 根据当前环境里“哪些 provider 可用”推导 `DEFAULT_MODEL` 与 `AVAILABLE_MODELS`
        2) 对某些 provider 做更严格的必填字段校验（例如 Azure 需要 deployment map）

        为什么不在字段定义阶段做？
        - provider 的可用性往往取决于多字段组合（例如 OpenAI Compatible 需要 base_url + model），
          更适合集中在一个地方处理。
        """
        # `api_keys` 的 value 不一定都是 SecretStr：有些 provider 以“布尔开关”或“字段组合”表示启用。
        # 这里的目标不是“拿到 key”，而是判断 provider 是否处于激活状态。
        api_keys = {
            Provider.OPENAI: self.OPENAI_API_KEY,
            Provider.OPENAI_COMPATIBLE: self.COMPATIBLE_BASE_URL and self.COMPATIBLE_MODEL,
            Provider.DEEPSEEK: self.DEEPSEEK_API_KEY,
            Provider.ANTHROPIC: self.ANTHROPIC_API_KEY,
            Provider.GOOGLE: self.GOOGLE_API_KEY,
            Provider.VERTEXAI: self.GOOGLE_APPLICATION_CREDENTIALS,
            Provider.GROQ: self.GROQ_API_KEY,
            Provider.AWS: self.USE_AWS_BEDROCK,
            Provider.OLLAMA: self.OLLAMA_MODEL,
            Provider.FAKE: self.USE_FAKE_MODEL,
            Provider.AZURE_OPENAI: self.AZURE_OPENAI_API_KEY,
            Provider.OPENROUTER: self.OPENROUTER_API_KEY,
        }
        active_keys = [k for k, v in api_keys.items() if v]
        if not active_keys:
            raise ValueError("At least one LLM API key must be provided.")

        # 注意：如果同时配置了多个 provider，这里会按 active_keys 的顺序选择第一个作为默认模型来源。
        #（这是一种可接受的默认策略；更复杂的实现可以提供明确的 DEFAULT_MODEL 覆盖。）
        for provider in active_keys:
            match provider:
                case Provider.OPENAI:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OpenAIModelName.GPT_5_NANO
                    self.AVAILABLE_MODELS.update(set(OpenAIModelName))
                case Provider.OPENAI_COMPATIBLE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OpenAICompatibleName.OPENAI_COMPATIBLE
                    self.AVAILABLE_MODELS.update(set(OpenAICompatibleName))
                case Provider.DEEPSEEK:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = DeepseekModelName.DEEPSEEK_CHAT
                    self.AVAILABLE_MODELS.update(set(DeepseekModelName))
                case Provider.ANTHROPIC:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = AnthropicModelName.HAIKU_45
                    self.AVAILABLE_MODELS.update(set(AnthropicModelName))
                case Provider.GOOGLE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = GoogleModelName.GEMINI_20_FLASH
                    self.AVAILABLE_MODELS.update(set(GoogleModelName))
                case Provider.VERTEXAI:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = VertexAIModelName.GEMINI_20_FLASH
                    self.AVAILABLE_MODELS.update(set(VertexAIModelName))
                case Provider.GROQ:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = GroqModelName.LLAMA_31_8B
                    self.AVAILABLE_MODELS.update(set(GroqModelName))
                case Provider.AWS:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = AWSModelName.BEDROCK_HAIKU
                    self.AVAILABLE_MODELS.update(set(AWSModelName))
                case Provider.OLLAMA:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OllamaModelName.OLLAMA_GENERIC
                    self.AVAILABLE_MODELS.update(set(OllamaModelName))
                case Provider.OPENROUTER:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OpenRouterModelName.GEMINI_25_FLASH
                    self.AVAILABLE_MODELS.update(set(OpenRouterModelName))
                case Provider.FAKE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = FakeModelName.FAKE
                    self.AVAILABLE_MODELS.update(set(FakeModelName))
                case Provider.AZURE_OPENAI:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = AzureOpenAIModelName.AZURE_GPT_4O_MINI
                    self.AVAILABLE_MODELS.update(set(AzureOpenAIModelName))
                    # 如果启用 Azure provider，则校验 Azure OpenAI 相关必填配置
                    if not self.AZURE_OPENAI_API_KEY:
                        raise ValueError("AZURE_OPENAI_API_KEY must be set")
                    if not self.AZURE_OPENAI_ENDPOINT:
                        raise ValueError("AZURE_OPENAI_ENDPOINT must be set")
                    if not self.AZURE_OPENAI_DEPLOYMENT_MAP:
                        raise ValueError("AZURE_OPENAI_DEPLOYMENT_MAP must be set")

                    # 如果 deployment map 来自环境变量字符串，则解析为 dict
                    if isinstance(self.AZURE_OPENAI_DEPLOYMENT_MAP, str):
                        try:
                            self.AZURE_OPENAI_DEPLOYMENT_MAP = loads(
                                self.AZURE_OPENAI_DEPLOYMENT_MAP
                            )
                        except Exception as e:
                            raise ValueError(f"Invalid AZURE_OPENAI_DEPLOYMENT_MAP JSON: {e}")

                    # 校验必需的 deployments 是否齐全
                    required_models = {"gpt-4o", "gpt-4o-mini"}
                    missing_models = required_models - set(self.AZURE_OPENAI_DEPLOYMENT_MAP.keys())
                    if missing_models:
                        raise ValueError(f"Missing required Azure deployments: {missing_models}")
                case _:
                    raise ValueError(f"Unknown provider: {provider}")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def BASE_URL(self) -> str:
        # 计算属性：把 host/port 组合成 base url，供 client/service 统一使用。
        return f"http://{self.HOST}:{self.PORT}"

    def is_dev(self) -> bool:
        """是否处于 dev 模式（通常用于控制日志、热重载、调试开关等）。"""
        return self.MODE == "dev"


settings = Settings()
