"""
LLM 模型枚举定义模块

本模块定义了系统支持的所有 LLM（大语言模型）提供商和模型名称的枚举类型。
支持的提供商包括：OpenAI、Azure OpenAI、DeepSeek、Anthropic、Google、
Vertex AI、Groq、AWS Bedrock、Ollama、OpenRouter 等。

使用示例：
    from schema.models import Provider, OpenAIModelName
    
    # 获取提供商
    provider = Provider.OPENAI
    
    # 获取模型名称
    model = OpenAIModelName.GPT_5_NANO
"""

from enum import StrEnum, auto
from typing import TypeAlias


class Provider(StrEnum):
    """
    LLM 服务提供商枚举
    
    定义了系统支持的所有 LLM 服务提供商，用于在配置中指定使用哪个提供商的服务。
    
    Attributes:
        OPENAI: OpenAI 官方 API
        OPENAI_COMPATIBLE: OpenAI 兼容的第三方 API（如 LM Studio、vLLM 等）
        AZURE_OPENAI: Microsoft Azure 上托管的 OpenAI 服务
        DEEPSEEK: DeepSeek（深度求索）AI
        ANTHROPIC: Anthropic（Claude 模型的提供商）
        GOOGLE: Google AI（Gemini 模型）
        VERTEXAI: Google Cloud Vertex AI 平台
        GROQ: Groq 高性能推理平台
        AWS: Amazon Web Services Bedrock 服务
        OLLAMA: Ollama 本地模型运行框架
        OPENROUTER: OpenRouter 模型路由服务
        FAKE: 用于测试的虚拟模型
    """
    OPENAI = auto()
    OPENAI_COMPATIBLE = auto()
    AZURE_OPENAI = auto()
    DEEPSEEK = auto()
    ANTHROPIC = auto()
    GOOGLE = auto()
    VERTEXAI = auto()
    GROQ = auto()
    AWS = auto()
    OLLAMA = auto()
    OPENROUTER = auto()
    FAKE = auto()


class OpenAIModelName(StrEnum):
    """
    OpenAI 模型名称枚举
    
    定义了 OpenAI 官方 API 支持的模型名称。
    
    参考文档: https://platform.openai.com/docs/models/gpt-4o
    
    Attributes:
        GPT_5_NANO: GPT-5 Nano 模型，适合轻量级任务
        GPT_5_MINI: GPT-5 Mini 模型，性价比较高
        GPT_5_1: GPT-5.1 模型，最新版本
    """
    GPT_5_NANO = "gpt-5-nano"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_1 = "gpt-5.1"


class AzureOpenAIModelName(StrEnum):
    """
    Azure OpenAI 模型名称枚举
    
    定义了 Azure OpenAI 服务支持的模型名称。Azure 上的模型名称可能与 OpenAI 官方略有不同。
    
    Attributes:
        AZURE_GPT_4O: Azure 上部署的 GPT-4o 模型
        AZURE_GPT_4O_MINI: Azure 上部署的 GPT-4o Mini 模型
    """
    AZURE_GPT_4O = "azure-gpt-4o"
    AZURE_GPT_4O_MINI = "azure-gpt-4o-mini"


class DeepseekModelName(StrEnum):
    """
    DeepSeek（深度求索）模型名称枚举
    
    定义了 DeepSeek AI 支持的模型名称。DeepSeek 是中国的 AI 公司，
    提供高性价比的大语言模型服务。
    
    参考文档: https://api-docs.deepseek.com/quick_start/pricing
    
    Attributes:
        DEEPSEEK_CHAT: DeepSeek Chat 模型，通用对话模型
    """
    DEEPSEEK_CHAT = "deepseek-chat"


class AnthropicModelName(StrEnum):
    """
    Anthropic（Claude）模型名称枚举
    
    定义了 Anthropic 公司的 Claude 系列模型名称。Claude 以安全性和有用性著称。
    
    参考文档: https://docs.anthropic.com/en/docs/about-claude/models#model-names
    
    Attributes:
        HAIKU_45: Claude Haiku 4.5，轻量快速模型
        SONNET_45: Claude Sonnet 4.5，平衡性能与成本
    """
    HAIKU_45 = "claude-haiku-4-5"
    SONNET_45 = "claude-sonnet-4-5"


class GoogleModelName(StrEnum):
    """
    Google AI（Gemini）模型名称枚举
    
    定义了 Google AI 的 Gemini 系列模型名称。Gemini 是 Google 最新的多模态 AI 模型。
    
    参考文档: https://ai.google.dev/gemini-api/docs/models/gemini
    
    Attributes:
        GEMINI_15_PRO: Gemini 1.5 Pro，强大的多模态模型
        GEMINI_20_FLASH: Gemini 2.0 Flash，快速响应模型
        GEMINI_20_FLASH_LITE: Gemini 2.0 Flash Lite，轻量版本
        GEMINI_25_FLASH: Gemini 2.5 Flash，新一代快速模型
        GEMINI_25_PRO: Gemini 2.5 Pro，高性能专业模型
        GEMINI_30_PRO: Gemini 3.0 Pro 预览版
    """
    GEMINI_15_PRO = "gemini-1.5-pro"
    GEMINI_20_FLASH = "gemini-2.0-flash"
    GEMINI_20_FLASH_LITE = "gemini-2.0-flash-lite"
    GEMINI_25_FLASH = "gemini-2.5-flash"
    GEMINI_25_PRO = "gemini-2.5-pro"
    GEMINI_30_PRO = "gemini-3-pro-preview"


class VertexAIModelName(StrEnum):
    """
    Google Cloud Vertex AI 模型名称枚举
    
    定义了通过 Google Cloud Vertex AI 平台访问的模型名称。
    Vertex AI 提供企业级的 AI 服务，适合生产环境部署。
    
    参考文档: https://cloud.google.com/vertex-ai/generative-ai/docs/models
    
    Attributes:
        GEMINI_15_PRO: Gemini 1.5 Pro
        GEMINI_20_FLASH: Gemini 2.0 Flash
        GEMINI_20_FLASH_LITE: Gemini 2.0 Flash Lite
        GEMINI_25_FLASH: Gemini 2.5 Flash
        GEMINI_25_PRO: Gemini 2.5 Pro
        GEMINI_30_PRO: Gemini 3.0 Pro 预览版
    """
    GEMINI_15_PRO = "gemini-1.5-pro"
    GEMINI_20_FLASH = "gemini-2.0-flash"
    GEMINI_20_FLASH_LITE = "models/gemini-2.0-flash-lite"
    GEMINI_25_FLASH = "models/gemini-2.5-flash"
    GEMINI_25_PRO = "gemini-2.5-pro"
    GEMINI_30_PRO = "gemini-3-pro-preview"


class GroqModelName(StrEnum):
    """
    Groq 模型名称枚举
    
    定义了 Groq 推理平台支持的模型名称。Groq 使用专用硬件（LPU）提供超快的推理速度。
    
    参考文档: https://console.groq.com/docs/models
    
    Attributes:
        LLAMA_31_8B: Llama 3.1 8B 模型
        LLAMA_33_70B: Llama 3.3 70B 模型
        LLAMA_GUARD_4_12B: Llama Guard 4 12B 内容安全模型
    """
    LLAMA_31_8B = "llama-3.1-8b"
    LLAMA_33_70B = "llama-3.3-70b"
    LLAMA_GUARD_4_12B = "meta-llama/llama-guard-4-12b"


class AWSModelName(StrEnum):
    """
    AWS Bedrock 模型名称枚举
    
    定义了 Amazon Bedrock 服务支持的模型名称。Bedrock 提供多个 AI 提供商的模型。
    
    参考文档: https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
    
    Attributes:
        BEDROCK_HAIKU: Claude 3.5 Haiku（通过 Bedrock 访问）
        BEDROCK_SONNET: Claude 3.5 Sonnet（通过 Bedrock 访问）
    """
    BEDROCK_HAIKU = "bedrock-3.5-haiku"
    BEDROCK_SONNET = "bedrock-3.5-sonnet"


class OllamaModelName(StrEnum):
    """
    Ollama 模型名称枚举
    
    定义了 Ollama 本地运行框架支持的模型。Ollama 允许在本地运行开源 LLM 模型，
    无需 API 密钥，适合离线使用和隐私敏感场景。
    
    参考文档: https://ollama.com/search
    
    Attributes:
        OLLAMA_GENERIC: 通用 Ollama 模型标识，实际模型需在配置中指定
    """
    OLLAMA_GENERIC = "ollama"


class OpenRouterModelName(StrEnum):
    """
    OpenRouter 模型名称枚举
    
    定义了 OpenRouter 路由服务支持的模型名称。OpenRouter 提供统一的 API
    来访问多个 AI 提供商的模型，便于切换和比较不同模型。
    
    参考文档: https://openrouter.ai/models
    
    Attributes:
        GEMINI_25_FLASH: 通过 OpenRouter 访问的 Gemini 2.5 Flash
    """
    GEMINI_25_FLASH = "google/gemini-2.5-flash"


class OpenAICompatibleName(StrEnum):
    """
    OpenAI 兼容 API 模型名称枚举
    
    用于支持任何实现了 OpenAI API 兼容接口的第三方服务，
    如 LM Studio、vLLM、LocalAI、Text Generation Inference 等。
    
    参考文档: https://platform.openai.com/docs/guides/text-generation
    
    Attributes:
        OPENAI_COMPATIBLE: 通用兼容模型标识
    """
    OPENAI_COMPATIBLE = "openai-compatible"


class FakeModelName(StrEnum):
    """
    虚拟模型名称枚举
    
    用于测试和开发目的的假模型，不会实际调用任何 LLM API。
    适用于单元测试、集成测试和 CI/CD 流水线。
    
    Attributes:
        FAKE: 虚拟测试模型
    """
    FAKE = "fake"


# 所有支持的模型类型别名
# 使用 TypeAlias 定义联合类型，包含所有提供商的模型枚举
# 这允许在类型注解中使用 AllModelEnum 来表示任意支持的模型
AllModelEnum: TypeAlias = (
    OpenAIModelName
    | OpenAICompatibleName
    | AzureOpenAIModelName
    | DeepseekModelName
    | AnthropicModelName
    | GoogleModelName
    | VertexAIModelName
    | GroqModelName
    | AWSModelName
    | OllamaModelName
    | OpenRouterModelName
    | FakeModelName
)
