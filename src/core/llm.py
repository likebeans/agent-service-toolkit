"""
本文件在整个项目中的角色：LLM “模型工厂”（Model Factory / Provider Adapter）。

它解决的核心问题是什么？
- 统一不同厂商/不同 SDK 的初始化方式（OpenAI / Azure OpenAI / Anthropic / Google / VertexAI / Groq / Bedrock / Ollama / OpenRouter 等）。
- 让上层（Agent Graph）只关心“我要用哪个模型名（enum）”，而不用关心：
  - 该模型属于哪个 Provider
  - 需要哪些环境变量/鉴权参数
  - 是否支持 streaming（token 流式）

典型调用者：
- `src/agents/*`：在图的 node 中通过 `get_model(configurable['model'] or settings.DEFAULT_MODEL)` 获取可调用的 ChatModel。

与端到端链路的关系：
- 服务端 `/stream`（见 `src/service/service.py:message_generator()`）会在 `stream_mode="messages"` 时产出 token。
- 这里把大多数模型初始化为 `streaming=True`，因此当客户端请求 `stream_tokens=True` 时，
  UI 能够实时收到 token（参见 `src/client/client.py:_parse_stream_line()` 对 "token" 事件的处理）。
"""

from functools import cache
from typing import TypeAlias

from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_community.chat_models import FakeListChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from core.settings import settings
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
    VertexAIModelName,
)

# `_MODEL_TABLE` 将“项目内统一的模型枚举”（AllModelEnum 及其子枚举）映射到真正的 API 模型名字符串。
# 设计意图：
# - 上层只传 enum，避免字符串拼写错误
# - 在不同 provider SDK 中都能拿到同一个“可用于初始化的 model 名称”
_MODEL_TABLE = (
    {m: m.value for m in OpenAIModelName}
    | {m: m.value for m in OpenAICompatibleName}
    | {m: m.value for m in AzureOpenAIModelName}
    | {m: m.value for m in DeepseekModelName}
    | {m: m.value for m in AnthropicModelName}
    | {m: m.value for m in GoogleModelName}
    | {m: m.value for m in VertexAIModelName}
    | {m: m.value for m in GroqModelName}
    | {m: m.value for m in AWSModelName}
    | {m: m.value for m in OllamaModelName}
    | {m: m.value for m in OpenRouterModelName}
    | {m: m.value for m in FakeModelName}
)


class FakeToolModel(FakeListChatModel):
    """
    测试用模型：在不接入任何真实 LLM Provider 的情况下，模拟一个可“绑定 tools” 的 ChatModel。

    为什么需要它？
    - LangChain 的部分 Agent/Graph 会对模型调用 `.bind_tools(...)`；
    - `FakeListChatModel` 默认未必满足所有调用路径，因此这里做一个最小适配，让测试更稳定。
    """

    def __init__(self, responses: list[str]):
        super().__init__(responses=responses)

    def bind_tools(self, tools):
        # 对 Fake 模型来说不需要真的绑定 tool schema，只要保持链式调用不报错即可。
        return self


# 统一对外暴露的“模型实例类型”：
# - 上层只需要知道它是一个可调用的 ChatModel（具备 ainvoke/astream 等能力），不需要关心具体实现类。
ModelT: TypeAlias = (
    AzureChatOpenAI
    | ChatOpenAI
    | ChatAnthropic
    | ChatGoogleGenerativeAI
    | ChatVertexAI
    | ChatGroq
    | ChatBedrock
    | ChatOllama
    | FakeToolModel
)


@cache
def get_model(model_name: AllModelEnum, /) -> ModelT:
    """
    根据“统一模型枚举”创建（或复用）一个可用的 ChatModel 实例。

    端到端链路中的位置：
    - Agent 在执行某个 node 时调用本函数，得到“可与 LLM 交互”的对象；
      然后 LangGraph 通过该对象生成回复/工具调用/structured output 等。

    设计取舍：
    - 使用 `@cache`：同一个进程内，相同的 `model_name` 只会初始化一次，后续复用实例。
      这能减少 SDK 初始化开销，并让服务在高并发下更稳定。

    注意：
    - 对于支持 streaming 的模型，这里通常设置 `streaming=True`：
      当服务端 `/stream` 且 `stream_tokens=True` 时，客户端可以收到 token 事件并实时渲染。
    """

    # 注意：streaming=True 的模型会在生成时不断输出 token。
    # 只有当服务端 `/stream` 端点被调用且 `stream_tokens=True`（默认）时，这些 token 才会透传给客户端。
    api_model_name = _MODEL_TABLE.get(model_name)
    if not api_model_name:
        raise ValueError(f"Unsupported model: {model_name}")

    # 下面按“模型名属于哪个枚举集合”来判断 provider，并拼装对应 SDK 的初始化参数。
    # 这种写法比在外部维护一个 “model -> provider” 表更直观，但也意味着新增 provider 时要改这里的分支。
    if model_name in OpenAIModelName:
        return ChatOpenAI(model=api_model_name, streaming=True)
    if model_name in OpenAICompatibleName:
        if not settings.COMPATIBLE_BASE_URL or not settings.COMPATIBLE_MODEL:
            raise ValueError("OpenAICompatible base url and endpoint must be configured")

        return ChatOpenAI(
            model=settings.COMPATIBLE_MODEL,
            temperature=0.5,
            streaming=True,
            openai_api_base=settings.COMPATIBLE_BASE_URL,
            openai_api_key=settings.COMPATIBLE_API_KEY,
        )
    if model_name in AzureOpenAIModelName:
        if not settings.AZURE_OPENAI_API_KEY or not settings.AZURE_OPENAI_ENDPOINT:
            raise ValueError("Azure OpenAI API key and endpoint must be configured")

        # Azure OpenAI 的关键区别：
        # - 不是直接传 model name，而是传 deployment_name（由用户在 Azure 侧创建并映射）
        return AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            deployment_name=api_model_name,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            temperature=0.5,
            streaming=True,
            timeout=60,
            max_retries=3,
        )
    if model_name in DeepseekModelName:
        # DeepSeek 使用 OpenAI 兼容协议，但 base_url 固定为 deepseek API。
        return ChatOpenAI(
            model=api_model_name,
            temperature=0.5,
            streaming=True,
            openai_api_base="https://api.deepseek.com",
            openai_api_key=settings.DEEPSEEK_API_KEY,
        )
    if model_name in AnthropicModelName:
        return ChatAnthropic(model=api_model_name, temperature=0.5, streaming=True)
    if model_name in GoogleModelName:
        return ChatGoogleGenerativeAI(model=api_model_name, temperature=0.5, streaming=True)
    if model_name in VertexAIModelName:
        return ChatVertexAI(model=api_model_name, temperature=0.5, streaming=True)
    if model_name in GroqModelName:
        if model_name == GroqModelName.LLAMA_GUARD_4_12B:
            # Llama Guard 是安全/审核模型：通常不需要随机性，因此 temperature=0 更合适。
            return ChatGroq(model=api_model_name, temperature=0.0)  # type: ignore[call-arg]
        return ChatGroq(model=api_model_name, temperature=0.5)  # type: ignore[call-arg]
    if model_name in AWSModelName:
        return ChatBedrock(model_id=api_model_name, temperature=0.5)
    if model_name in OllamaModelName:
        # Ollama 支持本地/自托管：base_url 可配置；model 名通常来自 settings.OLLAMA_MODEL。
        if settings.OLLAMA_BASE_URL:
            chat_ollama = ChatOllama(
                model=settings.OLLAMA_MODEL, temperature=0.5, base_url=settings.OLLAMA_BASE_URL
            )
        else:
            chat_ollama = ChatOllama(model=settings.OLLAMA_MODEL, temperature=0.5)
        return chat_ollama
    if model_name in OpenRouterModelName:
        # OpenRouter 使用 OpenAI 兼容 API，但 base_url 指向 openrouter。
        return ChatOpenAI(
            model=api_model_name,
            temperature=0.5,
            streaming=True,
            base_url="https://openrouter.ai/api/v1/",
            api_key=settings.OPENROUTER_API_KEY,
        )
    if model_name in FakeModelName:
        # 仅用于测试/本地验证：不需要任何外部 API key。
        return FakeToolModel(responses=["This is a test response from the fake model."])

    raise ValueError(f"Unsupported model: {model_name}")
