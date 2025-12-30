"""
核心数据模型定义模块

本模块定义了 Agent Service Toolkit 的核心数据模型，包括：
- 用户输入模型：处理来自客户端的请求
- 消息模型：表示对话中的各类消息
- 服务元数据：描述服务和 Agent 的配置信息
- 反馈模型：记录用户对响应的反馈

这些模型基于 Pydantic，提供了自动验证、序列化和文档生成功能。
"""

from typing import Any, Literal, NotRequired

from pydantic import BaseModel, Field, SerializeAsAny
from typing_extensions import TypedDict

from schema.models import AllModelEnum, AnthropicModelName, OpenAIModelName


class AgentInfo(BaseModel):
    """
    Agent 信息模型
    
    描述一个可用 Agent 的基本信息，用于服务发现和 Agent 选择。
    
    属性：
        key: Agent 的唯一标识符，用于在 API 调用中指定使用哪个 Agent
        description: Agent 的功能描述，帮助用户了解该 Agent 的用途
    
    示例：
        >>> agent = AgentInfo(
        ...     key="research-assistant",
        ...     description="一个用于生成研究报告的助手"
        ... )
    """
    key: str = Field(
        description="Agent 的唯一标识符",
        examples=["research-assistant"],
    )
    description: str = Field(
        description="Agent 的功能描述",
        examples=["一个用于生成研究报告的助手"],
    )


class ServiceMetadata(BaseModel):
    """
    服务元数据模型
    
    描述整个服务的配置信息，包括可用的 Agent 列表、支持的模型列表，
    以及默认的 Agent 和模型设置。客户端可以通过这些信息了解服务的能力。
    
    属性：
        agents: 可用 Agent 列表
        models: 支持的 LLM 模型列表
        default_agent: 未指定时使用的默认 Agent
        default_model: 未指定时使用的默认模型
    """
    agents: list[AgentInfo] = Field(
        description="可用的 Agent 列表",
    )
    models: list[AllModelEnum] = Field(
        description="支持的 LLM 模型列表",
    )
    default_agent: str = Field(
        description="默认使用的 Agent（未指定时）",
        examples=["research-assistant"],
    )
    default_model: AllModelEnum = Field(
        description="默认使用的模型（未指定时）",
    )


class UserInput(BaseModel):
    """
    用户输入模型
    
    表示用户发送给 Agent 的基本输入，包含消息内容和各种配置选项。
    这是与 Agent 交互的主要数据结构。
    
    属性：
        message: 用户输入的消息内容
        model: 可选，指定使用的 LLM 模型，默认使用服务配置的默认模型
        thread_id: 可选，会话线程 ID，用于保持多轮对话的上下文
        user_id: 可选，用户 ID，用于跨多个会话保持用户状态
        agent_config: 可选，传递给 Agent 的额外配置参数
    
    示例：
        >>> user_input = UserInput(
        ...     message="今天东京的天气怎么样？",
        ...     thread_id="847c6285-8fc9-4560-a83f-4e6285809254"
        ... )
    """
    message: str = Field(
        description="用户输入的消息内容",
        examples=["今天东京的天气怎么样？"],
    )
    model: SerializeAsAny[AllModelEnum] | None = Field(
        title="模型",
        description="使用的 LLM 模型。如果不指定，将使用服务配置的默认模型。",
        default=None,
        examples=[OpenAIModelName.GPT_5_NANO, AnthropicModelName.HAIKU_45],
    )
    thread_id: str | None = Field(
        description="会话线程 ID，用于保持多轮对话的上下文。",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    user_id: str | None = Field(
        description="用户 ID，用于跨多个会话线程保持用户状态。",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    agent_config: dict[str, Any] = Field(
        description="传递给 Agent 的额外配置参数",
        default={},
        examples=[{"spicy_level": 0.8}],
    )


class StreamInput(UserInput):
    """
    流式输入模型
    
    继承自 UserInput，增加了流式输出的控制选项。
    用于需要实时获取 Agent 响应的场景。
    
    属性：
        stream_tokens: 是否将 LLM 生成的 token 流式传输到客户端。
                       设为 True 可以实现打字机效果的实时输出。
    """
    stream_tokens: bool = Field(
        description="是否将 LLM 生成的 token 流式传输到客户端",
        default=True,
    )


class ToolCall(TypedDict):
    """
    工具调用数据结构
    
    表示 LLM 请求调用工具/函数的信息。当 Agent 需要执行外部操作时，
    会生成 ToolCall 来描述要调用的工具及其参数。
    
    属性：
        name: 要调用的工具名称
        args: 工具调用的参数字典
        id: 工具调用的唯一标识符，用于匹配工具调用和响应
        type: 类型标识，固定为 "tool_call"（可选）
    """
    name: str
    """要调用的工具名称"""
    args: dict[str, Any]
    """工具调用的参数字典"""
    id: str | None
    """工具调用的唯一标识符"""
    type: NotRequired[Literal["tool_call"]]


class ChatMessage(BaseModel):
    """
    聊天消息模型
    
    表示对话中的一条消息，可以是人类消息、AI 消息、工具消息或自定义消息。
    这是对话历史的基本组成单元。
    
    属性：
        type: 消息类型，可选值为：
              - "human": 人类用户发送的消息
              - "ai": AI 助手生成的消息
              - "tool": 工具执行结果的消息
              - "custom": 自定义类型的消息
        content: 消息的文本内容
        tool_calls: AI 消息中包含的工具调用列表
        tool_call_id: 如果是工具响应消息，这里是对应的工具调用 ID
        run_id: 消息所属的运行 ID，用于追踪和调试
        response_metadata: 响应元数据，如 HTTP 头、token 计数、logprobs 等
        custom_data: 自定义数据，用于存储额外的业务信息
    """
    type: Literal["human", "ai", "tool", "custom"] = Field(
        description="消息的角色/类型",
        examples=["human", "ai", "tool", "custom"],
    )
    content: str = Field(
        description="消息的文本内容",
        examples=["你好，世界！"],
    )
    tool_calls: list[ToolCall] = Field(
        description="消息中包含的工具调用列表",
        default=[],
    )
    tool_call_id: str | None = Field(
        description="此消息响应的工具调用 ID（仅工具消息使用）",
        default=None,
        examples=["call_Jja7J89XsjrOLA5r!MEOW!SL"],
    )
    run_id: str | None = Field(
        description="消息所属的运行 ID",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    response_metadata: dict[str, Any] = Field(
        description="响应元数据，例如：响应头、logprobs、token 计数等",
        default={},
    )
    custom_data: dict[str, Any] = Field(
        description="自定义消息数据",
        default={},
    )

    def pretty_repr(self) -> str:
        """
        获取消息的美化字符串表示
        
        生成一个格式化的字符串，包含带分隔线的标题和消息内容，
        便于在控制台或日志中查看。
        
        返回：
            格式化的消息字符串，包含类型标题和内容
            
        示例：
            >>> msg = ChatMessage(type="human", content="你好")
            >>> print(msg.pretty_repr())
            =============================== Human Message ================================
            
            你好
        """
        base_title = self.type.title() + " Message"
        padded = " " + base_title + " "
        sep_len = (80 - len(padded)) // 2
        sep = "=" * sep_len
        second_sep = sep + "=" if len(padded) % 2 else sep
        title = f"{sep}{padded}{second_sep}"
        return f"{title}\n\n{self.content}"

    def pretty_print(self) -> None:
        """
        打印消息的美化表示
        
        将 pretty_repr() 的结果直接打印到标准输出。
        """
        print(self.pretty_repr())  # noqa: T201


class Feedback(BaseModel):  # type: ignore[no-redef]
    """
    反馈模型
    
    用于记录用户对 Agent 响应的反馈，反馈数据会发送到 LangSmith 进行分析。
    这对于评估和改进 Agent 的表现非常重要。
    
    属性：
        run_id: 要记录反馈的运行 ID
        key: 反馈的类型/键名，如 "human-feedback-stars"
        score: 反馈分数，通常在 0.0 到 1.0 之间
        kwargs: 传递给 LangSmith 的额外参数，如评论等
        
    示例：
        >>> feedback = Feedback(
        ...     run_id="847c6285-8fc9-4560-a83f-4e6285809254",
        ...     key="human-feedback-stars",
        ...     score=0.8,
        ...     kwargs={"comment": "回答很有帮助"}
        ... )
    """
    run_id: str = Field(
        description="要记录反馈的运行 ID",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    key: str = Field(
        description="反馈类型/键名",
        examples=["human-feedback-stars"],
    )
    score: float = Field(
        description="反馈分数",
        examples=[0.8],
    )
    kwargs: dict[str, Any] = Field(
        description="传递给 LangSmith 的额外反馈参数",
        default={},
        examples=[{"comment": "内嵌人工反馈"}],
    )


class FeedbackResponse(BaseModel):
    """
    反馈响应模型
    
    表示反馈提交成功的响应。目前只包含一个表示成功的状态字段。
    
    属性：
        status: 响应状态，固定为 "success"
    """
    status: Literal["success"] = "success"


class ChatHistoryInput(BaseModel):
    """
    聊天历史查询输入模型
    
    用于请求获取特定会话线程的聊天历史记录。
    
    属性：
        thread_id: 要查询的会话线程 ID
        
    示例：
        >>> input = ChatHistoryInput(
        ...     thread_id="847c6285-8fc9-4560-a83f-4e6285809254"
        ... )
    """
    thread_id: str = Field(
        description="要查询的会话线程 ID",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )


class ChatHistory(BaseModel):
    """
    聊天历史模型
    
    表示一个会话线程的完整消息历史记录。
    
    属性：
        messages: 按时间顺序排列的消息列表
    """
    messages: list[ChatMessage]
