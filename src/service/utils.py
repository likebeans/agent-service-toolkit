"""
本文件在整个项目中的角色：Service 层的“消息格式适配器（Adapter）”与小工具集合。

它解决的核心问题是什么？
- 服务端内部使用的是 LangChain/LangGraph 的消息类型（`HumanMessage/AIMessage/ToolMessage/...`）；
- 对外 API 需要一个稳定、可序列化、与前端/客户端解耦的消息结构（`schema.ChatMessage`）；
- 因此这里负责把“内部消息对象”转换为“对外契约模型”，并在流式输出（SSE token）时做内容清洗。

典型调用者：
- `src/service/service.py`：
  - `invoke()`：把最终 LangChain message 转成 `ChatMessage` 返回 JSON
  - `message_generator()`：把中间消息/自定义事件转为 `ChatMessage` 并编码为 SSE
"""

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.messages import (
    ChatMessage as LangchainChatMessage,
)

from schema import ChatMessage


def convert_message_content_to_string(content: str | list[str | dict]) -> str:
    """
    将 LangChain message.content 规范化为字符串。

    背景：
    - 在 LangChain 里，message.content 既可能是纯字符串，也可能是一个“分块结构”的 list：
      例如 [{"type": "text", "text": "..."}, {"type": "tool_use", ...}]。
    - 对外 `schema.ChatMessage.content` 在本项目中选择用字符串承载，因此需要做一次归一化。

    注意：
    - 这里仅拼接 type=="text" 的内容；其它类型（例如 tool_use）会被忽略。
      这和 `remove_tool_calls()` 的目标一致：避免把工具调用元信息当成自然语言 token 输出。
    """
    if isinstance(content, str):
        return content
    text: list[str] = []
    for content_item in content:
        if isinstance(content_item, str):
            text.append(content_item)
            continue
        if content_item["type"] == "text":
            text.append(content_item["text"])
    return "".join(text)


def langchain_to_chat_message(message: BaseMessage) -> ChatMessage:
    """
    将 LangChain 的消息对象转换为对外契约 `schema.ChatMessage`。

    端到端链路中的位置：
    - 这是服务端“出站（egress）”的关键适配点：把内部结构化对象变成可 JSON 序列化的 schema。

    设计意图：
    - 只暴露客户端真正需要的信息（type/content/tool_calls/metadata/run_id 等），
      避免把 LangChain 内部类结构泄漏给前端，降低耦合度。
    """
    match message:
        case HumanMessage():
            human_message = ChatMessage(
                type="human",
                content=convert_message_content_to_string(message.content),
            )
            return human_message
        case AIMessage():
            ai_message = ChatMessage(
                type="ai",
                content=convert_message_content_to_string(message.content),
            )
            if message.tool_calls:
                ai_message.tool_calls = message.tool_calls
            if message.response_metadata:
                ai_message.response_metadata = message.response_metadata
            return ai_message
        case ToolMessage():
            tool_message = ChatMessage(
                type="tool",
                content=convert_message_content_to_string(message.content),
                tool_call_id=message.tool_call_id,
            )
            return tool_message
        case LangchainChatMessage():
            # `LangchainChatMessage` 是更通用的“role + content”消息表示。
            # 本项目利用它承载一些“自定义事件/结构”（role == "custom"）。
            if message.role == "custom":
                custom_message = ChatMessage(
                    type="custom",
                    content="",
                    custom_data=message.content[0],
                )
                return custom_message
            else:
                raise ValueError(f"Unsupported chat message role: {message.role}")
        case _:
            raise ValueError(f"Unsupported message type: {message.__class__.__name__}")


def remove_tool_calls(content: str | list[str | dict]) -> str | list[str | dict]:
    """
    从 content 中移除“工具调用”相关的分块，避免把它们当作自然语言 token 输出。

    背景：
    - 某些模型/SDK（尤其 Anthropic）在 streaming 时会把工具调用以 content item 的形式混入输出；
    - 服务端 `/stream` 会把 token 事件透传给客户端做增量渲染，
      因此需要先把 tool_use 这类非文本内容过滤掉。
    """
    if isinstance(content, str):
        return content
    # 当前主要是 Anthropic 模型会在流式输出中夹带 tool_use 分块。
    return [
        content_item
        for content_item in content
        if isinstance(content_item, str) or content_item["type"] != "tool_use"
    ]
