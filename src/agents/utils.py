"""
本文件在整个项目中的角色：定义 Agent 向“流式 UI”发送自定义消息（custom event）的最小协议封装。

为什么这个文件存在？
- LangGraph 的 streaming 除了 `messages`（模型 token/消息）和 `updates`（节点状态更新）之外，
  还支持 `custom` 事件：Agent 可以在执行过程中主动推送结构化事件给客户端/UI。
- 本项目用这个能力实现了“后台任务进度条/状态流”（见 `src/agents/bg_task_agent/`），让用户看到
  Agent 正在做什么、进度如何，而不仅仅是等模型输出。

它解决的核心问题是什么？
- 将“自定义数据”包装成 LangChain 的 `ChatMessage(role="custom")`，这样：
  - 服务端 `src/service/service.py:message_generator()` 能在 `stream_mode="custom"` 下把它当成一条消息流出来
  - 服务端 `src/service/utils.py:langchain_to_chat_message()` 能识别 `role=="custom"` 并转成协议 `ChatMessage(type="custom")`
  - 前端 `src/streamlit_app.py:draw_messages()` 再根据 `custom_data` 做专门渲染（例如 Task 状态）

典型调用者是谁？
- 任何需要向 UI 推送“非聊天文本”的 Agent 节点：例如后台任务、阶段进度、指标统计等。
"""

from typing import Any

from langchain_core.messages import ChatMessage
from langgraph.types import StreamWriter
from pydantic import BaseModel, Field


class CustomData(BaseModel):
    """
    Agent 发出的自定义结构化数据。

    抽象含义：
    - 这不是给 LLM 的“自然语言消息”，而是给 UI/客户端的“结构化事件”。
    - 通过把它编码成 `ChatMessage(role="custom")`，我们复用服务端现有的消息流通道，
      不需要另起一套 SSE 协议或额外 endpoint。

    协作模块：
    - 发送侧：Agent 节点通过 `dispatch(writer)` 推送事件（writer 由 LangGraph runtime 注入）
    - 接收侧：服务端在 `stream_mode="custom"` 将其透传，前端在 `ChatMessage.type=="custom"` 时专门渲染
    """

    data: dict[str, Any] = Field(description="The custom data")

    def to_langchain(self) -> ChatMessage:
        """
        将自定义数据转换为 LangChain message。

        设计点：
        - content 使用 list 包裹，保持与服务端 `langchain_to_chat_message()` 的解析约定一致：
          `custom_data = message.content[0]`。
        """
        return ChatMessage(content=[self.data], role="custom")

    def dispatch(self, writer: StreamWriter) -> None:
        """
        通过 LangGraph 的 StreamWriter 立即把 custom message 推送到流中。

        在端到端链路中的位置：
        - 发生在 Agent 执行过程中（图节点内部），用于实时向前端“报状态/报进度”。
        """
        writer(self.to_langchain())
