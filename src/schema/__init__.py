"""
Schema 模块初始化文件

本模块提供了 Agent Service Toolkit 的核心数据模型和类型定义，包括：
- 模型枚举类型（AllModelEnum）：支持的所有 LLM 模型枚举
- 用户输入相关（UserInput, StreamInput）：处理用户请求的数据结构
- 消息相关（ChatMessage, ChatHistory）：对话消息的数据结构
- 服务元数据（ServiceMetadata, AgentInfo）：服务和代理的配置信息
- 反馈相关（Feedback, FeedbackResponse）：用户反馈的数据结构
"""

from schema.models import AllModelEnum
from schema.schema import (
    AgentInfo,
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    ServiceMetadata,
    StreamInput,
    UserInput,
)

# 模块公开导出的所有类和类型
__all__ = [
    "AgentInfo",          # Agent 信息模型
    "AllModelEnum",       # 所有支持的模型枚举类型
    "UserInput",          # 用户输入模型
    "ChatMessage",        # 聊天消息模型
    "ServiceMetadata",    # 服务元数据模型
    "StreamInput",        # 流式输入模型
    "Feedback",           # 反馈模型
    "FeedbackResponse",   # 反馈响应模型
    "ChatHistoryInput",   # 获取聊天历史的输入模型
    "ChatHistory",        # 聊天历史模型
]
