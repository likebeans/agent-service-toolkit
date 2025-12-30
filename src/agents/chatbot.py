"""
本文件在整个项目中的角色：提供最小可运行的“聊天机器人”Agent（用于入门/对照）。

为什么这个文件存在？
- 这是整个仓库里最“轻量”的 Agent：不接工具、不做复杂路由，只有“把对话喂给模型 -> 返回回复”。
- 它使用 LangGraph 的函数式 API（`langgraph.func.entrypoint`）而不是 `StateGraph`：
  - 用于演示：LangGraph 不只有“画图式”的 StateGraph，也支持“函数式工作流”风格。

典型调用者是谁？
- `src/agents/agents.py`：注册为 `"chatbot"` 供服务 `/invoke`、`/stream` 调用

与哪些模块协作？
- 模型工厂：`src/core/llm.py:get_model`
- 服务层会在启动期给图注入 checkpointer/store（见 `src/service/service.py:lifespan()`），
  使得 `save={...}` 的内容能够持久化（取决于数据库配置）。

注意（学习点）：
- 这个 Agent 展示了 LangGraph 对“多轮对话”的一个关键机制：`previous` 参数 + `save` 字段。
  你可以把它理解为：函数每次只处理“本轮输入”，但框架会把历史状态（previous）传进来，并允许你保存新状态。
"""

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.func import entrypoint

from core import get_model, settings


@entrypoint()
async def chatbot(
    inputs: dict[str, list[BaseMessage]],
    *,
    previous: dict[str, list[BaseMessage]],
    config: RunnableConfig,
):
    """
    最小聊天 Agent 的执行函数（LangGraph entrypoint 风格）。

    在端到端链路中的位置：
    - 服务端 `/invoke` / `/stream` 调用图时，最终会执行到这个 entrypoint。

    参数语义：
    - inputs：本轮新增输入（通常是 {"messages": [HumanMessage(...)]}）
    - previous：历史状态（由 checkpointer 恢复），用于实现多轮对话
    - config：运行时配置（thread_id/user_id/model 等都在 config["configurable"] 里）

    返回语义：
    - `entrypoint.final(value=..., save=...)`：
      - value：本次执行的“输出”（服务端通常会取最后一条消息返回给用户）
      - save：需要写入持久化的“新状态”（下一轮作为 previous 传入）
    """
    messages = inputs["messages"]
    # previous 非空表示这是同一 thread 的续聊：把历史消息拼到本轮输入前面，形成完整上下文。
    if previous:
        messages = previous["messages"] + messages

    # 模型选择来自 config；如果请求没指定 model，则回退到 settings.DEFAULT_MODEL。
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    response = await model.ainvoke(messages)
    return entrypoint.final(
        value={"messages": [response]}, save={"messages": messages + [response]}
    )
