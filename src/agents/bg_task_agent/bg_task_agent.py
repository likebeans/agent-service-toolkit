"""
本文件在整个项目中的角色：演示“后台任务/进度事件”如何在 Agent 执行过程中通过流式通道推送给前端。

为什么这个文件存在？
- 真实的 Agent 往往不只是“等模型吐字”，还会执行一些长耗时步骤（检索、多阶段流程、外部 API 调用等）。
- 用户体验上，我们希望能看到“当前在做什么、进度如何、是否出错”，而不是空等。
- LangGraph 支持 `StreamWriter` 与 `stream_mode="custom"`：允许图节点在执行时主动向流里写入自定义事件。
  这个 Agent 用最小代码展示了这条能力链路如何打通到 Streamlit UI。

端到端链路（简化）：
1) 本文件的 `bg_task` 节点用 `Task`（`src/agents/bg_task_agent/task.py`）产出 custom 消息并写入 StreamWriter
2) 服务端 `src/service/service.py:message_generator()` 以 `stream_mode=["custom", ...]` 读取并透传
3) `src/service/utils.py:langchain_to_chat_message()` 把 role="custom" 的消息转成 `ChatMessage(type="custom")`
4) 前端 `src/streamlit_app.py:draw_messages()` 针对 custom 消息走 `TaskDataStatus` 渲染（见 `src/schema/task_data.py`）

注意（不改变行为的约束）：
- 本文件的核心目的是“演示流式自定义事件”，我们只添加注释，不调整 sleep/状态机等逻辑。
"""

import asyncio

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.types import StreamWriter

from agents.bg_task_agent.task import Task
from core import get_model, settings


class AgentState(MessagesState, total=False):
    """
    背景任务 Agent 的状态（State）。

    说明：
    - 继承 `MessagesState`，主要承载对话消息列表 `messages`。
    - 这里没有额外字段；但仍保留 TypedDict 形式，保持与其它 Agent 一致的建图范式。
    - `total=False`：字段可缺省，适配 LangGraph “按节点增量更新 state” 的机制。

    参考：https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    """
    将模型包装成一个 runnable，使其输入为 AgentState，输出为 AIMessage。

    与其它 Agent 的区别：
    - 这里没有系统提示词，也没有 tools；只是把 state["messages"] 原样交给模型。
    - 目的是把焦点放在“bg_task 节点如何发 custom 事件”，而不是提示词工程或工具调用。
    """
    preprocessor = RunnableLambda(
        lambda state: state["messages"],
        name="StateModifier",
    )
    return preprocessor | model  # type: ignore[return-value]


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    图节点：调用 LLM 生成一条 AI 回复。

    在端到端链路中的位置：
    - `bg_task` 节点跑完（发完进度事件）后，进入该节点，给用户一个最终“文本回复”。
    """
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # 返回 list 的原因：LangGraph 会把该列表追加到已有 messages 中（增量更新语义）。
    return {"messages": [response]}


async def bg_task(state: AgentState, writer: StreamWriter) -> AgentState:
    """
    图节点：模拟两个后台任务的执行过程，并通过 StreamWriter 把进度事件流式推送给客户端。

    关键抽象：
    - `writer` 是 LangGraph 运行时注入的“流写入器”，写入的消息会在服务端被当作 custom 事件透传。
    - `Task` 是一个小封装：负责把状态（new/running/complete）编码为 custom 消息并 dispatch。

    你在自己实现时要注意：
    - custom 事件最好是结构化的（dict），这样 UI 才能稳定解析与渲染；本项目用 `TaskData` 作为结构。
    - 事件频率要可控（这里用 sleep 模拟节奏）；真实场景要避免过高频率导致前端渲染压力。
    """
    task1 = Task("Simple task 1...", writer)
    task2 = Task("Simple task 2...", writer)

    task1.start()
    await asyncio.sleep(2)
    task2.start()
    await asyncio.sleep(2)
    task1.write_data(data={"status": "Still running..."})
    await asyncio.sleep(2)
    task2.finish(result="error", data={"output": 42})
    await asyncio.sleep(2)
    task1.finish(result="success", data={"output": 42})
    return {"messages": []}


# -----------------------------
# 图定义：bg_task（发进度） -> model（给最终回复） -> END
# -----------------------------
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("bg_task", bg_task)
agent.set_entry_point("bg_task")

agent.add_edge("bg_task", "model")
agent.add_edge("model", END)

bg_task_agent = agent.compile()
