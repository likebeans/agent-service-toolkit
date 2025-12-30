"""
本文件在整个项目中的角色：实现 `rag-assistant` Agent（基于向量库检索的 RAG 示例）。

为什么这个文件存在？
- 相比 `research-assistant` 的“联网搜索 + 计算”，这里演示另一类常见产品形态：企业知识库问答（RAG）。
- 该 Agent 的核心流程仍然是 LangGraph 的“模型-工具循环”，但工具变为：
  - `Database_Search`：从本地 Chroma 向量库检索相关文档片段（见 `src/agents/tools.py`）
- 同时保留了与默认 Agent 一致的“输入/输出安全审核”模式（LlamaGuard），便于复用与对比学习。

它解决的核心问题是什么？
- 让模型能够“只基于知识库内容回答”，把企业政策/手册等离线文本变成可对话的知识助手。

典型调用者是谁？
- `src/agents/agents.py`：注册 `"rag-assistant"`
- `src/service/service.py`：通过 `/invoke`、`/stream` 调用本图

注意（不改变行为的约束）：
- 变量 `instructions` 是系统提示词（会发给 LLM），修改（包括翻译）会改变回答边界与风格。
  因此我们只添加中文解释，不改动其内容。
"""

from datetime import datetime
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from agents.tools import database_search
from core import get_model, settings


class AgentState(MessagesState, total=False):
    """
    RAG Agent 的状态（State），基于 MessagesState 扩展。

    字段说明：
    - messages：对话与工具消息历史（来自 MessagesState）
    - safety：LlamaGuard 审核结果（用于条件路由）
    - remaining_steps：LangGraph 步数预算（限制工具循环次数）

    `total=False` 说明：TypedDict 字段允许缺省，适配“节点按需写入 state”的图执行模式。
    参考：https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    safety: LlamaGuardOutput
    remaining_steps: RemainingSteps


tools = [database_search]


current_date = datetime.now().strftime("%B %d, %Y")
# 注意：这是系统提示词（会发给 LLM），修改会改变行为；此处只加解释不改文本。
instructions = f"""
    You are AcmeBot, a helpful and knowledgeable virtual assistant designed to support employees by retrieving
    and answering questions based on AcmeTech's official Employee Handbook. Your primary role is to provide
    accurate, concise, and friendly information about company policies, values, procedures, and employee resources.
    Today's date is {current_date}.

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    A few things to remember:
    - If you have access to multiple databases, gather information from a diverse range of sources before crafting your response.
    - Please include markdown-formatted links to any citations used in your response. Only include one
    or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
    - Only use information from the database. Do not use information from outside sources.
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    """
    将模型包装成“带系统提示词 + 绑定工具”的 Runnable。

    与 `research_assistant.wrap_model` 基本一致：
    - 差别在于 tools 只有一个 database_search（向量库检索），用于实现 RAG。
    """
    bound_model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | bound_model  # type: ignore[return-value]


def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    """将安全拦截结果转成可返回给用户的提示消息。"""
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    图节点：调用模型生成下一条 AI 消息（可能包含 tool_calls）。

    业务意义：
    - 在 RAG 场景下，模型通常会先调用 `Database_Search` 拿到上下文，再生成最终回答。
    """
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # 输出侧安全审核：避免把不安全的模型输出返回给用户。
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
    if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
        return {
            "messages": [format_safety_message(safety_output)],
            "safety": safety_output,
        }

    # 步数预算不足但模型仍想调用工具时，直接返回提示，避免图执行异常或体验不确定。
    if state["remaining_steps"] < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
    # 返回 list 以便 LangGraph 追加到现有 messages。
    return {"messages": [response]}


async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    """图节点：对用户输入做安全审核（输入侧防护）。"""
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output, "messages": []}


async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    """图节点：命中不安全输入时，返回拦截提示并终止执行。"""
    safety: LlamaGuardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}


# -----------------------------
# 图定义：guard_input ->（不安全则 block，否则 model）->（有 tool_calls 则 tools，否则 END）
# -----------------------------
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.add_node("guard_input", llama_guard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.set_entry_point("guard_input")


# 条件边：根据输入审核结果选择“拦截”或“继续”。
def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: LlamaGuardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


agent.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "model"}
)

# 一旦拦截，直接结束。
agent.add_edge("block_unsafe_content", END)

# 工具执行后回到模型，让模型基于检索结果作答。
agent.add_edge("tools", "model")


# 条件边：如果模型产生 tool_calls，则进入 tools；否则结束。
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})

rag_assistant = agent.compile()
