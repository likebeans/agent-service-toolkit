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
    """`total=False` 是PEP589规范。

    文档：https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    safety: LlamaGuardOutput
    remaining_steps: RemainingSteps


tools = [database_search]


current_date = datetime.now().strftime("%Y年%m月%d日")
instructions = f"""
    你是AcmeBot，一个有用且知识渊博的虚拟助手，旨在通过检索和回答基于AcmeTech官方员工手册的问题来支持员工。你的主要职责是提供关于公司政策、价值观、程序和员工资源的准确、简洁且友好的信息。
    今天的日期是{current_date}。

    注意：用户看不到工具响应。

    需要记住的几件事：
    - 如果你有访问多个数据库的权限，在制定响应之前从不同的信息源收集信息。
    - 请在响应中包含markdown格式的链接到任何使用的引用。每个响应只包含一个或两个引用，除非需要更多。仅使用工具返回的链接。
    - 仅使用数据库中的信息。不要使用外部来源的信息。
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    """包装模型以使用工具。"""
    bound_model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | bound_model  # type: ignore[return-value]


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    """调用模型并检查安全性。"""
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # 在这里运行llama guard检查，以避免在不安全的情况下返回消息
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
    if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
        return {
            "messages": [format_safety_message(safety_output)],
            "safety": safety_output,
        }

    if state["remaining_steps"] < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="抱歉，需要更多步骤来处理此请求。",
                )
            ]
        }
    # 我们返回一个列表，因为这将被添加到现有列表中
    return {"messages": [response]}


def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    """格式化安全消息。"""
    content = (
        f"此对话被标记为不安全内容：{', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    """检查输入安全性。"""
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output, "messages": []}


async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    """阻止不安全内容。"""
    safety: LlamaGuardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}


# 定义图
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.add_node("guard_input", llama_guard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.set_entry_point("guard_input")


# 检查不安全的输入，如果发现则阻止进一步处理
def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    """检查安全性。"""
    safety: LlamaGuardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


agent.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "model"}
)

# 阻止不安全内容后总是结束
agent.add_edge("block_unsafe_content", END)

# 总是在"tools"之后运行"model"
agent.add_edge("tools", "model")


# 在"model"之后，如果有工具调用，运行"tools"。否则结束。
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    """检查待处理的工具调用。"""
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"期望AIMessage，得到{type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})

rag_assistant = agent.compile()
