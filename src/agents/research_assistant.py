"""
本文件在整个项目中的角色：实现默认的 `research-assistant` Agent（LangGraph 图）。

为什么这个文件存在？
- 这是仓库的“默认示例 Agent”，用最典型的方式演示：
  1) 如何把 LLM + Tools（搜索/计算/天气）组合成一个可循环执行的 ReAct-ish 工作流
  2) 如何在 Agent 内部做输入/输出安全审核（LlamaGuard）
  3) 如何通过 LangGraph 的 StateGraph 定义：节点、条件边、以及工具循环

它解决的核心问题是什么？
- 在一次对话请求中，允许模型：
  - 先规划（决定是否需要工具）
  - 触发工具调用（WebSearch/Calculator/Weather）
  - 将工具结果喂回模型继续推理
  - 最终输出答案（带引用链接）

典型调用者是谁？
- `src/agents/agents.py`：将本图注册为 `"research-assistant"`，并作为 `DEFAULT_AGENT`
- `src/service/service.py`：
  - `/invoke` 与 `/stream` 会通过 `get_agent("research-assistant")` 拿到此图并执行

与哪些模块协作？
- 工具：`src/agents/tools.py:calculator` + LangChain 社区工具（DuckDuckGo/Weather）
- 内容安全：`src/agents/llama_guard.py:LlamaGuard`
- 模型选择：`src/core/llm.py:get_model`（通过 RunnableConfig.configurable["model"] 决定）

注意（不改变行为的约束）：
- 变量 `instructions` 是系统提示词（System Prompt），会直接发送给 LLM。
  翻译/改写会改变 Agent 行为与回答风格；因此我们只添加中文解释，不改动其内容。
"""

from datetime import datetime
from typing import Literal

from langchain_community.tools import DuckDuckGoSearchResults, OpenWeatherMapQueryRun
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from agents.tools import calculator
from core import get_model, settings


class AgentState(MessagesState, total=False):
    """
    Agent 的状态（State），基于 LangGraph 预置的 MessagesState 扩展。

    关键点：
    - 这里的 State 本质上是一个 TypedDict；`total=False` 表示字段可以缺省（PEP 589）。
      这非常适合 LangGraph 的“逐节点增量写入 state”的模式：节点只更新它关心的字段。
    - `messages` 来自 MessagesState：对话历史（human/ai/tool/custom 等消息）
    - `safety`：LlamaGuard 的审核结果（用于条件路由）
    - `remaining_steps`：LangGraph 的步数预算（防止 tool 循环跑飞）

    参考：https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    safety: LlamaGuardOutput
    remaining_steps: RemainingSteps


web_search = DuckDuckGoSearchResults(name="WebSearch")
tools = [web_search, calculator]

# 如果配置了 OpenWeatherMap API key，就额外暴露一个天气工具给模型使用。
# 这属于“可选能力”：不影响核心链路，但展示了如何基于 settings 动态装配工具集。
# API key 申请：https://openweathermap.org/api/
if settings.OPENWEATHERMAP_API_KEY:
    wrapper = OpenWeatherMapAPIWrapper(
        openweathermap_api_key=settings.OPENWEATHERMAP_API_KEY.get_secret_value()
    )
    tools.append(OpenWeatherMapQueryRun(name="Weather", api_wrapper=wrapper))

current_date = datetime.now().strftime("%B %d, %Y")
# 注意：这是系统提示词（会发给 LLM），修改会改变模型行为；此处只加解释不改文本。
instructions = f"""
    You are a helpful research assistant with the ability to search the web and use other tools.
    Today's date is {current_date}.

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    A few things to remember:
    - Please include markdown-formatted links to any citations used in your response. Only include one
    or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
    - Use calculator tool with numexpr to answer math questions. The user does not understand numexpr,
      so for the final response, use human readable format - e.g. "300 * 200", not "(300 \\times 200)".
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    """
    把“裸模型”包装成“带系统提示词 + 可调用工具”的可执行 Runnable。

    在端到端链路中的位置：
    - 图中的 `model` 节点会调用这个 runnable，产出 AIMessage（可能包含 tool_calls）。

    设计意图：
    - 将“state -> prompt/messages”的预处理（StateModifier）抽成独立 runnable，便于复用与测试。
    - 通过 `model.bind_tools(tools)` 让模型具备工具调用能力，后续由 ToolNode 执行具体工具。
    """
    bound_model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | bound_model  # type: ignore[return-value]


def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    """
    将安全拦截结果转成“用户可见”的 AIMessage。

    说明：
    - 这里选择直接告诉用户“命中了哪些类别”，属于产品取舍；真实生产环境可能需要更温和/更合规的提示。
    """
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    图节点：调用模型生成下一条 AI 消息（可能触发工具调用）。

    在端到端链路中的位置：
    - 通常在 `guard_input` 通过安全检查后执行；
    - 也会在 ToolNode 执行完工具后再次执行，用于“看工具结果 -> 继续推理/总结”。

    输入/输出的业务意义：
    - 输入：当前 state（包含 messages/tool 输出等）+ config（包含 thread_id/user_id/model 等）
    - 输出：对 state 的增量更新，主要是追加一条 AIMessage 到 messages
    """
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # 关键设计：在把模型输出加入对话之前，先做一次输出审核，避免“不安全内容”被返回给用户。
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
    if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
        return {"messages": [format_safety_message(safety_output)], "safety": safety_output}

    # RemainingSteps 是 LangGraph 的“步数预算”机制。
    # 当步数不够且模型还想调用工具时，直接返回一条提示，避免图陷入“工具调用 -> 没步数执行”的异常状态。
    if state["remaining_steps"] < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
    # 注意：这里必须返回 list[AIMessage]，LangGraph 会把它追加到已有 messages 列表中。
    return {"messages": [response]}


async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    图节点：对“用户输入”做安全审核（输入侧防护）。

    设计意图：
    - 与输出审核（acall_model 内）配合，形成输入/输出双向防线。
    - 本节点不产出 messages，只写入 safety 字段，交给后续条件边做路由决策。
    """
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output, "messages": []}


async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    图节点：当输入被判定不安全时，直接返回拦截提示并结束图执行。
    """
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


# 条件边函数：根据输入审核结果决定走“拦截”还是“继续调用模型”。
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

# 一旦拦截，直接结束（不再调用模型与工具）。
agent.add_edge("block_unsafe_content", END)

# 工具执行完后，回到模型节点，让模型“看工具结果并继续推理/总结”。
agent.add_edge("tools", "model")


# 条件边函数：模型若产生 tool_calls，则进入 ToolNode；否则结束。
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})


# 对外暴露的 LangGraph 图对象（被 `src/agents/agents.py` 注册）。
research_assistant = agent.compile()
