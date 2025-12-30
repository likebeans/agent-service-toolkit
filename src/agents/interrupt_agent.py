"""
本文件在整个项目中的角色：演示 LangGraph 的 Human-in-the-Loop（`interrupt()`）能力 + 长期记忆（Store）的使用方式。

为什么这个文件存在？
- 真实的 Agent 常常需要在执行过程中“停下来向用户追问/确认”，例如：
  - 缺少关键参数（生日、地址、权限确认等）
  - 需要人工审批（高风险操作、付费操作、删除数据等）
- LangGraph 提供 `langgraph.types.interrupt()`：
  - 在图执行中抛出一个可恢复的中断（Interrupt）
  - 服务端下一次收到用户输入后，可通过 `Command(resume=...)` 继续执行
  - 本仓库的服务端实现见：`src/service/service.py:_handle_input()`（会检查是否存在 pending interrupt）

本 Agent 额外演示了“长期记忆 Store”的典型用法：
- 用 `user_id` 作为 namespace，把用户生日写入 store，实现跨会话/跨 thread 复用信息
- 服务端会在启动期把 store 注入到 agent 上（`src/service/service.py:lifespan()` 里 `agent.store = store`）

注意（不改变行为的约束）：
- 本文件里的多个 PromptTemplate（background_prompt/birthdate_extraction_prompt/response_prompt）会直接发给 LLM。
  翻译/改写会改变模型输出与中断触发条件，因此我们只添加中文解释，不改动这些字符串内容。
- `BirthdateExtraction` 的 Field.description 可能会影响结构化输出（LLM 生成 JSON 的提示），因此也不翻译。
"""

import logging
from datetime import datetime
from typing import Any

from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from core import get_model, settings

# 日志用于观察：什么时候触发中断、是否命中 store、是否成功写回等（学习与排障都很有用）
logger = logging.getLogger(__name__)


class AgentState(MessagesState, total=False):
    """
    interrupt Agent 的状态（State）。

    字段说明：
    - messages：对话历史（MessagesState 提供）
    - birthdate：从对话中抽取/从 store 读取的生日（示例里用 datetime）

    `total=False`：TypedDict 字段允许缺省，适配 LangGraph “节点按需写入 state”。
    参考：https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    birthdate: datetime | None


def wrap_model(
    model: BaseChatModel | Runnable[LanguageModelInput, Any], system_prompt: BaseMessage
) -> RunnableSerializable[AgentState, Any]:
    """
    将模型包装为 runnable：把系统提示词 + 对话历史拼接后再调用模型。

    为什么要抽这个函数？
    - 多个节点都需要“带不同 system prompt 的模型调用”，抽出来能减少重复并突出“节点之间的差异”。
    """
    preprocessor = RunnableLambda(
        lambda state: [system_prompt] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


# 注意：这是会发送给 LLM 的 prompt，不要随意改写/翻译（会改变行为）。
background_prompt = SystemMessagePromptTemplate.from_template("""
You are a helpful assistant that tells users there zodiac sign.
Provide a one sentence summary of the origin of zodiac signs.
Don't tell the user what their sign is, you are just demonstrating your knowledge on the topic.
""")


async def background(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    图节点：在触发 interrupt 之前先做一段“背景输出”。

    设计意图：
    - 演示：interrupt 并不一定发生在一开始；图可以先做一些工作（例如解释背景、收集上下文），
      然后在缺信息时再中断追问。
    """

    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m, background_prompt.format())
    response = await model_runnable.ainvoke(state, config)

    return {"messages": [AIMessage(content=response.content)]}


# 注意：这是会发送给 LLM 的 prompt，不要随意改写/翻译（会改变行为）。
birthdate_extraction_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert at extracting birthdates from conversational text.

Rules for extraction:
- Look for user messages that mention birthdates
- Consider various date formats (MM/DD/YYYY, YYYY-MM-DD, Month Day, Year)
- Validate that the date is reasonable (not in the future)
- If no clear birthdate was provided by the user, return None
""")


class BirthdateExtraction(BaseModel):
    """
    用于结构化输出的 schema：让模型输出 {birthdate, reasoning}。

    重要提示：
    - `Field.description` 往往会被 LangChain 用来生成“结构化输出提示”，影响模型产出的 JSON 质量与字段含义。
      因此这里不翻译 description，避免改变模型行为。
    """

    birthdate: str | None = Field(
        description="The extracted birthdate in YYYY-MM-DD format. If no birthdate is found, this should be None."
    )
    reasoning: str = Field(
        description="Explanation of how the birthdate was extracted or why no birthdate was found"
    )


async def determine_birthdate(
    state: AgentState, config: RunnableConfig, store: BaseStore
) -> AgentState:
    """
    图节点：确定用户生日（优先读 store，其次用 LLM 从对话中抽取；缺失则 interrupt 追问）。

    在端到端链路中的位置：
    - `background` 之后执行，是本 Agent 的核心“控制流节点”。

    输入/输出的业务意义：
    - 输入：对话历史（可能包含用户提到的生日），以及 config 里的 user_id
    - 输出：把 birthdate 写入 state；必要时触发 interrupt 让用户补充

    关键设计点：
    - 先读 store：避免重复询问同一用户（跨 thread/跨会话的长期记忆）
    - LLM 抽取使用 `with_structured_output(BirthdateExtraction)`：让输出更稳定可解析
    - 当缺生日时使用 `interrupt()`：暂停图，等待用户回复后继续（递归重试）
    """

    # 1) 从 config 获取 user_id：用于“按用户”区分长期记忆命名空间（namespace）。
    user_id = config["configurable"].get("user_id")
    logger.info(f"[determine_birthdate] Extracted user_id: {user_id}")
    namespace = None
    key = "birthdate"
    birthdate = None  # 初始化 birthdate 变量（后续可能来自 store 或 LLM 抽取）

    if user_id:
        # 用 user_id 作为 namespace，确保不同用户的键空间隔离。
        namespace = (user_id,)

        # 2) 优先从 store 读取：如果已存在就直接返回，避免再走抽取与中断流程。
        try:
            result = await store.aget(namespace, key=key)
            # 兼容 store.aget 可能返回 Item 或 list[Item] 的情况（不同 store 实现细节可能不同）。
            user_data = None
            if result:  # 检查 store 是否返回了内容
                if isinstance(result, list):
                    if result:  # 列表非空时取第一个元素
                        user_data = result[0]
                else:  # 否则认为 store 直接返回了 Item 对象
                    user_data = result

            if user_data and user_data.value.get("birthdate"):
                # store 里存的是 ISO 字符串，这里转回 datetime 以便后续 prompt 展示与逻辑处理。
                birthdate_str = user_data.value["birthdate"]
                birthdate = datetime.fromisoformat(birthdate_str) if birthdate_str else None
                # 已命中长期记忆：直接返回，不再触发 interrupt。
                logger.info(
                    f"[determine_birthdate] Found birthdate in store for user {user_id}: {birthdate}"
                )
                return {
                    "birthdate": birthdate,
                    "messages": [],
                }
        except Exception as e:
            # store 不可用/读失败时，不应阻断主流程；继续走 LLM 抽取逻辑（降级策略）。
            logger.error(f"Error reading from store for namespace {namespace}, key {key}: {e}")
            pass
    else:
        # 没有 user_id 时无法按用户隔离做长期记忆（跨会话复用会变得不可靠）；这里选择仅记录日志并继续。
        logger.warning(
            "Warning: user_id not found in config. Skipping persistent birthdate storage/retrieval for this run."
        )

    # 3) 未命中 store：用 LLM 从对话里抽取生日（结构化输出）。
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(
        m.with_structured_output(BirthdateExtraction), birthdate_extraction_prompt.format()
    ).with_config(tags=["skip_stream"])
    response: BirthdateExtraction = await model_runnable.ainvoke(state, config)

    # 4) 抽取失败：触发 interrupt，让用户补充生日；然后把用户回复追加到 messages，再递归重试。
    if response.birthdate is None:
        birthdate_input = interrupt(f"{response.reasoning}\nPlease tell me your birthdate?")
        state["messages"].append(HumanMessage(birthdate_input))
        return await determine_birthdate(state, config, store)

    # 5) 抽取成功：解析字符串为 datetime；如果格式不合法，再次 interrupt 让用户按要求输入。
    try:
        birthdate = datetime.fromisoformat(response.birthdate)
    except ValueError:
        birthdate_input = interrupt(
            "I couldn't understand the date format. Please provide your birthdate in YYYY-MM-DD format."
        )
        state["messages"].append(HumanMessage(birthdate_input))
        return await determine_birthdate(state, config, store)

    # 6) 写回 store：只有在有 user_id 时才可长期保存（否则无法按用户隔离）。
    if user_id and namespace:
        # 用 ISO 字符串存储，便于 JSON 序列化与跨语言/跨系统兼容。
        birthdate_str = birthdate.isoformat() if birthdate else None
        try:
            await store.aput(namespace, key, {"birthdate": birthdate_str})
        except Exception as e:
            # 写失败不应影响本次对话继续；记录日志即可（降级策略）。
            logger.error(f"Error writing to store for namespace {namespace}, key {key}: {e}")

    # 返回 birthdate（来自 store 或抽取结果）
    logger.info(f"[determine_birthdate] Returning birthdate {birthdate} for user {user_id}")
    return {
        "birthdate": birthdate,
        "messages": [],
    }


# 注意：这是会发送给 LLM 的 prompt，不要随意改写/翻译（会改变行为）。
response_prompt = SystemMessagePromptTemplate.from_template("""
You are a helpful assistant.

Known information:
- The user's birthdate is {birthdate_str}

User's latest message: "{last_user_message}"

Based on the known information and the user's message, provide a helpful and relevant response.
If the user asked for their birthdate, confirm it.
If the user asked for their zodiac sign, calculate it and tell them.
Otherwise, respond conversationally based on their message.
""")


async def generate_response(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    图节点：在已知 birthdate 的前提下生成最终回复。

    在端到端链路中的位置：
    - `determine_birthdate` 之后执行；此时 birthdate 应该已经存在（否则说明上游逻辑异常/被降级）。
    """
    birthdate = state.get("birthdate")
    if state.get("messages") and isinstance(state["messages"][-1], HumanMessage):
        last_user_message = state["messages"][-1].content
    else:
        last_user_message = ""

    if not birthdate:
        # 理论上不应发生：determine_birthdate 会通过 interrupt 确保拿到生日。
        # 这里是一个防御性兜底，避免图因为意外 state 而崩溃。
        return {
            "messages": [
                AIMessage(
                    content="I couldn't determine your birthdate. Could you please provide it?"
                )
            ]
        }

    birthdate_str = birthdate.strftime("%B %d, %Y")  # 用于展示的日期格式

    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(
        m, response_prompt.format(birthdate_str=birthdate_str, last_user_message=last_user_message)
    )
    response = await model_runnable.ainvoke(state, config)

    return {"messages": [AIMessage(content=response.content)]}


# -----------------------------
# 图定义：background -> determine_birthdate (可能 interrupt) -> generate_response -> END
# -----------------------------
agent = StateGraph(AgentState)
agent.add_node("background", background)
agent.add_node("determine_birthdate", determine_birthdate)
agent.add_node("generate_response", generate_response)

agent.set_entry_point("background")
agent.add_edge("background", "determine_birthdate")
agent.add_edge("determine_birthdate", "generate_response")
agent.add_edge("generate_response", END)

interrupt_agent = agent.compile()
interrupt_agent.name = "interrupt-agent"
