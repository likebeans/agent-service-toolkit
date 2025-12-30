"""
本文件在整个项目中的角色：演示 LangGraph 的 `Command`（显式“更新状态 + 路由跳转”）能力。

为什么这个文件存在？
- 很多同学学习 LangGraph 时会默认只用“条件边（conditional edges）”来做分支路由。
- 但 LangGraph 还提供了 `langgraph.types.Command`：让一个节点函数在返回时同时表达两件事：
  1) 对 state 的更新（update）
  2) 下一步跳转到哪个节点（goto）
- 这个 Agent 是一个“最小可读 demo”，帮助你理解 Command 的语义与适用场景。

典型调用者是谁？
- `src/agents/agents.py`：注册为 `"command-agent"`，用于在服务/UI 中体验分支路由效果。

注意：
- 这是教学示例，不强调“业务价值”，强调“框架能力”。
"""

import random
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.types import Command


class AgentState(MessagesState, total=False):
    """
    Command 示例 Agent 的状态（State）。

    这里不额外扩展字段，仅复用 MessagesState 的 messages。
    `total=False`：TypedDict 字段可缺省，适配 LangGraph 的增量更新。

    参考：https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """


# -----------------------------
# 定义节点函数（nodes）
# -----------------------------


def node_a(state: AgentState) -> Command[Literal["node_b", "node_c"]]:
    print("Called A")
    value = random.choice(["a", "b"])
    goto: Literal["node_b", "node_c"]
    # 这里用普通 if/else 决定路由，等价于“条件边函数”的作用。
    if value == "a":
        goto = "node_b"
    else:
        goto = "node_c"

    # 关键点：Command 允许你“同时”做两件事：
    # - update：写入状态（例如追加一条消息）
    # - goto：指定下一个节点（代替 add_edge/add_conditional_edges）
    return Command(
        # 状态更新：这里写入一条 AIMessage
        update={"messages": [AIMessage(content=f"Hello {value}")]},
        # 路由跳转：决定下一步执行哪个节点
        goto=goto,
    )


def node_b(state: AgentState):
    print("Called B")
    return {"messages": [AIMessage(content="Hello B")]}


def node_c(state: AgentState):
    print("Called C")
    return {"messages": [AIMessage(content="Hello C")]}


# -----------------------------
# 建图：注意这里只有 START -> node_a 的显式边
# 后续 node_a -> node_b/node_c 的跳转由 Command.goto 决定
# -----------------------------
builder = StateGraph(AgentState)
builder.add_edge(START, "node_a")
builder.add_node(node_a)
builder.add_node(node_b)
builder.add_node(node_c)
# 注意：A/B/C 之间没有显式 add_edge！
# - 这正是本示例要展示的点：路由关系写在 Command.goto 里，而不是图的边里。

command_agent = builder.compile()
