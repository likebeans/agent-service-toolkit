"""
本文件在整个项目中的角色：直接运行 Agent Graph 的最小示例（不经过 HTTP Service）。

它解决的核心问题是什么？
- 当你在学习/调试 Agent 构图时，有时不想启动 FastAPI + Client + Streamlit；
  这个脚本演示了如何在 Python 进程内直接：
  1) 获取默认 Agent（`agents.DEFAULT_AGENT`）
  2) 构造 LangGraph 的输入（MessagesState）
  3) 调用 `agent.ainvoke(...)` 得到最终 state，并打印最后一条消息

典型用法：
- `python src/run_agent.py`

与端到端链路的关系：
- 这里跳过了 `src/service/service.py` 的 HTTP 层，但底层调用的仍是同一个 AgentGraph；
  因此适合用来验证“Agent 本身是否工作正常”。
"""

import asyncio
from typing import cast
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph

load_dotenv()

from agents import DEFAULT_AGENT, get_agent  # noqa: E402

# 默认 Agent 使用 `StateGraph.compile()` 构建，因此返回类型是 `CompiledStateGraph`（可 ainvoke/astream）。
agent = cast(CompiledStateGraph, get_agent(DEFAULT_AGENT))


async def main() -> None:
    """最小运行示例：给 Agent 一个 HumanMessage，然后打印最后一条输出消息。"""
    inputs: MessagesState = {
        "messages": [HumanMessage("Find me a recipe for chocolate chip cookies")]
    }
    result = await agent.ainvoke(
        input=inputs,
        config=RunnableConfig(configurable={"thread_id": uuid4()}),
    )
    result["messages"][-1].pretty_print()

    # 可选：把 Agent Graph 绘制成 PNG（用于学习图结构）。
    # 依赖（示例以 macOS 为主）：
    # - brew install graphviz
    # - export CFLAGS="-I $(brew --prefix graphviz)/include"
    # - export LDFLAGS="-L $(brew --prefix graphviz)/lib"
    # - pip install pygraphviz
    #
    # 然后取消注释：
    # agent.get_graph().draw_png("agent_diagram.png")


if __name__ == "__main__":
    asyncio.run(main())
