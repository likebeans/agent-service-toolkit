"""
本文件在整个项目中的角色：Agent 注册表（Registry）与访问入口。

为什么这个文件存在？
- FastAPI 服务需要把 `/{agent_id}/invoke`、`/{agent_id}/stream` 这类请求路由到对应 Agent。
- 项目支持“多种 LangGraph 构建风格”（`@entrypoint` 返回 Pregel；`StateGraph.compile()` 返回 CompiledStateGraph）。
  服务层不希望关心这些差异，因此这里做了类型与访问的统一封装。
- 项目还支持“需要异步初始化”的 Agent（典型：MCP 工具需要在启动期拉取/建连接）。
  因此需要一个统一的 `load_agent()` 入口，让服务在启动阶段完成预热。

它解决的核心问题是什么？
- 多 Agent 的注册、元信息（用于 `/info`）、按 key 获取、以及 Lazy Loading 的启动期加载。

典型调用者是谁？
- `src/service/service.py`：
  - `lifespan()`：启动期遍历所有 Agent 并 `await load_agent(...)`，然后注入 memory（checkpointer/store）。
  - `invoke()` / `stream()`：请求期用 `get_agent(agent_id)` 得到可执行的 AgentGraph。

被谁依赖？
- `src/agents/__init__.py` 将本模块的入口函数 re-export，作为 `agents` 包的公共 API。
"""

from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel

from agents.bg_task_agent.bg_task_agent import bg_task_agent
from agents.chatbot import chatbot
from agents.command_agent import command_agent
from agents.github_mcp_agent.github_mcp_agent import github_mcp_agent
from agents.interrupt_agent import interrupt_agent
from agents.knowledge_base_agent import kb_agent
from agents.langgraph_supervisor_agent import langgraph_supervisor_agent
from agents.langgraph_supervisor_hierarchy_agent import langgraph_supervisor_hierarchy_agent
from agents.lazy_agent import LazyLoadingAgent
from agents.rag_assistant import rag_assistant
from agents.research_assistant import research_assistant
from schema import AgentInfo

DEFAULT_AGENT = "research-assistant"

# 统一不同 LangGraph 构建模式的返回类型：
# - `@entrypoint` 装饰的函数会返回 `Pregel`
# - `StateGraph(...).compile()` 会返回 `CompiledStateGraph`
# 服务层只需要“能 ainvoke/astream 的图”，不需要区分是哪一种，因此做类型别名抽象。
AgentGraph = CompiledStateGraph | Pregel  # `get_agent()` 的返回类型（确保已加载可执行）
AgentGraphLike = (
    CompiledStateGraph | Pregel | LazyLoadingAgent
)  # 注册表允许存放的类型（包含 LazyLoadingAgent）


@dataclass
class Agent:
    """
    Agent 的“注册表条目（entry）”。

    设计意图：
    - 将“对外可见的描述信息”（description，用于 `/info` 与 UI 展示）
      与“可执行的图对象”（graph_like）绑定在一起，便于统一管理。
    - graph_like 允许是 LazyLoadingAgent：即图对象可能需要异步初始化后才可用。
    """

    description: str
    graph_like: AgentGraphLike


agents: dict[str, Agent] = {
    "chatbot": Agent(description="A simple chatbot.", graph_like=chatbot),
    "research-assistant": Agent(
        description="A research assistant with web search and calculator.",
        graph_like=research_assistant,
    ),
    "rag-assistant": Agent(
        description="A RAG assistant with access to information in a database.",
        graph_like=rag_assistant,
    ),
    "command-agent": Agent(description="A command agent.", graph_like=command_agent),
    "bg-task-agent": Agent(description="A background task agent.", graph_like=bg_task_agent),
    "langgraph-supervisor-agent": Agent(
        description="A langgraph supervisor agent", graph_like=langgraph_supervisor_agent
    ),
    "langgraph-supervisor-hierarchy-agent": Agent(
        description="A langgraph supervisor agent with a nested hierarchy of agents",
        graph_like=langgraph_supervisor_hierarchy_agent,
    ),
    "interrupt-agent": Agent(
        description="An agent the uses interrupts.", graph_like=interrupt_agent
    ),
    "knowledge-base-agent": Agent(
        description="A retrieval-augmented generation agent using Amazon Bedrock Knowledge Base",
        graph_like=kb_agent,
    ),
    "github-mcp-agent": Agent(
        description="A GitHub agent with MCP tools for repository management and development workflows.",
        graph_like=github_mcp_agent,
    ),
}


async def load_agent(agent_id: str) -> None:
    """
    启动期加载（预热）某个 Agent。

    在端到端链路中的位置：
    - 属于“服务启动链路”的一部分，通常由 `src/service/service.py:lifespan()` 调用。

    业务意义：
    - 对于 LazyLoadingAgent：在启动阶段完成外部依赖初始化（例如 MCP client 建连、动态拉取 tools），
      并构建出可执行的 LangGraph 图，避免请求期再做昂贵/不稳定的初始化。

    注意：
    - 普通（非 Lazy）Agent 这里是 no-op。
    - agent_id 不存在会抛 KeyError；服务层可决定是否兜底或让启动失败。
    """
    graph_like = agents[agent_id].graph_like
    if isinstance(graph_like, LazyLoadingAgent):
        await graph_like.load()


def get_agent(agent_id: str) -> AgentGraph:
    """
    获取一个“可执行”的 Agent 图（AgentGraph）。

    在端到端链路中的位置：
    - 属于“请求处理链路”的入口步骤：`invoke()` / `stream()` 在调用图前都会先拿到 AgentGraph。

    设计意图：
    - 屏蔽 LangGraph 两种图类型（Pregel/CompiledStateGraph）的差异。
    - 对 LazyLoadingAgent：强制要求其已在启动期 load 完成，否则在请求期直接报错。
      这样可以把“初始化失败”提前暴露在启动阶段，而不是在用户请求时随机失败。

    注意：
    - 本函数不负责调用 `load()`（也不应该在请求期隐式做 IO），因此 LazyLoadingAgent 未加载会抛 RuntimeError。
      正确用法是在 `lifespan()` 中调用 `load_agent()`。
    """
    agent_graph = agents[agent_id].graph_like

    # LazyLoadingAgent 的 graph 在 load() 之前并不存在；这里用内部标记确保生命周期正确。
    if isinstance(agent_graph, LazyLoadingAgent):
        if not agent_graph._loaded:
            raise RuntimeError(f"Agent {agent_id} not loaded. Call load() first.")
        return agent_graph.get_graph()

    # 非 Lazy Agent：注册表里直接就是可执行的图对象。
    return agent_graph


def get_all_agent_info() -> list[AgentInfo]:
    """
    返回所有可用 Agent 的元信息。

    在端到端链路中的位置：
    - 由服务端 `/info` 端点调用（`src/service/service.py:info()`），用于：
      - UI 下拉选择 Agent
      - 客户端做 agent 校验（`AgentClient.update_agent(..., verify=True)`）
    """
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]
