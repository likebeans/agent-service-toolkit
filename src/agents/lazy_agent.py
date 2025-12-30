"""
本文件在整个项目中的角色：为“需要异步初始化”的 Agent 提供统一基类与生命周期约束。

为什么这个文件存在？
- 不是所有 Agent 都能在 import 时就构建出可用的图（Graph）。
  典型例子：MCP Agent 需要在启动时建立到 MCP Server 的连接、动态拉取工具列表，然后才能创建图。
- FastAPI 服务启动时（`src/service/service.py:lifespan()`）会主动 `await load_agent(...)`，
  以保证请求进来时图已经准备好，不在请求路径里做 IO/初始化。

它解决的核心问题是什么？
- 将“异步预热/加载（load）”与“请求期读取图（get_graph）”分离，并通过 `_loaded` 标记强制生命周期正确：
  - load(): 做一次性 async 初始化，并产出 `_graph`
  - get_graph(): 请求期只读拿到图；如果没 load 就报错

典型调用者是谁？
- 子类 Agent：实现 `load()` 并在其中设置 `self._graph`
- `src/agents/agents.py:load_agent()`：在启动期触发 `await agent.load()`
- `src/agents/agents.py:get_agent()`：在请求期调用 `get_graph()` 取出可执行图
"""

from abc import ABC, abstractmethod

from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel


class LazyLoadingAgent(ABC):
    """
    需要异步加载的 Agent 基类。

    设计意图（“如果你自己实现会怎么做”）：
    - 把“加载外部依赖/构建图”的副作用集中在 `load()`，让请求路径保持纯粹与可控（避免首请求抖动）。
    - 用 `_loaded` 明确标记状态，防止忘记 load 就被调用导致隐式错误。
    - 用 `_graph` 统一承载最终可执行的 LangGraph 图对象（CompiledStateGraph 或 Pregel）。
    """

    def __init__(self) -> None:
        """初始化 LazyLoadingAgent 的内部状态（此时尚未创建图）。"""
        self._loaded = False
        self._graph: CompiledStateGraph | Pregel | None = None

    @abstractmethod
    async def load(self) -> None:
        """
        执行该 Agent 的异步加载（一次性预热）。

        在端到端链路中的位置：
        - 服务启动阶段：`src/service/service.py:lifespan()` 会遍历所有 agent 并调用 `load_agent()`，
          进而触发这里的 `load()`。

        你通常需要在这里做的事（按优先级）：
        - 建立外部连接（例如 MCP clients、数据库连接、SDK 初始化等）
        - 动态拉取/构建 tools 或其他资源
        - 组装 LangGraph 图，并写入 `self._graph`
        - 最后将 `self._loaded = True`

        注意：
        - 这里应该尽量做到“失败可观测”：如果加载失败，记录日志并决定是否降级（例如：tools 为空）。
        - 不建议在 `get_graph()` 或请求路径里做 IO。
        """
        raise NotImplementedError  # pragma: no cover

    def get_graph(self) -> CompiledStateGraph | Pregel:
        """
        获取该 Agent 的可执行图（Graph）。

        设计意图：
        - get_graph() 是“请求期”调用的轻量方法：只做状态检查与返回，不做任何初始化。
        - 这样服务层 `get_agent()` 可以在每次请求里安全调用。

        返回值：
        - load() 阶段创建的图实例（CompiledStateGraph 或 Pregel）。
        """
        if not self._loaded:
            raise RuntimeError("Agent not loaded. Call load() first.")
        if self._graph is None:
            raise RuntimeError("Agent graph not created during load().")
        return self._graph
