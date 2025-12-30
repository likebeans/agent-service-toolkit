"""
`agents` 包的公共导出（Public API）。

设计意图：
- 让其它模块（尤其是服务层 `src/service/service.py`）只依赖 `agents` 这个包的稳定入口，
  而不是直接从内部文件（如 `agents.agents`）导入实现细节。
- 这也是一种“模块边界”实践：对外只暴露必要的函数/类型，内部结构可在不影响调用方的情况下演进。
"""

from agents.agents import (
    DEFAULT_AGENT,
    AgentGraph,
    AgentGraphLike,
    get_agent,
    get_all_agent_info,
    load_agent,
)

__all__ = [
    "get_agent",
    "load_agent",
    "get_all_agent_info",
    "DEFAULT_AGENT",
    "AgentGraph",
    "AgentGraphLike",
]
