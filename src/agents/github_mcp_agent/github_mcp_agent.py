"""
本文件在整个项目中的角色：演示如何把 MCP（Model Context Protocol）工具服务器接入到 Agent，并以“异步加载”方式在启动期完成初始化。

为什么这个文件存在？
- MCP 的核心价值：把“工具能力”从应用进程中解耦出来，Agent 通过统一协议动态发现/调用工具。
- 以 GitHub 为例：仓库管理、Issue/PR 操作、文件读写、提交历史等都可以通过 MCP server 统一暴露为 tools。
- 这类工具集通常需要：
  - 启动期建连接（HTTP/SSE/WebSocket 等）
  - 运行期动态拉取工具列表（取决于 token 权限/服务端能力）
  因此非常适合用 `LazyLoadingAgent` 在 FastAPI lifespan 阶段完成预热。

端到端链路（简化）：
1) 服务启动：`src/service/service.py:lifespan()` 调用 `agents.load_agent("github-mcp-agent")`
2) `GitHubMCPAgent.load()`：
   - 如果未配置 `GITHUB_PAT`：降级为“无工具”的 agent
   - 否则初始化 MCP client，拉取 tools 列表
   - 创建 LangGraph/Agent（create_agent）并写入 `self._graph`
3) 请求进入：服务端 `get_agent("github-mcp-agent")` 返回已加载的图

注意（不改变行为的约束）：
- `prompt` 是 system_prompt，会直接发给 LLM；翻译/改写会改变行为，因此只加注释不改内容。
"""

import logging
from datetime import datetime

from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StreamableHttpConnection
from langgraph.graph.state import CompiledStateGraph

from agents.lazy_agent import LazyLoadingAgent
from core import get_model, settings

logger = logging.getLogger(__name__)

current_date = datetime.now().strftime("%B %d, %Y")
# 注意：这是 system prompt，会直接发给 LLM；不要随意改写/翻译（会改变行为）。
prompt = f"""
You are GitHubBot, a specialized assistant for GitHub repository management and development workflows.
You have access to GitHub MCP tools that allow you to interact with GitHub repositories, issues, pull requests,
and other GitHub resources. Today's date is {current_date}.

Your capabilities include:
- Repository management (create, clone, browse)
- Issue management (create, list, update, close)
- Pull request management (create, review, merge)
- Branch management (create, switch, merge)
- File operations (read, write, search)
- Commit operations (create, view history)

Guidelines:
- Always be helpful and provide clear explanations of GitHub operations
- When creating or modifying content, ensure it follows best practices
- Be cautious with destructive operations (deletes, force pushes, etc.)
- Provide context about what you're doing and why
- Use appropriate commit messages and PR descriptions
- Respect repository permissions and access controls

NOTE: You have access to GitHub MCP tools that provide direct GitHub API access.
"""


class GitHubMCPAgent(LazyLoadingAgent):
    """
    GitHub MCP Agent：通过 MCP 工具实现 GitHub 工作流自动化的 Agent。

    为什么继承 LazyLoadingAgent？
    - MCP 工具需要在启动期异步拉取（get_tools），不适合在 import 时执行 IO。
    - 通过 `load()` 将 IO 与建图集中到服务启动阶段，提升请求期稳定性与延迟可控性。

    成员字段：
    - `_mcp_tools`：从 MCP server 动态获取到的工具列表（BaseTool）
    - `_mcp_client`：MCP 客户端（支持多 server；本例只配置 github）
    """

    def __init__(self) -> None:
        super().__init__()
        self._mcp_tools: list[BaseTool] = []
        self._mcp_client: MultiServerMCPClient | None = None

    async def load(self) -> None:
        """
        异步加载：初始化 MCP client 并拉取 tools，然后创建可执行图。

        设计取舍：
        - 没有 `GITHUB_PAT` 时不报错，而是降级为“无 tools”。
          这更符合模板工程的易用性：不开启 GitHub 能力也能跑通服务。
        - 初始化失败（网络/权限/协议）时同样降级为空 tools，并记录日志。
        """
        if not settings.GITHUB_PAT:
            logger.info("GITHUB_PAT is not set, GitHub MCP agent will have no tools")
            self._mcp_tools = []
            self._graph = self._create_graph()
            self._loaded = True
            return

        try:
            # 初始化 MCP client（这里使用 Streamable HTTP 连接，带 Bearer token）
            github_pat = settings.GITHUB_PAT.get_secret_value()
            connections = {
                "github": StreamableHttpConnection(
                    transport="streamable_http",
                    url=settings.MCP_GITHUB_SERVER_URL,
                    headers={
                        "Authorization": f"Bearer {github_pat}",
                    },
                )
            }

            self._mcp_client = MultiServerMCPClient(connections)
            logger.info("MCP client initialized successfully")

            # 从 MCP server 动态拉取 tools 列表（不同 token 权限可能返回不同工具集）
            self._mcp_tools = await self._mcp_client.get_tools()
            logger.info(f"GitHub MCP agent initialized with {len(self._mcp_tools)} tools")

        except Exception as e:
            logger.error(f"Failed to initialize GitHub MCP agent: {e}")
            self._mcp_tools = []
            self._mcp_client = None

        # 无论是否成功拉到 tools，都创建图：tool 列表为空时相当于“只能对话、不能调用 GitHub API”。
        self._graph = self._create_graph()
        self._loaded = True

    def _create_graph(self) -> CompiledStateGraph:
        """
        创建 GitHub MCP Agent 的图（LangChain create_agent 风格）。

        注意：
        - 这里使用的 `create_agent` 产物会被当成 LangGraph 图对待（CompiledStateGraph）。
        - tools 列表来自 `_mcp_tools`（可能为空），这决定了模型能否做 GitHub 操作。
        """
        model = get_model(settings.DEFAULT_MODEL)

        return create_agent(
            model=model,
            tools=self._mcp_tools,
            name="github-mcp-agent",
            system_prompt=prompt,
        )


# 创建 agent 实例
github_mcp_agent = GitHubMCPAgent()
