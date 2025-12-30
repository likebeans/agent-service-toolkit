"""
本包在整个项目中的角色：对外暴露“Python Client/SDK”的稳定入口。

为什么这个文件存在？
- 让使用方只需要 `from client import AgentClient`，而不必记住实现细节路径 `client.client`。
- 这是典型的“包级 API 门面（facade）”：内部实现可调整，但对外导入路径保持稳定。

典型调用者：
- `src/frontend/*`（例如 Streamlit/CLI）或用户自己的脚本/Notebook。
- 任何希望通过 HTTP 调用 `src/service/service.py` 所提供 FastAPI 接口的地方。
"""

from client.client import AgentClient, AgentClientError

__all__ = ["AgentClient", "AgentClientError"]
