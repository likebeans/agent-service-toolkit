"""
本文件在整个项目中的角色：Python Client（`AgentClient`）使用示例（同步 + 异步）。

它解决的核心问题是什么？
- 演示客户端如何与 FastAPI Service 交互（见 `src/service/service.py`）：
  - `invoke/ainvoke`：一次性调用，只返回最终 `schema.ChatMessage`
  - `stream/astream`：SSE 流式调用，会产出 token（str）与中间/最终消息（ChatMessage）

典型用法：
- 先启动服务：`python src/run_service.py`
- 再运行本脚本：`python src/run_client.py`

与端到端链路的关系：
- 这个脚本等价于“最小客户端”，可用于验证：
  - `/info` 能否访问
  - `/invoke`、`/stream` 的协议是否正常（包括 token streaming）
"""

import asyncio

from client import AgentClient
from core import settings
from schema import ChatMessage


async def amain() -> None:
    """异步模式示例（适合异步应用/高并发场景）。"""
    #### 异步 ####
    client = AgentClient(settings.BASE_URL)

    print("Agent info:")
    print(client.info)

    print("Chat example:")
    response = await client.ainvoke("Tell me a brief joke?", model="gpt-5-nano")
    response.pretty_print()

    print("\nStream example:")
    async for message in client.astream("Share a quick fun fact?"):
        if isinstance(message, str):
            print(message, flush=True, end="")
        elif isinstance(message, ChatMessage):
            print("\n", flush=True)
            message.pretty_print()
        else:
            print(f"ERROR: Unknown type - {type(message)}")


def main() -> None:
    """同步模式示例（适合脚本/简单调用）。"""
    #### 同步 ####
    client = AgentClient(settings.BASE_URL)

    print("Agent info:")
    print(client.info)

    print("Chat example:")
    response = client.invoke("Tell me a brief joke?", model="gpt-5-nano")
    response.pretty_print()

    print("\nStream example:")
    for message in client.stream("Share a quick fun fact?"):
        if isinstance(message, str):
            print(message, flush=True, end="")
        elif isinstance(message, ChatMessage):
            print("\n", flush=True)
            message.pretty_print()
        else:
            print(f"ERROR: Unknown type - {type(message)}")


if __name__ == "__main__":
    print("Running in sync mode")
    main()
    print("\n\n\n\n\n")
    print("Running in async mode")
    asyncio.run(amain())
