"""
本文件在整个项目中的角色：Service 启动入口（本地开发/直接运行用）。

它解决的核心问题是什么？
- 通过 `uvicorn` 启动 FastAPI 服务（`service:app`），让 `src/client` / `src/streamlit_app.py` 可以访问：
  - `/info`、`/{agent_id}/invoke`、`/{agent_id}/stream`、`/feedback`、`/history` 等端点
- 在启动前加载 `.env`，并根据 `core.settings` 配置日志级别与运行参数。

典型用法：
- 本地开发时直接 `python src/run_service.py`
- 或者用 `uvicorn service:app`（此文件额外做了 Windows 事件循环兼容处理）
"""

import asyncio
import logging
import sys

import uvicorn
from dotenv import load_dotenv

from core import settings

load_dotenv()

if __name__ == "__main__":
    root_logger = logging.getLogger()
    if root_logger.handlers:
        # 当某些环境（例如 notebooks/某些 IDE）提前配置了 root logger 时，
        # `logging.basicConfig(...)` 会被忽略，这里提示一下避免“为什么日志级别不生效”的困惑。
        print(
            f"Warning: Root logger already has {len(root_logger.handlers)} handler(s) configured. "
            f"basicConfig() will be ignored. Current level: {logging.getLevelName(root_logger.level)}"
        )

    logging.basicConfig(level=settings.LOG_LEVEL.to_logging_level())
    # Windows 兼容性：设置更适配数据库驱动的事件循环策略。
    # 背景：
    # - Windows 默认的 ProactorEventLoop 在某些 async 数据库驱动（例如 psycopg）上可能出现兼容问题，
    #   常见表现是数据库连接相关的 "RuntimeError: Event loop is closed"。
    # - WindowsSelectorEventLoopPolicy 通常更稳健，因此在启动服务前设置。
    #
    # 参考：https://www.psycopg.org/psycopg3/docs/advanced/async.html#asynchronous-operations
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run(
        "service:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.is_dev(),
        timeout_graceful_shutdown=settings.GRACEFUL_SHUTDOWN_TIMEOUT,
    )
