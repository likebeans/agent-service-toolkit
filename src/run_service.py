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
        print(
            f"Warning: Root logger already has {len(root_logger.handlers)} handler(s) configured. "
            f"basicConfig() will be ignored. Current level: {logging.getLevelName(root_logger.level)}"
        )

    logging.basicConfig(level=settings.LOG_LEVEL.to_logging_level())
    # 设置 Windows 系统的兼容事件循环策略
    # 在 Windows 系统上，默认的 ProactorEventLoop 可能会导致某些异步数据库驱动程序
    # （如 psycopg PostgreSQL 驱动程序）出现问题。
    # WindowsSelectorEventLoopPolicy 提供更好的兼容性，可防止在使用数据库连接时出现
    # "RuntimeError: Event loop is closed" 错误。
    # 这需要在运行应用程序服务器之前设置。
    # 更多信息请参考文档。
    # https://www.psycopg.org/psycopg3/docs/advanced/async.html#asynchronous-operations
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run(
        "service:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.is_dev(),
        timeout_graceful_shutdown=settings.GRACEFUL_SHUTDOWN_TIMEOUT,
    )
