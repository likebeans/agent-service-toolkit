"""
本包在整个项目中的角色：Memory（记忆/持久化）组件的统一初始化入口。

它解决的核心问题是什么？
- LangGraph 在运行时通常需要两类“状态承载”能力：
  1) Checkpointer（短期记忆 / 对话线程状态）：把每个 thread 的 graph state（messages 等）持久化下来，
     以便多轮对话、服务重启后恢复。
  2) Store（长期记忆 / 跨线程知识）：用于跨对话的检索、用户画像、长期知识积累（视具体 Agent 设计而定）。
- 不同部署环境会选择不同后端（SQLite/Postgres/Mongo），上层不应关心具体实现差异。
  因此这里提供“按配置选择实现”的工厂函数：`initialize_database()` 与 `initialize_store()`。

典型调用者：
- `src/service/service.py:lifespan()`：
  - `async with initialize_database() as saver, initialize_store() as store:` 在服务启动期完成初始化；
  - 然后把 `saver` 注入到每个 Agent 的 `checkpointer`，把 `store` 注入到每个 Agent 的 `store`。

为什么用 Context Manager？
- 连接池/数据库连接需要在启动期建立、在关闭时释放；
- 统一用 async context manager 能把“资源生命周期”收敛到一个可靠的入口。
"""

from contextlib import AbstractAsyncContextManager

from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from core.settings import DatabaseType, settings
from memory.mongodb import get_mongo_saver
from memory.postgres import get_postgres_saver, get_postgres_store
from memory.sqlite import get_sqlite_saver, get_sqlite_store


def initialize_database() -> AbstractAsyncContextManager[
    AsyncSqliteSaver | AsyncPostgresSaver | AsyncMongoDBSaver
]:
    """
    初始化“短期记忆”组件：LangGraph Checkpointer（数据库级持久化）。

    端到端链路中的位置：
    - 服务启动阶段（lifespan）创建 checkpointer；
    - 请求执行时，LangGraph 通过 checkpointer 把 thread state（messages 等）落盘/恢复。

    返回：
    - 一个 async context manager；进入后得到具体实现（SQLite/Postgres/Mongo）的 saver/checkpointer。
    """
    if settings.DATABASE_TYPE == DatabaseType.POSTGRES:
        return get_postgres_saver()
    if settings.DATABASE_TYPE == DatabaseType.MONGO:
        return get_mongo_saver()
    else:  # 默认使用 SQLite
        return get_sqlite_saver()


def initialize_store():
    """
    初始化“长期记忆”组件：LangGraph Store（更偏向跨线程/跨会话的存储接口）。

    注意（当前实现的取舍）：
    - Postgres：使用 `AsyncPostgresStore`，具备持久化能力。
    - SQLite：LangGraph 暂无“SQLite Store”实现，因此退化为 InMemoryStore（进程内，非持久化）。
    - Mongo：目前仅实现了 MongoDB checkpointer，store 尚未接入（因此也会走 SQLite 的默认分支）。
    """
    if settings.DATABASE_TYPE == DatabaseType.POSTGRES:
        return get_postgres_store()
    # TODO: 增加 Mongo store：https://pypi.org/project/langgraph-store-mongodb/
    else:  # 默认使用 SQLite
        return get_sqlite_store()


__all__ = ["initialize_database", "initialize_store"]
