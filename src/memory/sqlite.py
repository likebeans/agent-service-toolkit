"""
本文件在整个项目中的角色：SQLite 版 Memory 后端实现（Checkpointer + Store 兼容层）。

它解决的核心问题是什么？
- 为 LangGraph 提供基于 SQLite 的 checkpointer（对话线程状态持久化）。
- 同时为“长期记忆 store”提供一个兼容实现：
  - LangGraph 目前没有 SQLite 版 store；
  - 因此这里用 `InMemoryStore` 作为替代，并包一层 async context manager，
    让上层（`src/service/service.py:lifespan()`）可以用统一的 `async with ...` 写法。

典型调用者：
- `src/memory/__init__.py`：在 `settings.DATABASE_TYPE` 为 SQLite（默认）时选择本实现。
"""

from contextlib import AbstractAsyncContextManager, asynccontextmanager

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.memory import InMemoryStore

from core.settings import settings


def get_sqlite_saver() -> AbstractAsyncContextManager[AsyncSqliteSaver]:
    """
    创建 SQLite checkpointer（AsyncSqliteSaver）。

    业务意义：
    - 负责持久化 LangGraph 的 thread state（尤其是对话 messages），支撑多轮对话与服务重启恢复。
    """
    return AsyncSqliteSaver.from_conn_string(settings.SQLITE_DB_PATH)


class AsyncInMemoryStore:
    """
    将 `InMemoryStore` 包装为“异步上下文管理器”。

    为什么要包这一层？
    - 上层 `lifespan()` 期望所有 store 都能 `async with` 使用（以便对齐 PostgresStore 的生命周期）。
    - InMemoryStore 不需要真实资源释放，但我们仍提供一致的接口，降低上层分支复杂度。

    重要限制：
    - InMemoryStore 仅存在于进程内：服务重启即丢失，不适合真正的“长期记忆”。
    """

    def __init__(self):
        self.store = InMemoryStore()

    async def __aenter__(self):
        return self.store

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # InMemoryStore 无需清理资源；这里保留接口形态即可。
        pass

    async def setup(self):
        # 为了与 PostgresStore 接口对齐而提供的“空操作（no-op）”：
        # `src/service/service.py:lifespan()` 会检测是否存在 setup() 并调用。
        pass


@asynccontextmanager
async def get_sqlite_store():
    """
    创建“长期记忆 store”（SQLite 场景下的兼容实现）。

    注意：
    - 由于 LangGraph 暂无 SQLite Store，这里退化为 InMemoryStore；
    - 该 store 仅为接口兼容与本地开发便利，生产环境更推荐使用 Postgres（或未来的 Mongo Store）。
    """
    store_manager = AsyncInMemoryStore()
    yield await store_manager.__aenter__()
