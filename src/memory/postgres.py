"""
本文件在整个项目中的角色：PostgreSQL 版 Memory 后端实现（Checkpointer + Store）。

它解决的核心问题是什么？
- 为 LangGraph 提供“可持久化、可扩展”的两类能力：
  1) Checkpointer（`AsyncPostgresSaver`）：保存/恢复每个 thread 的 graph state（对话历史等）
  2) Store（`AsyncPostgresStore`）：长期记忆/跨线程存储接口
- 同时提供连接池（`psycopg_pool.AsyncConnectionPool`）封装：
  - 让连接更健壮（自动检查连接有效性、控制连接数）
  - 在服务 lifespan 生命周期内安全地初始化与释放资源

典型调用者：
- `src/memory/__init__.py`：当 `settings.DATABASE_TYPE == postgres` 时会选择这里的实现。
- `src/service/service.py:lifespan()`：启动期 `async with` 获取 saver/store，并注入到各 Agent。
"""

import logging
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from core.settings import settings

logger = logging.getLogger(__name__)


def validate_postgres_config() -> None:
    """
    校验 Postgres 配置是否齐全。

    为什么要在这里做校验？
    - 连接池/数据库初始化失败通常会导致服务无法正常启动；
    - 早在启动期把缺失项报出来，比在运行时随机报错更易定位。

    注意：这里不修改任何配置，只负责“Fail Fast”。
    """
    required_vars = [
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
    ]

    missing = [var for var in required_vars if not getattr(settings, var, None)]
    if missing:
        raise ValueError(
            f"Missing required PostgreSQL configuration: {', '.join(missing)}. "
            "These environment variables must be set to use PostgreSQL persistence."
        )

    if settings.POSTGRES_MIN_CONNECTIONS_PER_POOL > settings.POSTGRES_MAX_CONNECTIONS_PER_POOL:
        raise ValueError(
            f"POSTGRES_MIN_CONNECTIONS_PER_POOL ({settings.POSTGRES_MIN_CONNECTIONS_PER_POOL}) must be less than or equal to POSTGRES_MAX_CONNECTIONS_PER_POOL ({settings.POSTGRES_MAX_CONNECTIONS_PER_POOL})"
        )


def get_postgres_connection_string() -> str:
    """
    从 settings 拼装 Postgres 连接串。

    设计提示：
    - 密码使用 `SecretStr.get_secret_value()` 取出真实值（避免在 repr/log 里意外泄露）。
    - 该连接串会被交给 psycopg 连接池使用。
    """
    if settings.POSTGRES_PASSWORD is None:
        raise ValueError("POSTGRES_PASSWORD is not set")
    return (
        f"postgresql://{settings.POSTGRES_USER}:"
        f"{settings.POSTGRES_PASSWORD.get_secret_value()}@"
        f"{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/"
        f"{settings.POSTGRES_DB}"
    )


@asynccontextmanager
async def get_postgres_saver():
    """
    创建 Postgres checkpointer（AsyncPostgresSaver）。

    实现要点：
    - 使用连接池而不是单连接：提高稳定性与并发承载能力。
    - `kwargs={"autocommit": True, "row_factory": dict_row, ...}`：
      - LangGraph 的 Postgres checkpointer 要求 autocommit；
      - row_factory 设为 dict_row 便于以 dict 形式读写记录。
    - `application_name` 帮你在 Postgres 侧区分连接来源（saver vs store）。
    """
    validate_postgres_config()
    application_name = settings.POSTGRES_APPLICATION_NAME + "-" + "saver"

    async with AsyncConnectionPool(
        get_postgres_connection_string(),
        min_size=settings.POSTGRES_MIN_CONNECTIONS_PER_POOL,
        max_size=settings.POSTGRES_MAX_CONNECTIONS_PER_POOL,
        # LangGraph 要求 autocommit=true，且 row_factory 需要设置为 dict_row。
        # 通过 application_name 你可以在 Postgres 的连接管理/监控工具里识别这些连接。
        kwargs={"autocommit": True, "row_factory": dict_row, "application_name": application_name},
        # 在使用连接前检查连接是否仍然有效（避免连接已失效/陈旧）
        check=AsyncConnectionPool.check_connection,
    ) as pool:
        try:
            checkpointer = AsyncPostgresSaver(pool)
            # setup() 会创建/迁移 checkpointer 所需的表结构（由 LangGraph 提供实现）。
            await checkpointer.setup()
            yield checkpointer
        finally:
            await pool.close()


@asynccontextmanager
async def get_postgres_store():
    """
    创建 Postgres store（AsyncPostgresStore）。

    业务意义：
    - 作为“长期记忆/跨线程存储”的后端；
    - 与 checkpointer 的区别：checkpointer 更偏 thread 内状态快照；store 更偏通用 KV/文档式存储接口。

    """
    validate_postgres_config()
    application_name = settings.POSTGRES_APPLICATION_NAME + "-" + "store"

    async with AsyncConnectionPool(
        get_postgres_connection_string(),
        min_size=settings.POSTGRES_MIN_CONNECTIONS_PER_POOL,
        max_size=settings.POSTGRES_MAX_CONNECTIONS_PER_POOL,
        # LangGraph 要求 autocommit=true，且 row_factory 需要设置为 dict_row。
        # 通过 application_name 你可以在 Postgres 的连接管理/监控工具里识别这些连接。
        kwargs={"autocommit": True, "row_factory": dict_row, "application_name": application_name},
        # 在使用连接前检查连接是否仍然有效（避免连接已失效/陈旧）
        check=AsyncConnectionPool.check_connection,
    ) as pool:
        try:
            store = AsyncPostgresStore(pool)
            # setup() 会创建/迁移 store 所需的表结构（由 LangGraph 提供实现）。
            await store.setup()
            yield store
        finally:
            await pool.close()
