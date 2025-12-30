"""
本文件在整个项目中的角色：MongoDB 版 Checkpointer（对话线程状态持久化）。

它解决的核心问题是什么？
- 当你希望把 LangGraph 的 thread state（messages 等）持久化到 MongoDB 时，
  这里负责：
  - 校验 Mongo 配置是否齐全
  - 生成连接串（支持可选认证）
  - 创建 `AsyncMongoDBSaver`（LangGraph 提供的 Mongo checkpointer 实现）

重要现状（设计取舍）：
- 本项目当前只接入了 MongoDB 的 checkpointer；
- “长期记忆 store” 尚未接入 MongoDB（见 `src/memory/__init__.py:initialize_store()` 的 TODO）。

典型调用者：
- `src/memory/__init__.py:initialize_database()`：当 `settings.DATABASE_TYPE == mongo` 时选择本实现。
"""

import logging
import urllib.parse
from contextlib import AbstractAsyncContextManager

from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver

from core.settings import settings

logger = logging.getLogger(__name__)


def _has_auth_credentials() -> bool:
    """
    判断 Mongo 认证信息是否完整。

    规则：
    - 三者（MONGO_USER / MONGO_PASSWORD / MONGO_AUTH_SOURCE）要么全都提供，要么全都不提供。
    - 如果只提供了一部分，会直接抛错，避免生成半残缺连接串导致更隐蔽的问题。
    """
    required_auth = ["MONGO_USER", "MONGO_PASSWORD", "MONGO_AUTH_SOURCE"]
    set_auth = [var for var in required_auth if getattr(settings, var, None)]
    if len(set_auth) > 0 and len(set_auth) != len(required_auth):
        raise ValueError(
            f"If any of the following environment variables are set, all must be set: {', '.join(required_auth)}."
        )
    return len(set_auth) == len(required_auth)


def validate_mongo_config() -> None:
    """
    校验 MongoDB 配置是否齐全（Fail Fast）。

    注意：
    - 这里仅校验“能否建立连接”，不负责创建数据库/集合。
    """
    required_always = ["MONGO_HOST", "MONGO_PORT", "MONGO_DB"]
    missing_always = [var for var in required_always if not getattr(settings, var, None)]
    if missing_always:
        raise ValueError(
            f"Missing required MongoDB configuration: {', '.join(missing_always)}. "
            "These environment variables must be set to use MongoDB persistence."
        )

    _has_auth_credentials()


def get_mongo_connection_string() -> str:
    """
    从 settings 拼装 MongoDB 连接串。

    复杂点：
    - 密码可能包含特殊字符，需要 URL encode（这里用 `quote_plus`）。
    - 认证库（authSource）需要写在 query 参数中。
    """

    if _has_auth_credentials():
        if settings.MONGO_PASSWORD is None:  # 用于类型检查
            raise ValueError("MONGO_PASSWORD is not set")
        password = settings.MONGO_PASSWORD.get_secret_value().strip()
        password_escaped = urllib.parse.quote_plus(password)
        return (
            f"mongodb://{settings.MONGO_USER}:{password_escaped}@"
            f"{settings.MONGO_HOST}:{settings.MONGO_PORT}/"
            f"?authSource={settings.MONGO_AUTH_SOURCE}"
        )
    else:
        return f"mongodb://{settings.MONGO_HOST}:{settings.MONGO_PORT}/"


def get_mongo_saver() -> AbstractAsyncContextManager[AsyncMongoDBSaver]:
    """
    创建 MongoDB checkpointer（AsyncMongoDBSaver）。

    返回：
    - 一个 context manager（与 SQLite/Postgres saver 的接口形态保持一致），
      便于上层统一用 `async with` 管理生命周期。
    """
    validate_mongo_config()
    if settings.MONGO_DB is None:  # 用于类型检查
        raise ValueError("MONGO_DB is not set")
    return AsyncMongoDBSaver.from_conn_string(
        get_mongo_connection_string(), db_name=settings.MONGO_DB
    )
