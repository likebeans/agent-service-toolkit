"""
本文件在整个项目中的角色：HTTP Service 层（FastAPI）——把 Agent 能力以 API 形式对外暴露。

它解决的核心问题是什么？
- 对外提供稳定的服务接口（与 `src/client/client.py` 的调用/解析逻辑相互配套）：
  - `GET /info`：服务能力发现（可用 agents、可用 models、默认值）
  - `POST /{agent_id}/invoke`：一次性调用（仅返回最终 `schema.ChatMessage`）
  - `POST /{agent_id}/stream`：SSE 流式调用（中间消息 + token，最终以 `[DONE]` 结束）
  - `POST /feedback`：把 run_id 的评分/反馈转发给 LangSmith（凭证由服务端托管）
  - `POST /history`：按 thread_id 获取对话历史（从 checkpointer/state 里恢复）
  - `GET /health`：健康检查（可选验证 Langfuse 连通性）

系统运行时视角（端到端链路）：
1) 启动：FastAPI lifespan -> `initialize_database()/initialize_store()` -> `load_agent()` -> 将 saver/store 注入 agent
2) 请求进入：HTTP -> 校验 Bearer（可选）-> 解析 UserInput/StreamInput
3) Agent 编排：构造 RunnableConfig（thread_id/user_id/model/agent_config/run_id/callbacks）-> 调用 `agent.ainvoke()/agent.astream()`
4) 响应返回：
   - invoke：取最后一个事件，转为 `schema.ChatMessage` 返回 JSON
   - stream：把 LangGraph 事件编码为 SSE（message/token/error），最后发送 `data: [DONE]`

你在复刻项目时，可以把它当作“服务端协议定义”：
- 客户端只需要遵守这里的请求/响应约定，就能无痛替换内部的 Agent 实现。
"""

import inspect
import json
import logging
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse  # type: ignore[import-untyped]
from langfuse.langchain import (
    CallbackHandler,  # type: ignore[import-untyped]
)
from langgraph.types import Command, Interrupt
from langsmith import Client as LangsmithClient

from agents import DEFAULT_AGENT, AgentGraph, get_agent, get_all_agent_info, load_agent
from core import settings
from memory import initialize_database, initialize_store
from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    ServiceMetadata,
    StreamInput,
    UserInput,
)
from service.utils import (
    convert_message_content_to_string,
    langchain_to_chat_message,
    remove_tool_calls,
)

# LangChain/LangGraph 仍在快速迭代，beta warning 对学习/运行没有帮助，这里选择统一屏蔽。
warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)


def custom_generate_unique_id(route: APIRoute) -> str:
    # 让 OpenAPI operation_id 更稳定/可读（尤其对自动生成客户端 SDK 的场景很重要）。
    """Generate idiomatic operation IDs for OpenAPI client generation."""
    return route.name


def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.", auto_error=False)),
    ],
) -> None:
    """
    可选的 Bearer 鉴权校验。

    设计意图：
    - 若 `settings.AUTH_SECRET` 未配置：服务默认“无鉴权”（便于本地开发/快速上手）
    - 若配置了 AUTH_SECRET：要求客户端提供 `Authorization: Bearer <secret>`

    典型调用者：
    - `router = APIRouter(dependencies=[Depends(verify_bearer)])`：将其作为全局依赖应用到所有路由。
    """
    if not settings.AUTH_SECRET:
        return
    auth_secret = settings.AUTH_SECRET.get_secret_value()
    if not http_auth or http_auth.credentials != auth_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Configurable lifespan that initializes the appropriate database checkpointer, store,
    and agents with async loading - for example for starting up MCP clients.
    """
    try:
        # 初始化两类“记忆组件”：
        # - checkpointer：短期记忆/对话线程状态（messages 等）
        # - store：长期记忆/跨线程知识（取决于具体 Agent 是否使用）
        async with initialize_database() as saver, initialize_store() as store:
            # 统一做 setup（对齐不同后端的初始化行为）
            if hasattr(saver, "setup"):  # ignore: union-attr
                await saver.setup()
            # store 只有部分实现需要 setup（例如 PostgresStore），InMemoryStore 则是 no-op
            if hasattr(store, "setup"):  # ignore: union-attr
                await store.setup()

            # 启动期预加载所有 Agent，并把 memory 组件注入进去：
            # - 对 LazyLoadingAgent：这里会触发外部依赖初始化（例如 MCP client 建连）
            # - 对普通 Agent：load 是 no-op，但统一走一遍能让启动行为一致
            agents = get_all_agent_info()
            for a in agents:
                try:
                    await load_agent(a.key)
                    logger.info(f"Agent loaded: {a.key}")
                except Exception as e:
                    logger.error(f"Failed to load agent {a.key}: {e}")
                    # 不中断启动：允许某个 Agent 加载失败，但服务仍然可用（至少能用其它 Agent）

                agent = get_agent(a.key)
                # 注入 checkpointer：用于 thread 级对话历史/状态恢复
                agent.checkpointer = saver
                # 注入 store：用于长期记忆（跨会话知识/画像等，具体由 Agent 决定是否使用）
                agent.store = store
            yield
    except Exception as e:
        logger.error(f"Error during database/store/agents initialization: {e}")
        raise


app = FastAPI(lifespan=lifespan, generate_unique_id_function=custom_generate_unique_id)
# 将鉴权依赖挂到 router 上：router 内所有路由都会自动执行 `verify_bearer`。
router = APIRouter(dependencies=[Depends(verify_bearer)])


@router.get("/info")
async def info() -> ServiceMetadata:
    # 服务能力发现端点：客户端 UI/SDK 用它来构建下拉框（agent/model）与默认值。
    models = list(settings.AVAILABLE_MODELS)
    models.sort()
    return ServiceMetadata(
        agents=get_all_agent_info(),
        models=models,
        default_agent=DEFAULT_AGENT,
        default_model=settings.DEFAULT_MODEL,
    )


async def _handle_input(user_input: UserInput, agent: AgentGraph) -> tuple[dict[str, Any], UUID]:
    """
    Parse user input and handle any required interrupt resumption.
    Returns kwargs for agent invocation and the run_id.
    """
    # run_id：一次调用的唯一标识，主要用于 tracing/feedback（例如 LangSmith 评分）
    run_id = uuid4()
    # thread_id：对话线程标识，决定“短期记忆”的边界（同 thread 才能续聊）
    thread_id = user_input.thread_id or str(uuid4())
    # user_id：用户标识，决定“长期记忆”的边界（跨 thread 的用户画像/知识）
    user_id = user_input.user_id or str(uuid4())

    # RunnableConfig.configurable 会下沉到 LangGraph/LangChain 的运行环境中，
    # Agent 节点通常会从这里读取 model/thread/user 等配置。
    configurable = {"thread_id": thread_id, "user_id": user_id}
    if user_input.model is not None:
        configurable["model"] = user_input.model

    callbacks: list[Any] = []
    if settings.LANGFUSE_TRACING:
        # Langfuse tracing：通过 callback handler 把链路追踪事件上报到 Langfuse。
        langfuse_handler = CallbackHandler()

        callbacks.append(langfuse_handler)

    if user_input.agent_config:
        # agent_config 是“透传配置”，但必须防止覆盖保留字段，避免破坏 thread/user/model 的一致性。
        #（即使用户没显式传 model，model 也属于保留字段）
        reserved_keys = {"thread_id", "user_id", "model"}
        if overlap := reserved_keys & user_input.agent_config.keys():
            raise HTTPException(
                status_code=422,
                detail=f"agent_config contains reserved keys: {overlap}",
            )
        configurable.update(user_input.agent_config)

    # LangGraph 执行时的统一 config：
    # - configurable：业务侧可控参数
    # - run_id：用于 tracing/feedback 关联
    # - callbacks：用于 tracing（Langfuse/LangSmith 等）
    config = RunnableConfig(
        configurable=configurable,
        run_id=run_id,
        callbacks=callbacks,
    )

    # Interrupt 机制（LangGraph）：如果上一次执行停在 interrupt 上，
    # 这次 user_input.message 被视为“resume 输入”而不是新的 HumanMessage。
    state = await agent.aget_state(config=config)
    interrupted_tasks = [
        task for task in state.tasks if hasattr(task, "interrupts") and task.interrupts
    ]

    input: Command | dict[str, Any]
    if interrupted_tasks:
        # 将用户输入作为 resume value 继续执行 graph（而不是开始新一轮对话）
        input = Command(resume=user_input.message)
    else:
        # 普通对话：把用户输入包装为 HumanMessage 注入图的 messages state
        input = {"messages": [HumanMessage(content=user_input.message)]}

    kwargs = {
        "input": input,
        "config": config,
    }

    return kwargs, run_id


@router.post("/{agent_id}/invoke", operation_id="invoke_with_agent_id")
@router.post("/invoke")
async def invoke(user_input: UserInput, agent_id: str = DEFAULT_AGENT) -> ChatMessage:
    """
    Invoke an agent with user input to retrieve a final response.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    Use user_id to persist and continue a conversation across multiple threads.
    """
    # 注意：该接口当前只返回“最后一个消息或 interrupt 值”。
    # 如果 Agent 在一次执行中产生多个 AIMessage（例如 interrupt-agent 的后台步骤，
    # 或 research-assistant 的工具步骤），中间消息不会被返回。
    # 更通用的方案是返回 ChatMessage 列表（或返回 events），但这里选择了更简单的契约。
    agent: AgentGraph = get_agent(agent_id)
    kwargs, run_id = await _handle_input(user_input, agent)

    try:
        response_events: list[tuple[str, Any]] = await agent.ainvoke(**kwargs, stream_mode=["updates", "values"])  # type: ignore # fmt: skip
        response_type, response = response_events[-1]
        if response_type == "values":
            # 正常完成：values 里会包含最终 state（messages 等）
            output = langchain_to_chat_message(response["messages"][-1])
        elif response_type == "updates" and "__interrupt__" in response:
            # 最后发生的是 interrupt：把 interrupt.value 当作一条 AIMessage 返回给客户端展示
            output = langchain_to_chat_message(
                AIMessage(content=response["__interrupt__"][0].value)
            )
        else:
            raise ValueError(f"Unexpected response type: {response_type}")

        output.run_id = str(run_id)
        return output
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


async def message_generator(
    user_input: StreamInput, agent_id: str = DEFAULT_AGENT
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    """
    agent: AgentGraph = get_agent(agent_id)
    kwargs, run_id = await _handle_input(user_input, agent)

    try:
        # 这是 `/stream` 的核心：把 LangGraph 的 stream 事件翻译为 SSE（text/event-stream）输出。
        # SSE 协议约定（与 `src/client/client.py:_parse_stream_line()` 对齐）：
        # - message：完整 ChatMessage（中间/最终消息）
        # - token：模型输出 token（可选，取决于 user_input.stream_tokens）
        # - error：服务端错误事件
        # - [DONE]：流结束标记
        async for stream_event in agent.astream(
            **kwargs, stream_mode=["updates", "messages", "custom"], subgraphs=True
        ):
            if not isinstance(stream_event, tuple):
                continue
            # stream_event 的 tuple 结构会因是否开启 subgraphs 而变化：
            # - subgraphs=True: (node_path, stream_mode, event)
            # - subgraphs=False: (stream_mode, event)
            if len(stream_event) == 3:
                # subgraphs=True: (node_path, stream_mode, event)
                _, stream_mode, event = stream_event
            else:
                # subgraphs=False: (stream_mode, event)
                stream_mode, event = stream_event
            new_messages = []
            if stream_mode == "updates":
                for node, updates in event.items():
                    # Interrupt 的简易处理：
                    # - LangGraph 把 interrupt 作为特殊 node `__interrupt__` 输出
                    # - 这里把 interrupt.value 转成 AIMessage，作为“中间消息”推送给客户端
                    # 更完善的方案：定义结构化的 ChatMessage（type="interrupt"）返回，方便 UI 专门渲染。
                    if node == "__interrupt__":
                        interrupt: Interrupt
                        for interrupt in updates:
                            new_messages.append(AIMessage(content=interrupt.value))
                        continue
                    updates = updates or {}
                    update_messages = updates.get("messages", [])
                    # langgraph-supervisor 的特殊兼容：
                    # supervisor/子 agent 会产生一些 tool message 作为“handoff/handback”机制的一部分；
                    # 这里做裁剪，避免把过多内部细节暴露给客户端 UI。
                    if "supervisor" in node or "sub-agent" in node:
                        # 实际来自 agent 的“工具消息”主要是 handoff/handback 这两类
                        if isinstance(update_messages[-1], ToolMessage):
                            if "sub-agent" in node and len(update_messages) > 1:
                                # sub-agent：保留最后 2 条消息（handback tool + tool result）
                                update_messages = update_messages[-2:]
                            else:
                                # supervisor：只保留最后 1 条消息（handoff 结果；tool 来自 agent 节点）
                                update_messages = [update_messages[-1]]
                        else:
                            update_messages = []
                    new_messages.extend(update_messages)

            if stream_mode == "custom":
                # custom 事件通常来自图内部显式发送的自定义消息（例如用于 UI 展示结构化信息）。
                new_messages = [event]

            # LangGraph 在 streaming 时可能把一个 AIMessage 拆成多个 tuple 片段：
            # 例如 ('content', <str>), ('tool_calls', [...]), ('additional_kwargs', {...}) 等。
            # 这里把这些片段累积成 dict，最后再拼成一个完整的 AIMessage，避免客户端看到碎片化事件。
            # 参考：https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_messages/
            processed_messages = []
            current_message: dict[str, Any] = {}
            for message in new_messages:
                if isinstance(message, tuple):
                    key, value = message
                    # 临时累积 message 的字段片段
                    current_message[key] = value
                else:
                    # 如果当前正在累积一个 message，则先 flush；再追加完整 message
                    if current_message:
                        processed_messages.append(_create_ai_message(current_message))
                        current_message = {}
                    processed_messages.append(message)

            # flush：把最后尚未输出的片段拼装成完整消息
            if current_message:
                processed_messages.append(_create_ai_message(current_message))

            for message in processed_messages:
                try:
                    chat_message = langchain_to_chat_message(message)
                    chat_message.run_id = str(run_id)
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'})}\n\n"
                    continue
                # LangGraph 可能会把输入消息（human）也作为事件回放出来；客户端一般不希望重复展示，所以丢弃。
                if chat_message.type == "human" and chat_message.content == user_input.message:
                    continue
                yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

            if stream_mode == "messages":
                if not user_input.stream_tokens:
                    continue
                msg, metadata = event
                if "skip_stream" in metadata.get("tags", []):
                    continue
                # `stream_mode="messages"` 理论上只应有 LLM token chunk，但某些情况下非 LLM 节点也会产出消息。
                # 这里做一次过滤，避免把非 token 的事件当作 token 发送给客户端。
                if not isinstance(msg, AIMessageChunk):
                    continue
                content = remove_tool_calls(msg.content)
                if content:
                    # 在 OpenAI 等协议里，空 content 往往意味着“模型正在请求工具调用”。
                    # 因此这里仅发送非空 token，避免客户端渲染出大量空片段。
                    yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"
    except Exception as e:
        logger.error(f"Error in message generator: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': 'Internal server error'})}\n\n"
    finally:
        # 无论是否异常，都发送 DONE：客户端依赖这个标记来停止读取（见 `src/client/client.py`）。
        yield "data: [DONE]\n\n"


def _create_ai_message(parts: dict) -> AIMessage:
    # LangGraph 可能把 message 拆成多个字段片段；这里用 AIMessage 的签名做一次白名单过滤再构造。
    sig = inspect.signature(AIMessage)
    valid_keys = set(sig.parameters)
    filtered = {k: v for k, v in parts.items() if k in valid_keys}
    return AIMessage(**filtered)


def _sse_response_example() -> dict[int | str, Any]:
    # OpenAPI 文档示例：告诉使用方 `/stream` 返回的是 text/event-stream 的 SSE 文本。
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",
                    "schema": {"type": "string"},
                }
            },
        }
    }


@router.post(
    "/{agent_id}/stream",
    response_class=StreamingResponse,
    responses=_sse_response_example(),
    operation_id="stream_with_agent_id",
)
@router.post("/stream", response_class=StreamingResponse, responses=_sse_response_example())
async def stream(user_input: StreamInput, agent_id: str = DEFAULT_AGENT) -> StreamingResponse:
    """
    Stream an agent's response to a user input, including intermediate messages and tokens.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.
    Use user_id to persist and continue a conversation across multiple threads.

    Set `stream_tokens=false` to return intermediate messages but not token-by-token.
    """
    return StreamingResponse(
        message_generator(user_input, agent_id),
        media_type="text/event-stream",
    )


@router.post("/feedback")
async def feedback(feedback: Feedback) -> FeedbackResponse:
    """
    Record feedback for a run to LangSmith.

    This is a simple wrapper for the LangSmith create_feedback API, so the
    credentials can be stored and managed in the service rather than the client.
    See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
    """
    client = LangsmithClient()
    kwargs = feedback.kwargs or {}
    client.create_feedback(
        run_id=feedback.run_id,
        key=feedback.key,
        score=feedback.score,
        **kwargs,
    )
    return FeedbackResponse()


@router.post("/history")
async def history(input: ChatHistoryInput) -> ChatHistory:
    """
    Get chat history.
    """
    # TODO: 这里硬编码 DEFAULT_AGENT 有点别扭；更理想的是按 agent_id 拉取对应 thread 的 state。
    agent: AgentGraph = get_agent(DEFAULT_AGENT)
    try:
        state_snapshot = await agent.aget_state(
            config=RunnableConfig(configurable={"thread_id": input.thread_id})
        )
        messages: list[AnyMessage] = state_snapshot.values["messages"]
        chat_messages: list[ChatMessage] = [langchain_to_chat_message(m) for m in messages]
        return ChatHistory(messages=chat_messages)
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


@app.get("/health")
async def health_check():
    """Health check endpoint."""

    health_status = {"status": "ok"}

    if settings.LANGFUSE_TRACING:
        try:
            langfuse = Langfuse()
            health_status["langfuse"] = "connected" if langfuse.auth_check() else "disconnected"
        except Exception as e:
            logger.error(f"Langfuse connection error: {e}")
            health_status["langfuse"] = "disconnected"

    return health_status


app.include_router(router)
