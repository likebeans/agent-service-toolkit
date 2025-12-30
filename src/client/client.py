"""
本文件在整个项目中的角色：Agent Service 的 Python 客户端（SDK）。

它解决的核心问题是什么？
- 把服务端（`src/service/service.py`）暴露的 HTTP API 封装成易用的 Python 方法：
  - `/info`：获取可用 Agent、默认值、模型列表等服务元信息
  - `/{agent_id}/invoke`：一次性调用（只返回最终 ChatMessage）
  - `/{agent_id}/stream`：SSE 流式调用（同时返回“中间消息”和“token”）
  - `/feedback`：将 run_id 的反馈转发给 LangSmith（由服务端托管凭证）
  - `/history`：获取某个 thread 的历史消息（用于恢复对话）

典型调用者是谁？
- 前端/UI（例如 Streamlit）或命令行工具：它们不直接拼 URL/处理 SSE，而是调用这里的方法。
- 你在复刻项目时，也可以把它当作“客户端协议定义”：服务端输出什么，这里就解析什么。

与端到端链路的关系（高层视角）：
1) UI/脚本 -> `AgentClient.invoke()/stream()` 组装请求体（thread_id/user_id/model/agent_config）
2) 通过 `httpx` 调用 FastAPI 路由（`src/service/service.py`）
3) 服务端执行 LangGraph Agent（调用 LLM/Tools），并把过程事件编码为 SSE
4) `AgentClient` 解析 SSE 行（`_parse_stream_line()`）并把结果 yield 给上层消费
"""

import json
import os
from collections.abc import AsyncGenerator, Generator
from typing import Any

import httpx

from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    ServiceMetadata,
    StreamInput,
    UserInput,
)


class AgentClientError(Exception):
    """客户端侧异常：统一包装 HTTP/SSE 交互、解析失败等错误。"""

    pass


class AgentClient:
    """
    与 Agent Service 交互的客户端（同步/异步均支持）。

    设计意图：
    - 把“服务端协议细节”（URL、鉴权头、Pydantic schema、SSE 事件格式）收敛在一个类里；
      UI/业务侧只关心：invoke/stream 得到什么结果。
    - 同时提供 sync/async 两套 API：
      - `invoke()` / `stream()` 适合脚本/简单调用
      - `ainvoke()` / `astream()` 适合异步应用与高并发场景
    """

    def __init__(
        self,
        base_url: str = "http://0.0.0.0",
        agent: str | None = None,
        timeout: float | None = None,
        get_info: bool = True,
    ) -> None:
        """
        初始化客户端。

        Args:
            base_url (str): 服务端 base URL（例如 `http://localhost:8000`）
            agent (str): 默认使用的 agent key（对应服务端 `/info` 里返回的 agents[].key）
            timeout (float, optional): httpx 请求超时（秒）
            get_info (bool, optional): 初始化时是否拉取 `/info` 来填充 `self.info` 并设置默认 agent。
                默认：True
        """
        self.base_url = base_url
        # 与服务端 `src/service/service.py:verify_bearer()` 配套：
        # - 服务端如果配置了 AUTH_SECRET，会要求 `Authorization: Bearer <secret>`。
        # - 客户端从环境变量读取，避免在代码里硬编码。
        self.auth_secret = os.getenv("AUTH_SECRET")
        self.timeout = timeout
        self.info: ServiceMetadata | None = None
        self.agent: str | None = None
        if get_info:
            self.retrieve_info()
        if agent:
            self.update_agent(agent)

    @property
    def _headers(self) -> dict[str, str]:
        # 这里集中生成 headers，便于后续扩展（例如加自定义 user-agent、trace id 等）。
        headers = {}
        if self.auth_secret:
            headers["Authorization"] = f"Bearer {self.auth_secret}"
        return headers

    def retrieve_info(self) -> None:
        """
        拉取服务元信息（`GET /info`）。

        业务意义：
        - 客户端需要知道：有哪些 agent 可用、默认 agent/model 是什么，供 UI 下拉框/路由选择使用。
        - 同时也用于 `update_agent(..., verify=True)` 的校验依据。
        """
        try:
            response = httpx.get(
                f"{self.base_url}/info",
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error getting service info: {e}")

        self.info = ServiceMetadata.model_validate(response.json())
        # 如果当前没有选定 agent（或选的 agent 不在列表里），回退到服务端默认值。
        if not self.agent or self.agent not in [a.key for a in self.info.agents]:
            self.agent = self.info.default_agent

    def update_agent(self, agent: str, verify: bool = True) -> None:
        """
        设置默认 agent。

        为什么这里要提供 verify？
        - 作为 SDK，默认希望“早失败”：如果传入不存在的 agent key，立刻在客户端抛错，
          而不是等到请求时才 404/422。
        - 某些情况下你可能希望跳过校验（例如离线/启动期服务还没起来），因此提供 `verify=False`。
        """
        if verify:
            if not self.info:
                self.retrieve_info()
            agent_keys = [a.key for a in self.info.agents]  # type: ignore[union-attr]
            if agent not in agent_keys:
                raise AgentClientError(
                    f"Agent {agent} not found in available agents: {', '.join(agent_keys)}"
                )
        self.agent = agent

    async def ainvoke(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """
        异步调用 agent（只返回最终消息）。

        Args:
            message (str): 用户输入（通常会成为 HumanMessage）
            model (str, optional): 指定服务端使用的模型（会进入 RunnableConfig.configurable）
            thread_id (str, optional): 对话线程 ID（短期记忆/对话历史的 key）
            user_id (str, optional): 用户 ID（跨 thread 的长期记忆/画像的 key）
            agent_config (dict[str, Any], optional): 透传给 agent 的额外配置（服务端会与保留字段合并校验）

        Returns:
            ChatMessage: agent 最终输出（通常 type="ai"），并带有 run_id 便于反馈
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        # 请求体使用 `src/schema/*` 中定义的 Pydantic 模型，确保与服务端契约一致。
        request = UserInput(message=message)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/{self.agent}/invoke",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

        return ChatMessage.model_validate(response.json())

    def invoke(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """
        同步调用 agent（只返回最终消息）。

        Args:
            message (str): 用户输入（通常会成为 HumanMessage）
            model (str, optional): 指定服务端使用的模型
            thread_id (str, optional): 对话线程 ID（用于延续多轮）
            user_id (str, optional): 用户 ID（用于跨 thread 的记忆/画像）
            agent_config (dict[str, Any], optional): 透传给 agent 的额外配置

        Returns:
            ChatMessage: agent 最终输出
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = UserInput(message=message)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        try:
            response = httpx.post(
                f"{self.base_url}/{self.agent}/invoke",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

        return ChatMessage.model_validate(response.json())

    def _parse_stream_line(self, line: str) -> ChatMessage | str | None:
        """
        解析服务端 SSE（Server-Sent Events）单行 `data:` payload。

        与服务端的契约（见 `src/service/service.py:message_generator()`）：
        - 服务端按行发送：
          - `data: {"type":"message","content": <ChatMessage dict>}`
          - `data: {"type":"token","content": <str>}`（可选，取决于 stream_tokens）
          - `data: {"type":"error","content": <str>}`
          - `data: [DONE]`（流结束标记）

        返回值约定（供 `stream()/astream()` 使用）：
        - `ChatMessage`：一个完整的“中间/最终消息”
        - `str`：一个 token（用于 UI 增量拼接）
        - `None`：表示流结束（遇到 [DONE] 或非 data 行等）
        """
        line = line.strip()
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                return None
            try:
                parsed = json.loads(data)
            except Exception as e:
                raise Exception(f"Error JSON parsing message from server: {e}")
            match parsed["type"]:
                case "message":
                    # 将 JSON 形式的 ChatMessage 反序列化为对象；这里失败通常意味着服务端契约被破坏。
                    try:
                        return ChatMessage.model_validate(parsed["content"])
                    except Exception as e:
                        raise Exception(f"Server returned invalid message: {e}")
                case "token":
                    # token 直接按字符串向上游 yield，由上游决定如何拼接/展示。
                    return parsed["content"]
                case "error":
                    # 服务端错误统一包装为一个 AI 消息返回，便于 UI 按“消息”路径展示。
                    error_msg = "Error: " + parsed["content"]
                    return ChatMessage(type="ai", content=error_msg)
        return None

    def stream(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        stream_tokens: bool = True,
    ) -> Generator[ChatMessage | str, None, None]:
        """
        同步流式调用 agent（SSE）。

        你会在迭代器里收到两类产物：
        - `ChatMessage`：图执行过程中产生的“中间消息”（例如工具调用结果、阶段性总结、最终答复等）
        - `str`：模型输出 token（当 `stream_tokens=True` 时）

        设计提示（写 UI 时很重要）：
        - token 事件通常需要“追加到当前正在生成的 assistant 消息”；
        - message 事件通常表示“一个完整消息已经就绪”，可以直接追加到消息列表里。

        Args:
            message (str): 用户输入
            model (str, optional): 指定模型
            thread_id (str, optional): 对话线程 ID
            user_id (str, optional): 用户 ID
            agent_config (dict[str, Any], optional): 透传给 agent 的额外配置
            stream_tokens (bool, optional): 是否输出 token 级别事件（默认 True）

        Returns:
            Generator[ChatMessage | str, None, None]: 逐步产出消息/token，直到 [DONE]
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = StreamInput(message=message, stream_tokens=stream_tokens)
        if thread_id:
            request.thread_id = thread_id
        if user_id:
            request.user_id = user_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/{self.agent}/stream",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.strip():
                        # 每一行是一条 SSE `data:` 事件；解析后要么 yield，要么遇到 [DONE] 结束。
                        parsed = self._parse_stream_line(line)
                        if parsed is None:
                            break
                        yield parsed
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

    async def astream(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        stream_tokens: bool = True,
    ) -> AsyncGenerator[ChatMessage | str, None]:
        """
        异步流式调用 agent（SSE）。

        语义与 `stream()` 相同，只是返回 async generator，适合异步应用。

        Args:
            message (str): 用户输入
            model (str, optional): 指定模型
            thread_id (str, optional): 对话线程 ID
            user_id (str, optional): 用户 ID
            agent_config (dict[str, Any], optional): 透传给 agent 的额外配置
            stream_tokens (bool, optional): 是否输出 token 级别事件（默认 True）

        Returns:
            AsyncGenerator[ChatMessage | str, None]: 逐步产出消息/token
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = StreamInput(message=message, stream_tokens=stream_tokens)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/{self.agent}/stream",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            parsed = self._parse_stream_line(line)
                            if parsed is None:
                                break
                            # 不要 yield 空字符串 token：部分消费端会把它当作“生成结束/无效事件”导致问题。
                            if parsed != "":
                                yield parsed
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

    async def acreate_feedback(
        self, run_id: str, key: str, score: float, kwargs: dict[str, Any] = {}
    ) -> None:
        """
        为某次 run 记录反馈（异步）。

        设计意图：
        - 这是对 LangSmith `create_feedback` 的轻量封装，但把凭证放在服务端统一管理，
          客户端只需要把 run_id/score 等信息发给 `/feedback` 即可。
        参考：https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
        """
        # 注意：这里的 kwargs 默认值是可变对象（{}）。这在一般代码里并不推荐，
        # 但为了“不改变原有行为”，我们只在注释里提醒，保持实现不动。
        request = Feedback(run_id=run_id, key=key, score=score, kwargs=kwargs)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/feedback",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                response.json()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

    def get_history(self, thread_id: str) -> ChatHistory:
        """
        获取某个 thread 的对话历史（同步）。

        Args:
            thread_id (str, optional): 对话线程 ID（服务端用它从 checkpointer/state 中取回消息）
        """
        request = ChatHistoryInput(thread_id=thread_id)
        try:
            response = httpx.post(
                f"{self.base_url}/history",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

        return ChatHistory.model_validate(response.json())
