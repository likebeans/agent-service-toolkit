"""
本文件在整个项目中的角色：Streamlit 前端应用（Web UI）。

它解决的核心问题是什么？
- 给用户提供一个“可交互的聊天界面”，通过 `AgentClient` 调用后端 FastAPI Service：
  - `/info`：读取可用 Agent/模型，渲染侧边栏配置
  - `/{agent_id}/invoke`：非流式聊天
  - `/{agent_id}/stream`：流式聊天（token + 中间消息）
  - `/feedback`：把用户评分与 run_id 关联起来上报（LangSmith）
  - `/history`：按 thread_id 恢复历史消息，实现“分享/续聊”
- 将“UI 复杂度（Streamlit 的状态/容器/重跑）”与“后端协议（SSE、消息结构）”解耦：
  - 后端协议由 `src/service/service.py` 定义
  - 客户端协议由 `src/client/client.py` 封装
  - 本文件只关心如何把这些事件渲染成良好的交互体验

端到端链路（从 UI 到服务端再回来）：
1) 用户输入（文本/语音） -> 组装 `message/thread_id/user_id/model` -> `AgentClient.astream()/ainvoke()`
2) 服务端执行 LangGraph Agent -> SSE 推送 message/token/error -> 客户端解析后 yield
3) `draw_messages()`：
   - token(str)：写入同一个 placeholder，实现“打字机效果”
   - message(ChatMessage)：渲染人类/AI/工具结果/自定义任务状态，并更新 session_state
4) 最后一条 AI 消息展示评分组件 -> `handle_feedback()` 调用 `/feedback`

你在复刻项目时，可以把它当作“一个参考 UI”：重点学习它如何处理：
- Streamlit 的 `session_state` 生命周期
- 线程（thread_id）与用户（user_id）在 URL 中的传递与恢复
- 流式 token 与工具调用的可视化（status/popover/nested status）
"""

import asyncio
import os
import urllib.parse
import uuid
from collections.abc import AsyncGenerator

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from client import AgentClient, AgentClientError
from schema import ChatHistory, ChatMessage
from schema.task_data import TaskData, TaskDataStatus
from voice import VoiceManager

# 一个用于与 LangGraph Agent 交互的 Streamlit 应用（聊天 UI）。
# 核心协作关系：
# - UI 侧：本文件负责渲染与交互（streaming token、工具调用状态、评分等）
# - 客户端：`src/client/client.py` 负责 HTTP/SSE 协议与 schema 解析
# - 服务端：`src/service/service.py` 定义 API 与 SSE 事件格式


APP_TITLE = "Agent Service Toolkit"
APP_ICON = "🧰"
USER_ID_COOKIE = "user_id"


def get_or_create_user_id() -> str:
    """
    获取或创建 user_id（用户标识）。

    设计意图：
    - Streamlit 本质是“会话 + 重跑”的交互模型；
    - 我们希望 user_id 在一次会话内稳定，并且可通过 URL 分享/恢复；
    - user_id 会传给服务端（见 `src/service/service.py:_handle_input()`）用于长期记忆/跨 thread 的用户状态。
    """
    # 优先从 session_state 读取（同一会话内最稳定）
    if USER_ID_COOKIE in st.session_state:
        return st.session_state[USER_ID_COOKIE]

    # 其次从 URL query params 读取（用于分享/恢复；同一 URL 打开能保持 user_id 不变）
    if USER_ID_COOKIE in st.query_params:
        user_id = st.query_params[USER_ID_COOKIE]
        st.session_state[USER_ID_COOKIE] = user_id
        return user_id

    # 都没有则生成新的 user_id（第一次访问）
    user_id = str(uuid.uuid4())

    # 保存到 session_state（对本次会话生效）
    st.session_state[USER_ID_COOKIE] = user_id

    # 同时写入 URL（便于 bookmark/share）
    st.query_params[USER_ID_COOKIE] = user_id

    return user_id


async def main() -> None:
    """
    Streamlit 应用主入口（异步）。

    高层流程：
    1) 初始化页面配置与 UI 外观
    2) 初始化/复用 `AgentClient`（连接后端服务）
    3) 初始化/复用 `VoiceManager`（可选语音能力）
    4) 初始化 thread_id，并可从服务端拉取历史（/history）
    5) 渲染 sidebar 配置与对话历史
    6) 读取用户输入并触发 invoke/stream
    7) 渲染反馈组件并上报到 /feedback
    """
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
    )

    # 隐藏 Streamlit 右上角状态/工具栏区域，让界面更“产品化”。
    st.html(
        """
        <style>
        [data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
        </style>
        """,
    )
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    # 获取/创建 user_id：用于跨 thread 的用户身份。
    user_id = get_or_create_user_id()

    if "agent_client" not in st.session_state:
        load_dotenv()
        agent_url = os.getenv("AGENT_URL")
        if not agent_url:
            host = os.getenv("HOST", "0.0.0.0")
            port = os.getenv("PORT", 8080)
            agent_url = f"http://{host}:{port}"
        try:
            with st.spinner("Connecting to agent service..."):
                st.session_state.agent_client = AgentClient(base_url=agent_url)
        except AgentClientError as e:
            st.error(f"Error connecting to agent service at {agent_url}: {e}")
            st.markdown("The service might be booting up. Try again in a few seconds.")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client

    # 初始化语音模块（每个 session 一次；未配置会返回 None）
    if "voice_manager" not in st.session_state:
        st.session_state.voice_manager = VoiceManager.from_env()
    voice = st.session_state.voice_manager

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = str(uuid.uuid4())
            messages = []
        else:
            try:
                messages: ChatHistory = agent_client.get_history(thread_id=thread_id).messages
            except AgentClientError:
                st.error("No message history found for this Thread ID.")
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    # Sidebar：配置项（模型、agent、是否流式、是否启用音频、分享/架构等）
    with st.sidebar:
        st.header(f"{APP_ICON} {APP_TITLE}")

        ""
        "Full toolkit for running an AI agent service built with LangGraph, FastAPI and Streamlit"
        ""

        if st.button(":material/chat: New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            # 新对话：清理上一条消息缓存的音频，避免“新对话播放旧音频”
            if "last_audio" in st.session_state:
                del st.session_state.last_audio
            st.rerun()

        with st.popover(":material/settings: Settings", use_container_width=True):
            model_idx = agent_client.info.models.index(agent_client.info.default_model)
            model = st.selectbox("LLM to use", options=agent_client.info.models, index=model_idx)
            agent_list = [a.key for a in agent_client.info.agents]
            agent_idx = agent_list.index(agent_client.info.default_agent)
            agent_client.agent = st.selectbox(
                "Agent to use",
                options=agent_list,
                index=agent_idx,
            )
            use_streaming = st.toggle("Stream results", value=True)
            # 音频开关：关闭时清理缓存音频，避免 UI 误显示
            enable_audio = st.toggle(
                "Enable audio generation",
                value=True,
                disabled=not voice or not voice.tts,
                help="Configure VOICE_TTS_PROVIDER in .env to enable"
                if not voice or not voice.tts
                else None,
                on_change=lambda: st.session_state.pop("last_audio", None)
                if not st.session_state.get("enable_audio", True)
                else None,
                key="enable_audio",
            )

            # 展示 user_id（便于调试/理解“分享链接为什么能保持用户身份”）
            st.text_input("User ID (read-only)", value=user_id, disabled=True)

        @st.dialog("Architecture")
        def architecture_dialog() -> None:
            st.image(
                "https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png?raw=true"
            )
            "[View full size on Github](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png)"
            st.caption(
                "App hosted on [Streamlit Cloud](https://share.streamlit.io/) with FastAPI service running in [Azure](https://learn.microsoft.com/en-us/azure/app-service/)"
            )

        if st.button(":material/schema: Architecture", use_container_width=True):
            architecture_dialog()

        with st.popover(":material/policy: Privacy", use_container_width=True):
            st.write(
                "Prompts, responses and feedback in this app are anonymously recorded and saved to LangSmith for product evaluation and improvement purposes only."
            )

        @st.dialog("Share/resume chat")
        def share_chat_dialog() -> None:
            session = st.runtime.get_instance()._session_mgr.list_active_sessions()[0]
            st_base_url = urllib.parse.urlunparse(
                [session.client.request.protocol, session.client.request.host, "", "", "", ""]
            )
            # 如果不是 localhost，默认切换到 https（更符合真实部署环境）
            if not st_base_url.startswith("https") and "localhost" not in st_base_url:
                st_base_url = st_base_url.replace("http", "https")
            # 分享链接同时包含 thread_id 与 user_id：既能恢复对话线程，也能保持用户身份（长期记忆）
            chat_url = (
                f"{st_base_url}?thread_id={st.session_state.thread_id}&{USER_ID_COOKIE}={user_id}"
            )
            st.markdown(f"**Chat URL:**\n```text\n{chat_url}\n```")
            st.info("Copy the above URL to share or revisit this chat")

        if st.button(":material/upload: Share/resume chat", use_container_width=True):
            share_chat_dialog()

        "[View the source code](https://github.com/JoshuaC215/agent-service-toolkit)"
        st.caption(
            "Made with :material/favorite: by [Joshua](https://www.linkedin.com/in/joshua-k-carroll/) in Oakland"
        )

    # 渲染已有消息（可能来自本地 session_state，也可能来自 /history 恢复）
    messages: list[ChatMessage] = st.session_state.messages

    if len(messages) == 0:
        match agent_client.agent:
            case "chatbot":
                WELCOME = "Hello! I'm a simple chatbot. Ask me anything!"
            case "interrupt-agent":
                WELCOME = "Hello! I'm an interrupt agent. Tell me your birthday and I will predict your personality!"
            case "research-assistant":
                WELCOME = "Hello! I'm an AI-powered research assistant with web search and a calculator. Ask me anything!"
            case "rag-assistant":
                WELCOME = """Hello! I'm an AI-powered Company Policy & HR assistant with access to AcmeTech's Employee Handbook.
                I can help you find information about benefits, remote work, time-off policies, company values, and more. Ask me anything!"""
            case _:
                WELCOME = "Hello! I'm an AI agent. Ask me anything!"

        with st.chat_message("ai"):
            st.write(WELCOME)

    # draw_messages() 需要一个 async iterator；这里把已存在的消息列表包装成 async generator。
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # 重新渲染上一条 AI 消息缓存的音频（如果存在）：
    # Streamlit rerun 会重建 UI 树，因此需要显式把音频再挂回容器里。
    if (
        voice
        and enable_audio
        and "last_audio" in st.session_state
        and st.session_state.last_message
        and len(messages) > 0
        and messages[-1].type == "ai"
    ):
        with st.session_state.last_message:
            audio_data = st.session_state.last_audio
            st.audio(audio_data["data"], format=audio_data["format"])

    # 获取用户新输入：
    # - 如果启用了 VoiceManager：支持语音输入并转写
    # - 否则退化为普通文本输入
    # 要启用语音功能（在“Streamlit App 的 .env”，而不是 service 的 .env）配置：
    # - VOICE_STT_PROVIDER / VOICE_TTS_PROVIDER / OPENAI_API_KEY
    if voice:
        user_input = voice.get_chat_input()
    else:
        user_input = st.chat_input()

    if user_input:
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)
        try:
            if use_streaming:
                stream = agent_client.astream(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                )
                await draw_messages(stream, is_new=True)
                # 流式模式下：文本已经由 draw_messages() 边收 token 边渲染。
                # 如果启用了 TTS，则在流结束后为“最终 AI 消息”再生成一次整段音频并渲染到同一容器里。
                # 注意：draw_messages() 会把最终消息写入 st.session_state.messages，
                # 并把最后一个 AI 容器引用存到 st.session_state.last_message。
                if voice and enable_audio and st.session_state.messages:
                    last_msg = st.session_state.messages[-1]
                    # 只为 AI 且有内容的消息生成音频
                    if last_msg.type == "ai" and last_msg.content:
                        # audio_only=True：避免重复渲染文本（文本已在流式过程中显示）
                        voice.render_message(
                            last_msg.content,
                            container=st.session_state.last_message,
                            audio_only=True,
                        )
            else:
                response = await agent_client.ainvoke(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                )
                messages.append(response)
                # 非流式模式：一次性渲染 AI 回复（可选语音）
                with st.chat_message("ai"):
                    if voice and enable_audio:
                        voice.render_message(response.content)
                    else:
                        st.write(response.content)
            st.rerun()  # 清理旧容器引用，避免后续渲染混淆
        except AgentClientError as e:
            st.error(f"Error generating response: {e}")
            st.stop()

    # 如果已产生消息，则在最后一条 AI 消息下方展示评分组件
    if len(messages) > 0 and st.session_state.last_message:
        with st.session_state.last_message:
            await handle_feedback()


async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
) -> None:
    """
    渲染一组聊天消息（回放历史 or 渲染新流）。

    这是整个 UI 的“渲染中枢”，它要同时处理两类事件：
    - token（str）：来自 `/stream` 的 token 片段，用 placeholder 实现增量渲染（打字机效果）
    - message（ChatMessage）：中间/最终消息，包括 human/ai/tool/custom

    关键逻辑：
    - 对 token：把连续 token 拼成 streaming_content，并写入同一个 `st.empty()` 容器
    - 对 tool_calls：为每个 tool call 创建 `st.status(...)`，并在收到 ToolMessage 后更新对应 status
    - 对 sub-agent（transfer_to/transfer_back_to）：进入递归处理，把子 Agent 的输出塞进嵌套 status 容器
    - 通过 `st.session_state.last_message` 保存“最后一个 AI 消息容器”，方便后续挂载反馈组件/语音播放器

    Args:
        messages_agen: 一个 async generator，逐个 yield ChatMessage 或 token(str)
        is_new: True 表示这是“新产生的消息流”，需要写入 st.session_state.messages；False 表示回放历史
    """

    # 追踪“最后一个消息容器”（Streamlit 的 chat_message/status 都依赖容器上下文）
    last_message_type = None
    st.session_state.last_message = None

    # token 流式渲染的占位符与累积缓冲
    streaming_content = ""
    streaming_placeholder = None

    # 不断从 async generator 读取事件并渲染
    while msg := await anext(messages_agen, None):
        # str 事件表示 token（流式输出片段）
        if isinstance(msg, str):
            # 第一个 token 到来时，创建一个新的 AI chat_message 容器，并在其中放一个 placeholder。
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue
        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()

        match msg.type:
            # human：最简单，直接渲染
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            # ai：最复杂，需要同时处理“流式 token placeholder”与“工具调用状态”
            case "ai":
                # 渲染新消息时，把消息落到 session_state 里（便于 rerun 后回放/反馈）
                if is_new:
                    st.session_state.messages.append(msg)

                # 确保当前处在 AI chat_message 容器中（不同消息类型之间需要切换容器）
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

                with st.session_state.last_message:
                    # 如果本条 AI message 有完整 content，则输出它。
                    # 同时如果之前在 streaming（有 placeholder），要在这里“收口”并重置 streaming 状态。
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    if msg.tool_calls:
                        # 对每个 tool call 创建一个 status 容器，并用 tool_call_id 做映射，
                        # 这样后续 ToolMessage 到来时能更新到正确的 status 上。
                        call_results = {}
                        for tool_call in msg.tool_calls:
                            # transfer_to 表示“把控制权交给子 Agent”，在 UI 上用不同 label 区分
                            if "transfer_to" in tool_call["name"]:
                                label = f"""💼 Sub Agent: {tool_call["name"]}"""
                            else:
                                label = f"""🛠️ Tool Call: {tool_call["name"]}"""

                            status = st.status(
                                label,
                                state="running" if is_new else "complete",
                            )
                            call_results[tool_call["id"]] = status

                        # 约定：每个 tool_call 后面应该对应一个 ToolMessage（工具输出）。
                        for tool_call in msg.tool_calls:
                            if "transfer_to" in tool_call["name"]:
                                status = call_results[tool_call["id"]]
                                status.update(expanded=True)
                                await handle_sub_agent_msgs(messages_agen, status, is_new)
                                break

                            # 普通工具调用：渲染 input，并等待下一条 tool message 作为 output
                            status = call_results[tool_call["id"]]
                            status.write("Input:")
                            status.write(tool_call["args"])
                            tool_result: ChatMessage = await anext(messages_agen)

                            if tool_result.type != "tool":
                                st.error(f"Unexpected ChatMessage type: {tool_result.type}")
                                st.write(tool_result)
                                st.stop()

                            # 记录新消息，并把 output 写回到对应的 status 容器中
                            if is_new:
                                st.session_state.messages.append(tool_result)
                            if tool_result.tool_call_id:
                                status = call_results[tool_result.tool_call_id]
                            status.write("Output:")
                            status.write(tool_result.content)
                            status.update(state="complete")

            case "custom":
                # custom：用于承载结构化的 UI 事件（例如 bg-task-agent 的任务状态更新）。
                # 参考：
                # - `src/agents/utils.py` CustomData
                # - `src/agents/bg_task_agent/task.py`
                try:
                    task_data: TaskData = TaskData.model_validate(msg.custom_data)
                except ValidationError:
                    st.error("Unexpected CustomData message received from agent")
                    st.write(msg.custom_data)
                    st.stop()

                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "task":
                    last_message_type = "task"
                    st.session_state.last_message = st.chat_message(
                        name="task", avatar=":material/manufacturing:"
                    )
                    with st.session_state.last_message:
                        status = TaskDataStatus()

                status.add_and_draw_task_data(task_data)

            # 兜底：未知类型直接报错并停止（避免 UI 进入不一致状态）
            case _:
                st.error(f"Unexpected ChatMessage type: {msg.type}")
                st.write(msg)
                st.stop()


async def handle_feedback() -> None:
    """渲染评分组件并上报用户反馈（关联 run_id）。"""

    # 记录上一次发送的反馈，避免重复上报（Streamlit rerun 可能触发重复执行）
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    latest_run_id = st.session_state.messages[-1].run_id
    feedback = st.feedback("stars", key=latest_run_id)

    # 如果评分或 run_id 发生变化，则发送新的反馈记录
    if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
        # Streamlit stars 返回的是 index（0..4），这里归一化到 0..1 以适配 LangSmith
        normalized_score = (feedback + 1) / 5.0

        agent_client: AgentClient = st.session_state.agent_client
        try:
            await agent_client.acreate_feedback(
                run_id=latest_run_id,
                key="human-feedback-stars",
                score=normalized_score,
                kwargs={"comment": "In-line human feedback"},
            )
        except AgentClientError as e:
            st.error(f"Error recording feedback: {e}")
            st.stop()
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("Feedback recorded", icon=":material/reviews:")


async def handle_sub_agent_msgs(messages_agen, status, is_new):
    """
    处理“子 Agent”输出：把子 Agent 的过程消息收敛到一个 status 容器中。

    背景：
    - 使用 langgraph-supervisor/hierarchy 时，主 Agent 会通过 tool call `transfer_to_*` 把控制权交给子 Agent；
    - 子 Agent 执行期间会产生一系列 ai/tool 消息；
    - 最终通过 `transfer_back_to_*` 再把控制权交回主 Agent。

    本函数负责：
    - 从 generator 中消费子 Agent 的消息序列
    - 将工具调用以 popover 的形式展示 input/output
    - 支持嵌套：子 Agent 内部再 transfer_to 其它子 Agent 时递归处理

    Args:
        messages_agen: 消息/事件的 async generator（与 draw_messages 共用同一条流）
        status: 当前子 Agent 的 status 容器
        is_new: 是否为新消息（决定是否写入 session_state）
    """
    nested_popovers = {}

    # 第一个消息通常是 transfer_to tool call 的“success”回执/中间消息，先消费掉
    first_msg = await anext(messages_agen)
    if is_new:
        st.session_state.messages.append(first_msg)

    # 一直读到出现明确的 transfer_back_to（表示控制权归还）
    while True:
        # 读取下一条消息
        sub_msg = await anext(messages_agen)

        # 只有当服务端取消 skip_stream 过滤时，才可能在这里收到 token(str)；目前默认不会发生。
        # if isinstance(sub_msg, str):
        #     continue

        if is_new:
            st.session_state.messages.append(sub_msg)

        # 处理工具输出：如果之前为 tool_call 创建了 popover，这里把 output 填进去
        if sub_msg.type == "tool" and sub_msg.tool_call_id in nested_popovers:
            popover = nested_popovers[sub_msg.tool_call_id]
            popover.write("**Output:**")
            popover.write(sub_msg.content)
            continue

        # 处理 transfer_back_to：子 Agent 结束并归还控制权
        if (
            hasattr(sub_msg, "tool_calls")
            and sub_msg.tool_calls
            and any("transfer_back_to" in tc.get("name", "") for tc in sub_msg.tool_calls)
        ):
            # 处理 transfer_back_to tool call，并消费对应的 tool result
            for tc in sub_msg.tool_calls:
                if "transfer_back_to" in tc.get("name", ""):
                    # 读取对应的 tool result
                    transfer_result = await anext(messages_agen)
                    if is_new:
                        st.session_state.messages.append(transfer_result)

            # 完成：更新 status 并退出循环
            if status:
                status.update(state="complete")
            break

        # 将子 Agent 的内容与工具调用展示在同一个嵌套 status 中
        if status:
            if sub_msg.content:
                status.write(sub_msg.content)

            if hasattr(sub_msg, "tool_calls") and sub_msg.tool_calls:
                for tc in sub_msg.tool_calls:
                    # 如果是嵌套 transfer_to，则创建更深一层的 status 并递归处理
                    if "transfer_to" in tc["name"]:
                        # 为子 Agent 创建嵌套 status 容器
                        nested_status = status.status(
                            f"""💼 Sub Agent: {tc["name"]}""",
                            state="running" if is_new else "complete",
                            expanded=True,
                        )

                        # 递归处理子 Agent 的子 Agent
                        await handle_sub_agent_msgs(messages_agen, nested_status, is_new)
                    else:
                        # 普通工具调用：用 popover 展示 input，并在收到 tool result 后补上 output
                        popover = status.popover(f"{tc['name']}", icon="🛠️")
                        popover.write(f"**Tool:** {tc['name']}")
                        popover.write("**Input:**")
                        popover.write(tc["args"])
                        # 用 tool_call_id 关联 popover，后续 tool message 到来时可回填 output
                        nested_popovers[tc["id"]] = popover


if __name__ == "__main__":
    asyncio.run(main())
