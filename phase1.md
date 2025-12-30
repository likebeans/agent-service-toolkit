# 《agent-service-toolkit》项目讲解文档（第一阶段：系统全景认知）

> 目标：让你在脑中形成一个“可复刻的系统模型”。本阶段只讲系统是什么、怎么跑、模块如何协作、扩展点在哪里；不深入到函数级实现细节，但会引用关键文件/模块/入口函数作为“锚点”。

---

## 1. 项目定位与设计目标（它为什么存在）

### A. 业务运行逻辑（系统在“做什么”）
`agent-service-toolkit`把“一个 LangGraph Agent”产品化成“可部署的在线服务”，并提供完整的端到端交付链：
- 后端：把多个 Agent 作为 HTTP API 暴露（支持流式/非流式）。
- 前端：提供一个可直接使用的聊天 UI（Streamlit），包含多轮对话、模型选择、Agent 选择、语音输入输出、反馈收集。
- 工程：提供可扩展的 Agent/Tool/Provider 接入模板，外加 Docker/测试/配置体系。

你可以把它理解为：**Agent-as-a-Service 的参考实现**，重点不是“某一个很强的 Agent”，而是“把 Agent 可靠地上线、调用、观测、扩展”的一整套工程骨架。

### B. 架构与模块逻辑（组件如何协作）
四段式链路（也是仓库 README 的核心叙事）：
1) LangGraph Agent（`src/agents/`）  
2) FastAPI 服务（`src/service/service.py`）  
3) 客户端 SDK（`src/client/client.py`）  
4) Streamlit Chat App（`src/streamlit_app.py`）

并由统一的配置与协议模型贯穿：
- 配置：`src/core/settings.py`
- LLM Provider 选择：`src/core/llm.py`
- API 协议/数据结构：`src/schema/`

### C. 代码级逻辑（关键抽象如何实现）
最重要的“入口锚点”：
- 服务启动：`src/run_service.py` → `service:app`（`src/service/__init__.py` 导出 `app`）
- 服务 API：`src/service/service.py`（`/info`、`/invoke`、`/stream`、`/feedback`、`/history`）
- Agent 注册与加载：`src/agents/agents.py`（`agents` registry、`get_agent()`、`load_agent()`、`DEFAULT_AGENT`）
- 模型获取：`src/core/llm.py:get_model()`（多 Provider）
- 配置加载：`src/core/settings.py:Settings`（pydantic-settings，从 `.env` 注入）

---

## 2. 核心概念解释（Agent / Tool / Provider / Service）

下面每个概念都按 A/B/C 三层解释。

### 2.1 Agent 是什么
**A. 业务运行逻辑**  
Agent 是“可对话的任务执行体”：接收用户消息、决定是否调用工具、产出最终回答；部分 Agent 还支持“中断→人类补充→继续执行”。

**B. 架构与模块逻辑**  
在本项目里，Agent 的本体是 **LangGraph 图**（可编排的节点/边/状态机），运行时以 `AgentGraph` 的形式被服务调用：
- `src/agents/agents.py` 里定义类型别名：
  - `AgentGraph = CompiledStateGraph | Pregel`
- 不同 Agent 采用不同 LangGraph 模式：
  - `@entrypoint` 函数风格（如 `src/agents/chatbot.py`）返回 `Pregel`
  - `StateGraph(...).compile()` 风格（如 `src/agents/research_assistant.py`）返回 `CompiledStateGraph`

**C. 代码级逻辑**  
- Agent 统一通过注册表暴露：`src/agents/agents.py:agents`（dict）
- 服务调用入口：
  - 非流式：`src/service/service.py:invoke()` → `agent.ainvoke(...)`
  - 流式：`src/service/service.py:stream()` → `agent.astream(...)`
- 支持“异步初始化”的 Agent：`src/agents/lazy_agent.py:LazyLoadingAgent` + `src/service/service.py:lifespan()` 在启动期 `await load_agent(...)`（典型例子：MCP Agent）

### 2.2 Tool 是什么
**A. 业务运行逻辑**  
Tool 是 Agent 的“外部能力插件”：搜索、计算、查数据库、调用第三方系统、做 repo 操作等；模型通过 tool call 让系统执行真实动作，再把结果回到模型继续推理。

**B. 架构与模块逻辑**  
项目里有三类 Tool 形态：
1) LangChain 内置/社区工具：如搜索、天气  
   - 例：`src/agents/research_assistant.py` 使用 `DuckDuckGoSearchResults`、`OpenWeatherMapQueryRun`
2) 自定义本地工具：用 `langchain_core.tools.tool` 包装成 `BaseTool`  
   - 例：`src/agents/tools.py:calculator`、`database_search`
3) 外部 Tool Server（MCP）：运行时从 MCP server 动态拉取 tools 列表  
   - 例：`src/agents/github_mcp_agent/github_mcp_agent.py` 使用 `langchain_mcp_adapters`

LangGraph 里通常用 `ToolNode(tools)` 来执行工具：见 `src/agents/research_assistant.py`、`src/agents/rag_assistant.py`。

**C. 代码级逻辑**  
- 自定义工具集中在：`src/agents/tools.py`
- 工具执行节点：`langgraph.prebuilt.ToolNode`（在各 Agent 文件里被挂到图中）
- UI 对工具调用的呈现：`src/streamlit_app.py:draw_messages()` 会把 `ChatMessage.tool_calls` 渲染成状态块（避免用户“看不懂模型正在干什么”）

### 2.3 Provider 是什么
**A. 业务运行逻辑**  
Provider 是“模型供应商/调用方式”的抽象：OpenAI、Anthropic、Google、Groq、AWS Bedrock、Ollama、本地兼容端点、Fake 模型等；系统要能在不改业务逻辑的情况下切换底层模型。

**B. 架构与模块逻辑**  
Provider 能力由两块拼起来：
- “可选模型枚举/协议层”：`src/schema/models.py`（`Provider`、各 ModelName 枚举、`AllModelEnum`）
- “具体模型构造与参数”：`src/core/llm.py:get_model()`（返回 LangChain ChatModel）

配置与可用性判定在 `src/core/settings.py`：
- `Settings.model_post_init()` 会根据环境变量是否提供 key 来决定 `AVAILABLE_MODELS` 与 `DEFAULT_MODEL`
- 强制要求：至少一个 LLM key 存在（否则启动报错）

**C. 代码级逻辑**  
- 模型选择入口：`src/core/llm.py:get_model(model_name)`
- 配置入口：`src/core/settings.py:settings`（全局单例）
- `/info` 暴露可用模型列表：`src/service/service.py:info()`（来自 `settings.AVAILABLE_MODELS`）

### 2.4 Service 在系统中承担什么角色
**A. 业务运行逻辑**  
Service 是“生产化入口”：负责鉴权、线程/用户隔离、统一协议、把 Agent 的执行过程（含中间事件）以 API 方式输出给客户端/前端。

**B. 架构与模块逻辑**  
FastAPI 服务主要承担：
- 多 Agent 路由：`/{agent_id}/invoke`、`/{agent_id}/stream`（默认走 `DEFAULT_AGENT`）
- 流式输出：SSE（Server-Sent Events），同时支持 token 级与 message 级事件
- 运行期初始化：数据库 checkpointer（短期记忆）+ store（长期记忆）+ 异步 Agent 加载
- 反馈上报：对 LangSmith 的 feedback 写入做代理，避免把凭证下放到客户端

**C. 代码级逻辑**  
核心都在 `src/service/service.py`：
- 生命周期初始化：`lifespan()`  
- 认证：`verify_bearer()`（`AUTH_SECRET` 开关）  
- 请求处理：`_handle_input()`（生成 `thread_id`/`user_id`/`run_id`，并处理 interrupt resume）  
- 非流式：`invoke()`  
- 流式：`message_generator()` + `stream()`  
- 反馈：`feedback()`  
- 历史：`history()`  

---

## 3. 整体架构图（文字版组件关系）

```
[Streamlit UI]  src/streamlit_app.py
     |
     | (HTTP via AgentClient)
     v
[AgentClient SDK]  src/client/client.py
     |
     |  /info /invoke /stream /feedback /history
     v
[FastAPI Service]  src/service/service.py
     |
     |  get_agent(agent_id) + RunnableConfig(thread_id,user_id,model,...)
     v
[LangGraph Agent Graphs]  src/agents/*
     |
     |  (optional) ToolNode(tools) / interrupt() / custom stream events
     v
[Tools + Providers]
  - LLM Providers: src/core/llm.py
  - Local tools: src/agents/tools.py
  - MCP tools: src/agents/github_mcp_agent/*
     |
     v
[Persistence]
  - Checkpointer: src/memory/*  (SQLite/Postgres/Mongo)
  - Long-term Store: src/memory/postgres.py or in-memory fallback
```

---

## 4. 系统运行时视角：端到端链路（至少 3 条）

> 你后面复刻项目时，本质上就是把这几条链路逐条实现出来。

### 链路 1：服务启动链路（配置加载 → 依赖初始化 → Agent 装配）
**A. 业务运行逻辑**  
服务启动时完成“可服务化”的准备：确定可用模型、初始化记忆存储、加载所有 Agent（含需要异步初始化的 Agent），确保后续请求能直接命中。

**B. 架构与模块逻辑**  
- 配置从 `.env` 读取：`src/run_service.py` 调用 `dotenv.load_dotenv()`，`src/core/settings.py` 自动 `find_dotenv()`
- FastAPI lifespan 负责资源初始化：`src/service/service.py:lifespan()`
- memory 初始化分为两件事：
  - checkpointer（短期、thread 级）：`src/memory/initialize_database()`
  - store（长期、user 级）：`src/memory/initialize_store()`

**C. 代码级逻辑（关键文件/函数）**
1) `src/run_service.py` 启动 uvicorn → `"service:app"`  
2) `src/service/service.py:app = FastAPI(lifespan=lifespan, ...)`  
3) `lifespan()` 内部：
   - `async with initialize_database() as saver, initialize_store() as store: ...`
   - `await load_agent(agent_id)`（支持 `LazyLoadingAgent`）
   - `agent = get_agent(agent_id)` 后统一注入：
     - `agent.checkpointer = saver`
     - `agent.store = store`
4) 结果：每个 Agent graph 都具备“线程记忆 + 长期记忆”能力（是否真正持久化取决于数据库类型）。

### 链路 2：非流式请求 `/invoke`（HTTP → Agent 执行 → 最终消息返回）
**A. 业务运行逻辑**  
客户端发送一条用户消息，服务返回最终 AI 回复（或 interrupt 提示）。

**B. 架构与模块逻辑**  
- API 协议：`src/schema/schema.py:UserInput`（message/model/thread_id/user_id/agent_config）
- 服务路由：`src/service/service.py:invoke()` 支持默认 agent 或 path 指定 agent
- 多轮对话靠 `thread_id`；跨线程长期记忆靠 `user_id`

**C. 代码级逻辑（关键文件/函数）**
1) `POST /invoke` 或 `POST /{agent_id}/invoke` → `src/service/service.py:invoke()`  
2) `invoke()`：
   - `agent = get_agent(agent_id)`（`src/agents/agents.py`）
   - `kwargs, run_id = await _handle_input(user_input, agent)`
3) `_handle_input()` 的关键动作：
   - 生成/补全 `thread_id`、`user_id`、`run_id`
   - 组装 `RunnableConfig(configurable={thread_id,user_id,model?,...agent_config}, run_id=...)`
   - `state = await agent.aget_state(config)`，如果发现未完成 interrupt，则把输入包装成 `Command(resume=...)`（否则正常 HumanMessage）
4) `agent.ainvoke(..., stream_mode=["updates","values"])` 取最后一个事件：
   - 正常完成：取 `values["messages"][-1]`
   - interrupt：取 `updates["__interrupt__"][0].value`
5) 统一转成协议 `ChatMessage`：`src/service/utils.py:langchain_to_chat_message()` 并回填 `run_id`。

### 链路 3：流式请求 `/stream`（SSE：token + message 双通道流）
**A. 业务运行逻辑**  
UI/客户端不仅要最终答案，还要“过程可见”：  
- 逐 token 输出（更像 ChatGPT）
- 同时还能看到中间步骤（工具调用、子 Agent 交接、后台任务、interrupt 等）

**B. 架构与模块逻辑**  
- 协议输入：`src/schema/schema.py:StreamInput`（继承 UserInput，增加 `stream_tokens`）
- 输出是 SSE 文本流，事件类型主要三类：
  - `type=token`：字符串 token（可关）
  - `type=message`：结构化 `ChatMessage`（包含 tool_calls/custom/task 等）
  - `type=error`：错误提示
- 服务通过 LangGraph 的多 stream_mode 组合实现：
  - `updates`：节点更新（拿到中间消息/interrupt）
  - `messages`：LLM chunk（拿 token）
  - `custom`：Agent 自定义事件（后台任务/进度等）

**C. 代码级逻辑（关键文件/函数）**
1) `POST /stream` → `src/service/service.py:stream()` → `StreamingResponse(message_generator(...))`
2) `message_generator()`：
   - `agent.astream(..., stream_mode=["updates","messages","custom"], subgraphs=True)`
   - 处理两种 event 结构（是否含 node_path）
   - 把 LangChain message 转成 `ChatMessage`（`service.utils.langchain_to_chat_message`）
   - 额外处理：
     - LangGraph 可能把 message 分片成 `(field, value)` tuple：用 `_create_ai_message()` 聚合
     - `remove_tool_calls()` 过滤某些 provider 的工具调用流片段
     - special-case：对 `langgraph-supervisor` 的子图消息做“降噪”（见 `src/service/service.py` 对 `"supervisor"`/`"sub-agent"` 节点的处理逻辑）
   - 最终 yield SSE：`data: {"type": "...", "content": ...}\n\n`
3) UI 消费：
   - `src/streamlit_app.py` 用 `AgentClient.astream()`（`src/client/client.py`）解析 SSE
   - `draw_messages()` 把 token 用 placeholder 逐步渲染，把 `tool_calls`/`custom task` 用 status 容器呈现
4) 可选闭环：
   - UI 星级反馈 → `AgentClient.acreate_feedback()` → `POST /feedback`
   - 服务端代写 LangSmith：`src/service/service.py:feedback()`（`LangsmithClient.create_feedback`）

---

## 5. 模块划分与职责边界（谁负责什么/不负责什么）

### A. 业务运行逻辑
你可以把系统拆成“产品交付层次”：
- Agent（能力）  
- Service（在线化）  
- Client/UI（交互）  
- Observability（反馈/追踪）  
- Persistence（记忆/状态）

### B. 架构与模块逻辑（责任边界）
- `src/agents/`：只关心“如何完成任务”（图怎么编排、工具怎么用、何时 interrupt）。  
  不负责 HTTP、不负责 UI。
- `src/service/`：只关心“如何把 Agent 变成 API”（鉴权、协议、SSE、初始化资源）。  
  不负责 Agent 的业务逻辑细节。
- `src/client/`：服务调用 SDK（同步/异步、流式/非流式）。  
  不负责图编排、不负责数据库。
- `src/streamlit_app.py`：交互层（会话状态、渲染、反馈、可选语音）。  
  不负责服务端鉴权策略、不负责 agent 内部图实现。
- `src/core/`：全局配置与模型构造（Provider 选择）。  
  不负责 HTTP 路由、不负责 UI。
- `src/schema/`：协议/类型（输入输出结构、模型枚举）。  
  不负责运行时逻辑。
- `src/memory/`：持久化与记忆基础设施（checkpointer/store）。  
  不负责业务图节点。

### C. 代码级逻辑（依赖关系建议）
一个健康的依赖方向（本仓库也基本如此）：
- `service` → 依赖 `agents/core/schema/memory`
- `agents` → 依赖 `core/schema`（以及 LangGraph/LangChain）
- `client` → 依赖 `schema`
- `streamlit_app` → 依赖 `client/schema/voice`
- `memory` → 依赖 `core.settings`

---

## 6. 代码目录结构说明（每个目录的工程意义）

- `src/agents/`：Agent 图实现与样例集合（展示 LangGraph 的多种特性）
  - `agents.py`：注册表与默认 agent（多 agent 支持的核心）
  - `research_assistant.py`：默认 agent（工具调用 + 内容安全）
  - `interrupt_agent.py`：human-in-the-loop（`interrupt()` + `Command(resume=...)`）
  - `bg_task_agent/`：自定义流事件（后台任务进度）
  - `github_mcp_agent/`：MCP 工具服务器接入（异步加载）
  - `langgraph_supervisor*_agent.py`：多 agent 监督/分层协作（langgraph-supervisor）
- `src/service/`：FastAPI 服务（协议、SSE、初始化、鉴权、反馈）
- `src/client/`：HTTP 客户端 SDK（强烈建议你复刻时也保留这一层）
- `src/core/`：settings + LLM factory（Provider 层）
- `src/schema/`：pydantic 协议模型、模型枚举
- `src/memory/`：SQLite/Postgres/Mongo checkpointer + store
- `src/voice/`：语音输入输出（刻意与 service 解耦，属于“客户端能力”）
- `scripts/`：离线数据准备（Chroma 建库脚本 `scripts/create_chroma_db.py`）
- `docs/`：对特定扩展（RAG/Ollama/VertexAI/MCP/凭证文件）的操作指南
- `tests/`：单测 + 集成测（覆盖 service、client、agents、voice 等关键路径）
- `compose.yaml` + `docker/`：一键本地/部署形态（FastAPI + Streamlit + Postgres）

---

## 7. 关键设计取舍与优缺点（为什么这样设计）

### 取舍 1：用 LangGraph 做编排，而不是自己写 Orchestrator
- 好处：图编排、状态、stream、interrupt、checkpointer/store 都有现成能力（见 `src/agents/*` 的多样示例）。
- 代价：系统核心抽象受 LangGraph 模型约束；要理解 `stream_mode`、state、message 类型、subgraph 事件结构。

### 取舍 2：SSE 流式同时输出 token 与结构化 message
- 好处：UI 既能“像 ChatGPT 一样打字”，又能呈现工具调用/中间步骤/自定义事件（`src/service/service.py:message_generator()` + `src/streamlit_app.py:draw_messages()`）。
- 代价：协议更复杂；客户端要解析并区分 token/message；服务端需要做事件降噪和 message 聚合（tuple parts → AIMessage）。

### 取舍 3：把“短期记忆”和“长期记忆”拆成两套
- 短期：checkpointer（thread 级对话状态）  
- 长期：store（user 级跨线程数据）  
对应实现：`src/memory/__init__.py` + `src/service/service.py:lifespan()` 的双注入
- 好处：符合真实产品需求（对话线程 vs 用户画像/偏好/事实）。
- 代价：SQLite 模式下长期 store 退化为内存（`src/memory/sqlite.py`），需要 Postgres 才真正持久化。

### 取舍 4：多 Provider 用“枚举 + factory”，并由 settings 决定可用模型
- 好处：/info 能准确告诉客户端“当前可用模型集”；避免客户端传一个不存在的模型。
- 代价：每加一个新 Provider，通常要同时改 `src/schema/models.py`、`src/core/settings.py`、`src/core/llm.py`。

### 取舍 5：支持需要异步初始化的 Agent（LazyLoadingAgent）
- 好处：能接入 MCP 这类“启动期要拉工具/建连接”的场景（`src/agents/github_mcp_agent/github_mcp_agent.py`）。
- 代价：服务启动链路更复杂；要考虑“部分 agent 加载失败也继续启动”（`lifespan()` 里是容错策略）。

---

## 8. 适合与不适合的使用场景

### 适合
- 你要做一个“可上线的 Agent 服务骨架”，并希望：
  - 多 agent、多模型可选
  - 流式输出 + 过程可见
  - 具备基本记忆、反馈、可观测性
  - 快速替换成你自己的 agent/tool/provider
- 团队里需要一个参考实现来统一工程形态（API、schema、client、UI、Docker、测试）。

### 不适合
- 需要极致吞吐/超低延迟的推理网关（本项目更多是模板与教学示例）。
- 强隔离多租户/复杂权限模型/审计合规（这里只有简单 bearer）。
- 需要强一致、跨区域的长期记忆系统（这里的 store/checkpointer 是“够用/示范级”）。
- 想完全脱离 LangGraph/LangChain 生态（本项目深度绑定它们的抽象）。

---

## 9. 项目中最重要的 10 个文件/目录（为什么重要 + 推荐学习顺序）

1) `src/service/service.py`  
   - 这是“服务化核心”：生命周期、路由、SSE、interrupt、反馈、历史全在这。
2) `src/agents/agents.py`  
   - 多 agent 支持的中枢：注册表、默认 agent、lazy loading 入口。
3) `src/core/settings.py`  
   - 全局配置与可用模型集合的来源；决定系统能不能启动、能用哪些 provider、用什么 DB。
4) `src/core/llm.py`  
   - Provider 的真正落地：不同模型如何被构造、如何开启 streaming。
5) `src/schema/schema.py`  
   - API 协议与前后端契约：UserInput/StreamInput/ChatMessage/ServiceMetadata 等。
6) `src/client/client.py`  
   - 客户端 SDK：如何调用 /invoke /stream、如何解析 SSE，Streamlit 也依赖它。
7) `src/streamlit_app.py`  
   - UI 如何消费 token+message 双通道流、如何展示 tool_calls、如何做反馈与会话管理。
8) `src/memory/`  
   - 记忆基础设施：SQLite/Postgres/Mongo 的 checkpointer，以及 Postgres store（长期记忆）。
9) `src/agents/research_assistant.py`  
   - 默认 agent 的“标准范式”：StateGraph + ToolNode + conditional edges + 内容安全。
10) `src/agents/github_mcp_agent/github_mcp_agent.py`（配合 `docs/GitHub_MCP_Agent.md`）  
   - 最关键的“外部工具生态接入”示例：LazyLoadingAgent + MCP tools 动态加载。

推荐学习顺序（从“系统”到“局部”）：
1) `README.md`（产品形态）  
2) `src/run_service.py` → `src/service/service.py`（服务入口与端到端链路）  
3) `src/schema/`（协议） + `src/client/client.py`（怎么被调用）  
4) `src/agents/agents.py`（多 agent 架构）  
5) `src/core/settings.py` + `src/core/llm.py`（provider 与配置）  
6) `src/agents/research_assistant.py`（典型 agent 图）  
7) `src/memory/`（记忆与持久化）  
8) 再看高级样例：interrupt/bg_task/supervisor/MCP/voice

---

## 10. 复刻路线图（从最小可运行 MVP 开始）

> 你最终要能“不看原仓库代码”复刻一个等价系统。下面是按工程落地顺序拆解的路线图：每一步都有目标、模块/接口、文件、验收方式。

### Step 0：最小骨架（能启动一个 FastAPI）
- 目标：跑起服务进程，暴露 `/health`
- 模块/接口：FastAPI app + `run_service.py`
- 文件：
  - `src/run_service.py`
  - `src/service/service.py`（先只有 `app` + `/health`）
- 验收：`curl http://localhost:8080/health` 返回 ok

### Step 1：协议层（schema 先行）
- 目标：定义前后端契约：输入/输出/元信息
- 模块/接口：`UserInput`、`StreamInput`、`ChatMessage`、`ServiceMetadata`
- 文件：
  - `src/schema/schema.py`
  - `src/schema/models.py`（先最小模型枚举也行）
- 验收：OpenAPI 文档能看到这些模型（FastAPI 自动生成）

### Step 2：配置与模型（Provider 最小闭环）
- 目标：服务能根据 env 选择一个“可用模型”（可先用 Fake）
- 模块/接口：`Settings` + `get_model()`
- 文件：
  - `src/core/settings.py`（至少：HOST/PORT/DEFAULT_MODEL/AVAILABLE_MODELS/USE_FAKE_MODEL）
  - `src/core/llm.py`（至少：Fake 模型）
  - `pyproject.toml`（依赖）
- 验收：服务启动不报“无 API key”；`/info` 能返回 default_model 与 models 列表

### Step 3：Agent 抽象与注册表（先做一个 chatbot）
- 目标：有一个最简单 agent 能对话
- 模块/接口：`AgentGraph`、`agents` registry、`get_agent()`
- 文件：
  - `src/agents/chatbot.py`
  - `src/agents/agents.py`
- 验收：`POST /invoke` 能返回 AI 回复（哪怕是 fake）

### Step 4：服务化调用（/info + /invoke）
- 目标：完成非流式端到端链路
- 模块/接口：`/info`、`/invoke`、多 agent path
- 文件：
  - `src/service/service.py`（实现 `info()`、`invoke()`、`_handle_input()`）
  - `src/service/utils.py`（message → ChatMessage 转换）
- 验收：
  - `/info` 列出 agents/models/defaults
  - `/invoke` 返回 `ChatMessage(type="ai")`

### Step 5：流式（SSE + 双通道 token/message）
- 目标：完成 `/stream` + SSE 协议 + 客户端解析
- 模块/接口：`message_generator()`、`AgentClient.stream/astream`
- 文件：
  - `src/service/service.py`（实现 `stream()`、`message_generator()`）
  - `src/client/client.py`（解析 `data: ...`）
- 验收：
  - `curl -N` 能看到连续 SSE 输出，末尾 `[DONE]`
  - client 能区分 token 与 message

### Step 6：Streamlit UI（把“可用服务”产品化）
- 目标：能像 ChatGPT 一样聊天，并支持选择 agent/model
- 模块/接口：Streamlit 会话状态、渲染 streaming、thread_id/user_id
- 文件：
  - `src/streamlit_app.py`
- 验收：浏览器里聊天可用；刷新/分享链接能恢复 thread

### Step 7：记忆系统（checkpointer + store）
- 目标：多轮对话不丢；可选持久化（Postgres）
- 模块/接口：`initialize_database()`、`initialize_store()`、lifespan 注入
- 文件：
  - `src/memory/__init__.py`
  - `src/memory/sqlite.py`、`src/memory/postgres.py`、`src/memory/mongodb.py`
  - `src/service/service.py`（lifespan 装配）
- 验收：
  - 同 thread_id 多轮对话能续上
  - `/history` 能返回历史（参考 `src/service/service.py:history()`）
  - Postgres 模式下重启服务对话仍可恢复

### Step 8：工具与高级能力（按需逐个加）
- 目标：具备真实 agent 能力与扩展点
- 建议顺序：
  1) ToolNode + 本地工具（`src/agents/tools.py`，如 calculator）
  2) RAG（Chroma 建库：`scripts/create_chroma_db.py` + `src/agents/rag_assistant.py`）
  3) interrupt/human-in-the-loop（`src/agents/interrupt_agent.py` + 服务端 `Command(resume=...)`）
  4) 多 agent supervisor（`src/agents/langgraph_supervisor*_agent.py` + 服务端子图事件处理）
  5) MCP 接入（`src/agents/github_mcp_agent/` + `LazyLoadingAgent`）
  6) 语音（`src/voice/`，注意它是客户端能力）

---

## 11. 扩展系统时应改哪些模块（先给你“地图”）

- 新增 Agent：
  - 新建 `src/agents/<your_agent>.py`
  - 在 `src/agents/agents.py:agents` 注册 key/description/graph
  - 若需要异步初始化：继承 `src/agents/lazy_agent.py:LazyLoadingAgent`，并确保能在 `lifespan()` 期间 `load_agent()` 成功
  - UI 侧可选：`src/streamlit_app.py` 增加欢迎语/交互适配

- 新增 Tool：
  - 本地 tool：放在 `src/agents/tools.py` 或 agent 文件内定义，确保是 LangChain Tool 形态
  - 在 agent 内把 tool 加入 `tools` 列表，并通过 `model.bind_tools(tools)` + `ToolNode(tools)` 接入
  - UI：如果希望展示更友好，可在 `src/streamlit_app.py:draw_messages()` 对特定 tool_calls 做定制渲染

- 集成新 LLM Provider：
  - `src/schema/models.py` 增加模型枚举
  - `src/core/settings.py` 增加对应 API key/配置字段与 `model_post_init()` 逻辑（把模型加入 AVAILABLE_MODELS、设置 DEFAULT_MODEL）
  - `src/core/llm.py:get_model()` 增加构造分支

- 新增 Service/API/Workflow：
  - 协议优先：`src/schema/schema.py` 增加模型
  - 服务端：`src/service/service.py` 增加 endpoint
  - 客户端：`src/client/client.py` 增加调用封装
  - UI：`src/streamlit_app.py` 再决定是否需要新的交互入口

---

## 12. 第一阶段学习检查点问题（请你回答，确认全景模型已经建立）
1) 这个项目把“Agent”产品化成服务时，最关键的 4 个组件分别是什么？它们的文件路径各在哪里？  
2) `/invoke` 与 `/stream` 的核心差异是什么？为什么 `/stream` 需要同时输出 `token` 和 `message` 两类事件？  
3) `thread_id` 与 `user_id` 在架构上分别解决什么问题？它们分别对应哪类“记忆”？  
4) 为什么需要 `LazyLoadingAgent`？`lifespan()` 在启动期做了哪两类基础设施初始化？  
5) 如果你要新增一个 Provider，你认为必须改动哪三个模块（路径）？为什么？

---

如果你确认已经理解这份“系统全景”，并能回答上面的问题，我们就进入第二阶段：从入口与运行方式开始，按“可复刻”的顺序逐阶段拆解与实现指导。你希望先从“启动/配置/入口分析（阶段 0）”开始吗？
