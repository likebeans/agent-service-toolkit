"""
本文件在整个项目中的角色：演示基于 Amazon Bedrock Knowledge Base 的 RAG Agent（云端知识库检索 + 生成）。

为什么这个文件存在？
- `rag_assistant.py` 演示了本地 Chroma 向量库的 RAG；而企业场景里常见的是“托管的知识库服务”。
- Amazon Bedrock Knowledge Base 提供：
  - 文档导入/索引/向量化/检索的托管能力
  - 应用侧只需要在运行时做 query -> retrieve，再把检索结果喂给 LLM
- 本文件用 `AmazonKnowledgeBasesRetriever` 做最小集成示例。

端到端链路（简化）：
1) `retrieve_documents`：从 Knowledge Base 检索相关文档片段（需要 `AWS_KB_ID`）
2) `prepare_augmented_prompt`：把文档片段格式化后写入 state（kb_documents）
3) `model`：构造 SystemMessage（包含文档上下文）并生成回答

典型调用者是谁？
- `src/agents/agents.py`：注册 `"knowledge-base-agent"`
- `src/service/service.py`：通过 `/invoke`、`/stream` 执行本图

注意（不改变行为的约束）：
- `base_prompt`、`document_prompt` 等字符串会直接发给 LLM，翻译/改写会改变回答风格与约束边界；
  因此我们只添加中文解释，不修改这些 prompt 文本。
"""

import logging
import os
from typing import Any

from langchain_aws import AmazonKnowledgeBasesRetriever
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langchain_core.runnables.base import RunnableSequence
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps

from core import get_model, settings

logger = logging.getLogger(__name__)


class AgentState(MessagesState, total=False):
    """
    Knowledge Base Agent 的状态（State）。

    字段说明：
    - messages：对话历史（MessagesState 提供）
    - remaining_steps：LangGraph 步数预算（预留字段；该示例里主要用于与其它 Agent 结构保持一致）
    - retrieved_documents：检索到的文档摘要（结构化 dict 列表，便于调试与二次处理）
    - kb_documents：格式化后的文档文本（会拼进 system prompt 给模型阅读）
    """

    remaining_steps: RemainingSteps
    retrieved_documents: list[dict[str, Any]]
    kb_documents: str


def get_kb_retriever():
    """
    构造并返回 Knowledge Base retriever。

    依赖：
    - 环境变量 `AWS_KB_ID`：指定要检索的 Knowledge Base。
    - AWS 权限与 Bedrock/KB 配置：由运行环境负责（本仓库只做应用侧调用示例）。
    """
    # 从环境变量读取 Knowledge Base ID
    kb_id = os.environ.get("AWS_KB_ID", "")
    if not kb_id:
        raise ValueError("AWS_KB_ID environment variable must be set")

    # 配置检索器：这里仅设置返回 TopK=3，作为演示的默认值。
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=kb_id,
        retrieval_config={
            "vectorSearchConfiguration": {
                "numberOfResults": 3,
            }
        },
    )
    return retriever


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    """
    将模型包装成 runnable：根据 state 动态构造 SystemMessage（包含检索到的文档上下文）。

    设计意图：
    - 不直接把“文档”拼到用户消息里，而是放入 SystemMessage：
      - 更符合“系统约束 + 上下文材料”的语义
      - 让模型在回答时更明确“只能使用这些 documents”
    """

    def create_system_message(state):
        # 注意：base_prompt 会直接发给 LLM；不改动内容以避免改变行为。
        base_prompt = """You are a helpful assistant that provides accurate information based on retrieved documents.

        You will receive a query along with relevant documents retrieved from a knowledge base. Use these documents to inform your response.

        Follow these guidelines:
        1. Base your answer primarily on the retrieved documents
        2. If the documents contain the answer, provide it clearly and concisely
        3. If the documents are insufficient, state that you don't have enough information
        4. Never make up facts or information not present in the documents
        5. Always cite the source documents when referring to specific information
        6. If the documents contradict each other, acknowledge this and explain the different perspectives

        Format your response in a clear, conversational manner. Use markdown formatting when appropriate.
        """

        # 如果 state 中已经有 kb_documents，则把它拼到系统提示词里作为“可引用材料”。
        if "kb_documents" in state:
            document_prompt = f"\n\nI've retrieved the following documents that may be relevant to the query:\n\n{state['kb_documents']}\n\nPlease use these documents to inform your response to the user's query. Only use information from these documents and clearly indicate when you are unsure."
            return [SystemMessage(content=base_prompt + document_prompt)] + state["messages"]
        else:
            # 未检索到文档：仍然返回 system prompt，但明确告诉模型没有材料可用。
            no_docs_prompt = (
                "\n\nNo relevant documents were found in the knowledge base for this query."
            )
            return [SystemMessage(content=base_prompt + no_docs_prompt)] + state["messages"]

    preprocessor = RunnableLambda(
        create_system_message,
        name="StateModifier",
    )
    return RunnableSequence(preprocessor, model)


async def retrieve_documents(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    图节点：从 Knowledge Base 检索与用户 query 相关的文档片段。

    端到端链路中的位置：
    - 图的起点（entry point）。先拿到材料，再进入 prompt 构造与模型回答。
    """
    # 从对话历史中取最后一条 HumanMessage 作为 query
    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not human_messages:
        # 理论上请求都会带一条 human 输入；这里是防御性兜底。
        return {"messages": [], "retrieved_documents": []}

    query = human_messages[-1].content

    try:
        # 初始化 retriever（可能依赖环境变量/权限配置）
        retriever = get_kb_retriever()

        # 执行检索（异步）
        retrieved_docs = await retriever.ainvoke(query)

        # 将检索结果整理成结构化摘要，写入 state，便于后续格式化与调试。
        document_summaries = []
        for i, doc in enumerate(retrieved_docs, 1):
            summary = {
                "id": doc.metadata.get("id", f"doc-{i}"),
                "source": doc.metadata.get("source", "Unknown"),
                "title": doc.metadata.get("title", f"Document {i}"),
                "content": doc.page_content,
                "relevance_score": doc.metadata.get("score", 0),
            }
            document_summaries.append(summary)

        logger.info(f"Retrieved {len(document_summaries)} documents for query: {query[:50]}...")

        return {"retrieved_documents": document_summaries, "messages": []}

    except Exception as e:
        # 检索失败的降级策略：返回空 documents，让模型走“无材料”路径（由 wrap_model 处理）。
        logger.error(f"Error retrieving documents: {str(e)}")
        return {"retrieved_documents": [], "messages": []}


async def prepare_augmented_prompt(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    图节点：把 retrieved_documents 格式化为可读文本，并写入 state["kb_documents"]。

    设计意图：
    - 模型通常更擅长阅读“排版过的文本”，而不是原始 dict 列表。
    - 将格式化结果放入 state，后续 wrap_model 就能把它拼到 SystemMessage 中。
    """
    documents = state.get("retrieved_documents", [])

    if not documents:
        return {"messages": []}

    # 为模型格式化文档：包含来源、标题、正文（简化示例，可按业务增强为引用 id/链接等）。
    formatted_docs = "\n\n".join(
        [
            f"--- Document {i + 1} ---\n"
            f"Source: {doc.get('source', 'Unknown')}\n"
            f"Title: {doc.get('title', 'Unknown')}\n\n"
            f"{doc.get('content', '')}"
            for i, doc in enumerate(documents)
        ]
    )

    return {"kb_documents": formatted_docs, "messages": []}


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    图节点：基于 state 中的 kb_documents（如果有）生成最终回答。

    注意：
    - 具体“是否严格只使用 documents”依赖 system prompt 的约束强度与模型遵循程度；
      生产环境可能需要更强的引用/对齐机制。
    """
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)

    response = await model_runnable.ainvoke(state, config)

    return {"messages": [response]}


# -----------------------------
# 图定义：retrieve_documents -> prepare_augmented_prompt -> model -> END
# -----------------------------
agent = StateGraph(AgentState)

agent.add_node("retrieve_documents", retrieve_documents)
agent.add_node("prepare_augmented_prompt", prepare_augmented_prompt)
agent.add_node("model", acall_model)

agent.set_entry_point("retrieve_documents")

agent.add_edge("retrieve_documents", "prepare_augmented_prompt")
agent.add_edge("prepare_augmented_prompt", "model")
agent.add_edge("model", END)

kb_agent = agent.compile()
