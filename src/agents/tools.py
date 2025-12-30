"""
本文件在整个项目中的角色：定义“Agent 可调用的本地工具（Tools）”集合。

为什么这个文件存在？
- 在 LangChain/LangGraph 体系中，工具（Tool）是模型“调用外部能力”的标准接口：
  - 模型在输出中产生 tool_calls（想调用哪个工具、参数是什么）
  - LangGraph 的 ToolNode 负责真正执行工具函数，并把结果作为 ToolMessage 写回 messages
- 本仓库把通用工具放在这里，供多个 Agent 复用：
  - `Calculator`：用 numexpr 安全地计算表达式（服务端运行，不依赖模型）
  - `Database_Search`：从 Chroma 向量库检索（RAG 示例）

典型调用者是谁？
- `src/agents/research_assistant.py`：使用 `calculator` + WebSearch/Weather
- `src/agents/rag_assistant.py`：使用 `database_search`

注意（不改变行为的约束，尤其重要）：
- LangChain 的 `tool()` 装饰/封装会读取函数的 docstring 作为“工具描述”，并将其发给 LLM。
  这会直接影响模型选择工具与参数生成，因此【不要翻译/改写这些工具函数的 docstring】。
  本文件会用中文注释解释设计意图，但保留原始英文 docstring。
"""

import math
import re

import numexpr
from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, tool
from langchain_openai import OpenAIEmbeddings


def calculator_func(expression: str) -> str:
    # 重要：下面这个 docstring 会被 LangChain 作为工具描述发给模型；改动会改变 LLM 行为。
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # 限制对全局变量的访问（安全性：避免任意执行）
                local_dict=local_dict,  # 仅注入必要的常量/函数（pi、e 等）
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"


# 将检索到的文档列表拼接成一段上下文文本，供模型阅读与总结。
def format_contexts(docs):
    """把检索结果格式化为单个字符串（每段之间用空行分隔）。"""
    return "\n\n".join(doc.page_content for doc in docs)


def load_chroma_db():
    """
    加载本地 Chroma 向量库，并返回一个 retriever。

    端到端链路中的位置：
    - `Database_Search` 工具调用时触发，属于“工具执行阶段”的一部分。

    依赖说明：
    - Embeddings 使用 OpenAIEmbeddings（需要 OPENAI_API_KEY）。
    - Chroma 的持久化目录默认是仓库根目录 `./chroma_db`（可通过脚本生成：`scripts/create_chroma_db.py`）。
    """

    # 为向量库准备 embedding 函数（将 query/文本块向量化，用于相似度检索）
    try:
        embeddings = OpenAIEmbeddings()
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize OpenAIEmbeddings. Ensure the OpenAI API key is set."
        ) from e

    # 加载已持久化的向量库
    chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = chroma_db.as_retriever(search_kwargs={"k": 5})
    return retriever


def database_search_func(query: str) -> str:
    # 重要：下面这个 docstring 会被 LangChain 作为工具描述发给模型；改动会改变 LLM 行为。
    """Searches chroma_db for information in the company's handbook."""
    # 获取 retriever（对 query 做相似度检索）
    retriever = load_chroma_db()

    # 检索相关文档片段
    documents = retriever.invoke(query)

    # 格式化为一段可读的上下文，返回给模型作为工具输出
    context_str = format_contexts(documents)

    return context_str


database_search: BaseTool = tool(database_search_func)
database_search.name = "Database_Search"  # 可按你的业务语义调整名字（这会影响模型如何选择工具）
