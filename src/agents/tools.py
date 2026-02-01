import math
import re

import numexpr
from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, tool
from langchain_openai import OpenAIEmbeddings


def calculator_func(expression: str) -> str:
    """使用 numexpr 计算数学表达式。

    当你需要使用 numexpr 回答关于数学的问题时很有用。
    此工具仅用于数学问题，没有其他用途。只能输入
    数学表达式。

    参数:
        expression (str): 一个有效的 numexpr 格式数学表达式。

    返回:
        str: 数学表达式的结果。
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # 限制对全局变量的访问
                local_dict=local_dict,  # 添加常用数学函数
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") 引发错误: {e}.'
            " 请使用有效的数值表达式重试"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"


# Format retrieved documents
def format_contexts(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_chroma_db():
    # 为我们的项目描述数据库创建嵌入函数
    try:
        embeddings = OpenAIEmbeddings()
    except Exception as e:
        raise RuntimeError(
            "初始化 OpenAIEmbeddings 失败。请确保已设置 OpenAI API 密钥。"
        ) from e

    # 加载存储的向量数据库
    chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = chroma_db.as_retriever(search_kwargs={"k": 5})
    return retriever


def database_search_func(query: str) -> str:
    """在公司手册中搜索 chroma_db 中的信息。"""
    # 获取 chroma 检索器
    retriever = load_chroma_db()

    # 在数据库中搜索相关文档
    documents = retriever.invoke(query)

    # 将文档格式化为字符串
    context_str = format_contexts(documents)

    return context_str


database_search: BaseTool = tool(database_search_func)
database_search.name = "Database_Search"  # 用您数据库的目的更新名称
