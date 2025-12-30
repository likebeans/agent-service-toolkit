"""
本文件在整个项目中的角色：离线构建 RAG 用的向量库（Chroma DB）脚本。

它解决的核心问题是什么？
- `rag-assistant` 这类 Agent 需要一个可检索的知识库（向量库）：
  - 文档（PDF/DOCX）-> 切分成 chunk -> 生成 embedding -> 写入 Chroma 持久化目录
- 该脚本用于在本地/预处理阶段构建数据库，避免在服务运行时边加载边构建导致启动慢/不稳定。

典型调用者：
- 人工执行：`python scripts/create_chroma_db.py`
- 或在你复刻项目时，把它当作“数据准备 pipeline”的参考实现。

与项目其它模块的关系：
- `src/agents/rag_assistant.py`（RAG Agent）会读取/使用同一个 Chroma 持久化目录进行检索。
"""

import os
import shutil

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings

# 从 `.env` 加载环境变量（例如 OPENAI_API_KEY）
load_dotenv()


def create_chroma_db(
    folder_path: str,
    db_name: str = "./chroma_db",
    delete_chroma_db: bool = True,
    chunk_size: int = 2000,
    overlap: int = 500,
):
    """
    从指定目录中的文档构建 Chroma 向量库，并返回 Chroma 实例。

    端到端链路中的位置（RAG 数据准备阶段）：
    1) 读取文件夹中的文档（当前支持 PDF/DOCX）
    2) 使用 `RecursiveCharacterTextSplitter` 切分为 chunk
    3) 用 `OpenAIEmbeddings` 计算向量
    4) 写入 Chroma 持久化目录（persist_directory）

    Args:
        folder_path: 文档所在目录（例如 `./data`）
        db_name: Chroma 持久化目录名（默认 `./chroma_db`）
        delete_chroma_db: 是否删除已有数据库目录（True 表示每次重建）
        chunk_size: chunk 大小（字符级别）
        overlap: chunk 重叠大小（字符级别）
    """
    # embedding 模型需要 OPENAI_API_KEY；未配置会抛 KeyError，属于“环境未准备好”的早失败。
    embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])

    # 初始化 Chroma 向量库（可选择删除旧目录，避免增量构建造成重复写入）
    if delete_chroma_db and os.path.exists(db_name):
        shutil.rmtree(db_name)
        print(f"Deleted existing database at {db_name}")

    chroma = Chroma(
        embedding_function=embeddings,
        persist_directory=f"./{db_name}",
    )

    # 初始化文本切分器：chunk_size/overlap 需要结合文档长度、模型上下文窗口、检索效果做权衡。
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    # 遍历目录中文件并写入向量库
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 按扩展名选择文档 loader：
        # - 需要支持更多类型时，可在此处扩展（例如 TxtLoader/JSONLoader 等）。
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            continue  # Skip unsupported file types

        # 加载并切分文档为 chunk
        document = loader.load()
        chunks = text_splitter.split_documents(document)

        # 将 chunk 写入 Chroma 向量库
        for chunk in chunks:
            chunk_id = chroma.add_documents([chunk])
            if chunk_id:
                print(f"Chunk added with ID: {chunk_id}")
            else:
                print("Failed to add chunk")

        print(f"Document {filename} added to database.")

    print(f"Vector database created and saved in {db_name}.")
    return chroma


if __name__ == "__main__":
    # 文档目录路径
    folder_path = "./data"

    # 构建 Chroma 向量库
    chroma = create_chroma_db(folder_path=folder_path)

    # 从向量库创建 retriever（k=3 表示返回最相似的 3 个 chunk）
    retriever = chroma.as_retriever(search_kwargs={"k": 3})

    # 简单相似度检索示例
    query = "What's my company's mission and values"
    similar_docs = retriever.invoke(query)

    # 打印检索结果
    for i, doc in enumerate(similar_docs, start=1):
        print(f"\n🔹 Result {i}:\n{doc.page_content}\nTags: {doc.metadata.get('source', [])}")
