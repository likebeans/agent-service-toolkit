# 创建 RAG 助手

你可以使用 Chroma 数据库构建一个 RAG 助手。

## 设置 Chroma

要创建一个 Chroma 数据库：

1. 将要使用的数据添加到一个文件夹（如 `./data`）。目前支持 Word 和 PDF 文件。
2. 打开[`create_chroma_db.py` 文件](../scripts/create_chroma_db.py)，将 `folder_path` 变量设置为你的数据路径（如 `./data`）。
3. 你可以更改数据库名称、分块大小以及重叠大小。
4. 假设你已按照[快速开始](#quickstart)完成并激活了虚拟环境，运行以下命令以创建数据库：

```sh
python scripts/create_chroma_db.py
```

5. 如果成功，将在仓库根目录创建一个 Chroma 数据库。

## 配置 RAG 助手

要创建一个 RAG 助手：
1. 打开[`tools.py` 文件](../src/agents/tools.py)，确保 `persist_directory` 指向你之前创建的数据库。
2. 修改返回的文档数量，当前设置为 5。
3. 更新 `database_search_func` 函数的描述，以准确说明你的数据库用途与包含的内容。
4. 打开[`rag_assistant.py` 文件](../src/agents/rag_assistant.py)，更新代理的指令以描述助手的专长以及可访问的知识，例如：

```python
instructions = f"""
    你是一名乐于助人的人力资源（HR）助手，能够搜索一个包含公司政策、福利以及员工手册信息的数据库。
    今天的日期是 {current_date}。

    注意：用户无法看到工具的原始响应。

    请记住以下事项：
    - 如果你可以访问多个数据库，请在撰写回答前从多样来源收集信息。
    - 请在回复中包含所使用信息的来源。
    - 回复时使用友好但专业的语气。
    - 只使用数据库中的信息。不要使用外部来源的信息。
    """
```

5. 打开[`streamlit_app.py` 文件](../src/streamlit_app.py)，更新代理的欢迎语：

```python
WELCOME = """你好！我是你的 AI 驱动 HR 助手，帮你查阅公司政策、员工手册与福利。随时向我提问！"""
```

6. 运行应用并测试你的 RAG 助手。
