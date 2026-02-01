# GitHub MCP 代理

GitHub MCP 代理是一个使用 GitHub MCP（Model Context Protocol，模型上下文协议）工具来进行代码仓库管理与开发流程的专用代理。它基于 LangGraph 的 `create_react_agent` 构建，实现了清晰的 ReAct（推理与行动）模式。

**此代理旨在演示使用 MCP（模型上下文协议）服务器与工具的代理。**

## 功能
[GitHub MCP 服务器](https://github.com/github/github-mcp-server) 在配置了 PAT 后提供多种工具。

- 仓库管理（创建、克隆、浏览）
- Issue 管理（创建、列表、更新、关闭）
- Pull Request 管理（创建、评审、合并）
- 分支管理（创建、切换、合并）
- 文件操作（读取、写入、搜索）
- 提交操作（创建、查看历史）

## 配置

要启用 GitHub MCP 代理，你需要配置以下环境变量：

### 必需设置

```bash
# GitHub 个人访问令牌（PAT），用于 GitHub MCP 服务器
# 如果未设置，GitHub MCP 代理将没有可用工具
GITHUB_PAT=your_github_personal_access_token_here
```

### 可选设置

```bash
# GitHub MCP 服务器 URL（默认：https://api.githubcopilot.com/mcp/）
MCP_GITHUB_SERVER_URL=https://api.githubcopilot.com/mcp/
```

## GitHub 个人访问令牌

使用 GitHub MCP 代理需要一个具备适当权限的 GitHub 个人访问令牌（PAT）。GitHub MCP 服务器提供了广泛的仓库管理工具，不同工具需要不同的作用域。

1. 前往 GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. 生成一个具有以下作用域的令牌：
   - `repo`（对私有仓库的完全控制）——启用诸如 `create_issue`、`create_pull_request`、`get_file_contents`、`list_commits` 等工具
   - `read:org`（读取组织和团队成员信息）——启用与组织相关的工具
   - `read:user`（读取用户资料数据）——启用用户资料相关工具
   - `user:email`（访问用户邮箱地址）——启用邮箱相关功能

**注意**：可用的具体工具取决于你的 PAT 作用域。使用上述推荐作用域，你将获得大多数仓库管理工具的访问能力，包括创建 issue、pull request、读取文件内容以及列出提交记录。

## 用法

配置完成后，GitHub MCP 代理将在服务中以 `github-mcp-agent` 的名称可用。

### 示例提示词

以下是一些可与 GitHub MCP 代理配合使用的示例提示词：

- **“描述 JoshuaC215/agent-service-toolkit 仓库”** —— 展示仓库信息和 README 内容
- **“列出该仓库的近期提交”** —— 显示最近的提交历史
- **“src 目录里有哪些文件？”** —— 列出特定目录的文件
- **“显示 README 文件”** —— 显示仓库 README 内容
- **“创建一个标题为‘Bug：登录不可用’，描述为‘登录表单没有响应用户输入’的新 issue”** —— 创建一个新 issue
- **“该仓库有哪些未关闭的 issue？”** —— 列出未关闭的 issue
- **“从 feature-branch 到 main 创建一个标题为‘Add new feature’的 pull request”** —— 创建一个 pull request
- **“显示该仓库的信息”** —— 展示仓库详情
