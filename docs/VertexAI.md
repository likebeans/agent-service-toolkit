# 使用 Google 模型

Google 提供两种访问其模型的方式，它们受不同的使用条款和使用政策约束。

建议你选择哪种方法超出了本文档的范围，但可以考虑以下差异。

可选方式：

1. Gemini 开发者 API（[文档链接](https://ai.google.dev/gemini-api/docs)）
2. Google Cloud Platform 上的 Google Vertex AI（[文档链接](https://cloud.google.com/vertex-ai/docs)）


## 使用 Gemini 开发者 API

[从 Google 获取 Gemini API Key](https://ai.google.dev/gemini-api/docs)，并在 Agent Service Toolkit 中快速使用。

1. 将你的 API Key 写入 `.env` 文件中的 `GOOGLE_API_KEY` 环境变量；
2. Agent Service Toolkit 会识别这些凭据，你即可开始使用。


## 在 Google Cloud Platform 使用 Google Vertex AI

### 前置条件
确保你拥有一个[Google Cloud 项目](https://console.cloud.google.com/projectcreate)，并且[已启用计费](https://console.cloud.google.com/billing)。

### 关于认证
要以编程方式使用 Vertex AI，你需要创建一个**服务账号**，并使用其凭据让你的应用进行身份验证。该凭据与个人 Google 账号凭据不同，会决定你的应用对 Google Cloud 服务和 API 的访问权限。

Vertex 使用 JSON 格式的凭据文件，并通过读取 `GOOGLE_APPLICATION_CREDENTIALS` 环境变量在运行时获取该凭据文件路径。


### 模型

Vertex AI 包含稳定版与实验/预览版模型。实验与预览模型可能会变更或在未通知的情况下停用，因此对于**生产应用**，强烈建议使用稳定版模型。请在[Vertex AI 文档](https://cloud.google.com/vertex-ai/docs)中查看关于模型状态的最新信息。


### 步骤

#### 1. 启用 Vertex AI API
- 前往 [Google Cloud API 库](https://console.cloud.google.com/apis/library)。
- 从顶部下拉选择你的项目。
- 搜索 “Vertex AI API”，点击 **启用**。

#### 2. 创建并配置服务账号
- 进入[凭据页面](https://console.cloud.google.com/apis/credentials)。
- 点击 **创建凭据** > **服务账号**。
- 填写相关信息（如名称与描述）。
- **分配角色**：对于 Vertex AI，至少授予 “Vertex AI User” 角色。
- 点击 **完成**，然后找到你的服务账号，点击右侧三个点（⋮），选择 **管理密钥**。
- 点击 **添加密钥** > **创建新密钥**，选择 **JSON**，并点击 **创建**。
- 将下载 JSON 密钥文件。**请妥善保存**——该文件无法再次下载。

#### 3. 将 JSON 密钥文件放入[基于文件的凭据](docs/File_Based_Credentials.md)路径
将下载的 JSON 文件放入项目的 `privatecredentials/`（例如 `privatecredentials/service-account-key.json`）。

[基于文件的凭据](docs/File_Based_Credentials.md)路径的内容会在运行时以 `/privatecredentials/` 的形式提供给你的容器，同时会被排除在 git 提交与 docker 构建之外。

#### 4. 设置 `GOOGLE_APPLICATION_CREDENTIALS` 环境变量
将 `GOOGLE_APPLICATION_CREDENTIALS` 设置为你的 JSON 文件的**完整路径**：
  - **.env**（用于 docker compose）：
    ```
    GOOGLE_APPLICATION_CREDENTIALS=/privatecredentials/service-account-key.json
    ```
  - **类 Unix 系统**：
    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/project/privatecredentials/service-account-key.json
    ```
  - **Windows**：
    ```cmd
    set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\your\project\privatecredentials\service-account-key.json
    ```

#### 5. 保护你的凭据
- 确保该 JSON 文件被 `.gitignore` 覆盖（如果你已放在提供的 `privatecredentials/` 文件夹中，则已完成）：
  ```
  service-account-key.json
  ```
- **请务必将该文件保密**，因为它可以访问你的 Google Cloud 资源，可能导致**未经授权的使用**或产生费用。


### 验证你的设置
使用以下命令测试你的凭据：
  ```bash
  gcloud auth activate-service-account --key-file=/path/to/your/service-account-key.json
  gcloud auth list
  ```
  你的服务账号应显示为已激活。

### 生产环境说明
上述设置非常适合开发环境。在生产环境中，请考虑更安全的替代方案。部分选项列在[基于文件的凭据](docs/File_Based_Credentials.md)页面中。
