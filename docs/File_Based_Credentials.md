# 基于文件的凭据

在开发你的代理时，你可能会发现有一些需要存储在磁盘上的凭据，但又不希望它们存放在 Git 仓库中或被打包进容器镜像。

示例：
- 基于文件的 LLM 凭据文件（例如 Google Vertex）
- 与外部 API 通信所需的证书或私钥


`privatecredentials/` 文件夹在开发阶段为这些文件提供了一个快捷存放位置。


## 工作原理

*保护*
- `.dockerignore` 文件会排除整个文件夹，使其不参与构建过程。
- `.gitignore` 只允许 `.gitkeep` 文件——因为 git 不跟踪空目录。


*挂载卷*

Docker Compose 会将 `privatecredentials/` 挂载到容器中的 `/privatecredentials/`。运行中的容器能够访问你在开发环境中未被跟踪的这些文件。


*为何不使用 Docker Watch*

未使用 Docker Watch 的同步功能，原因如下：
- docker watch 遵循 `.dockerignore` 的规则，因此不会看到这些凭据；
- 即便能看到，docker watch 在容器启动时不会做初始同步，只会在服务运行期间同步发生的变更。


## 建议用法


对于每个基于文件的凭据，请执行以下操作：
1. 将文件（例如 `example-creds.txt`）放入 `privatecredentials/` 文件夹；
2. 在你的 `.env` 文件中为该凭据创建一个环境变量（例如 `EXAMPLE_CREDENTIAL=/privatecredentials/example-creds.txt`），供你的代理在运行时引用其路径；
3. 在你的代理中，在需要凭据路径的地方使用该环境变量。


### 示例

#### Google Vertex
Google Vertex SDK 使用 `GOOGLE_APPLICATION_CREDENTIALS` 环境变量定位你的凭据文件。

操作步骤：
1. 将 `service-account-key.json`（或 `google-credentials.json`）放入 `privatecredentials/` 文件夹；
2. 在你的 `.env` 文件中定义 `GOOGLE_APPLICATION_CREDENTIALS=/privatecredentials/service-account-key.json`；
3. Vertex SDK 会自动引用 `GOOGLE_APPLICATION_CREDENTIALS` 环境变量。


#### 使用远程 API 的签名通信证书
如果你的代理调用的远程 API 需要客户端证书，你的代理需要能访问该公有证书。

例如，假设你有一个名为 `my_remote_api_certificate.cer` 的证书。

操作步骤：
1. 将 `my_remote_api_certificate.cer` 放入 `privatecredentials/` 文件夹；
2. 在你的 `.env` 文件中定义 `MY_REMOTE_API_CERTIFICATE=/privatecredentials/my_remote_api_certificate.cer`；
3. 让代理中的 HTTP 客户端使用该环境变量访问该文件。


## 生产环境选项

在生产环境中，你需要让应用可以访问这些基于文件的凭据，并通过环境变量定义容器可访问它们的路径。

可以采用以下方法：

- 使用以数据卷方式挂载的 Kubernetes Secrets 或 Docker Secrets，使应用能够将它们作为文件访问；
- 使用云托管环境的密钥管理功能（Google Cloud Secrets、AWS Secrets Manager 等）；
- 使用第三方密钥管理平台；
- 在你的 Docker 主机上手动放置凭据，并通过挂载卷将凭据映射到容器中（安全性较低）。
