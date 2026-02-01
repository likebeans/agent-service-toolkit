# 使用 Ollama

⚠️ _**注意：** agent-service-toolkit 中的 Ollama 支持仍处于实验阶段，可能无法按预期工作。以下说明已在 MacBook Pro 上的 Docker Desktop 环境测试。如遇到任何问题，请提交 issue。_

你也可以使用 [Ollama](https://ollama.com) 来运行为代理服务提供能力的 LLM。

1. 按照 https://github.com/ollama/ollama 的说明安装 Ollama
1. 安装你希望使用的模型，例如 `ollama pull llama3.2`，并将 `OLLAMA_MODEL` 环境变量设置为所用模型，例如 `OLLAMA_MODEL=llama3.2`

如果你在本地运行服务（例如 `python src/run_service.py`），即可直接使用！

如果你在 Docker 中运行服务，还需要：

1. 按照[此处的说明配置 Ollama 服务器](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-configure-ollama-server)，例如在 macOS 上运行 `launchctl setenv OLLAMA_HOST "0.0.0.0"`，并重启 Ollama。
1. 将 `OLLAMA_BASE_URL` 环境变量设为 Ollama 服务器的基础地址，例如 `OLLAMA_BASE_URL=http://host.docker.internal:11434`
1. 或者，你也可以在 Docker 中运行 `ollama/ollama` 镜像并使用类似的配置（但在部分场景下可能更慢）。
