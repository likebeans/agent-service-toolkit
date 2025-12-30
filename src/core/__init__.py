"""
本包在整个项目中的角色：Core 基础能力聚合层（配置 + 模型工厂）。

为什么这个包存在？
- `settings`：统一的配置入口（来自环境变量/.env），供 service/agents/memory 等各层使用。
- `get_model`：统一的 LLM Provider/模型实例创建入口，屏蔽不同厂商 SDK 的差异。

典型调用者：
- `src/service/service.py`：读取 `settings` 进行鉴权、可用模型暴露、启动参数等。
- `src/agents/*`：通过 `get_model(...)` 获取可调用的 ChatModel（并在图中执行）。

设计意图：
- 提供一个“对外稳定”的 import 路径：上层代码可以 `from core import settings, get_model`，
  而不需要关心具体实现文件的名字与位置。
"""

from core.llm import get_model
from core.settings import settings

__all__ = ["settings", "get_model"]
