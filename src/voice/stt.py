"""
本文件在整个项目中的角色：STT（Speech-to-Text）工厂 / Provider 选择器。

它解决的核心问题是什么？
- 对上层暴露一个稳定的接口：`SpeechToText.transcribe(audio_file) -> str`
- 内部按配置选择具体 provider 的实现（当前实现了 OpenAI；预留 Deepgram 等扩展点）
- 将“读取环境变量、选择 provider、构造 provider 实例”的复杂度收敛在一个地方，
  让 UI/业务代码不用关心不同 provider 的 SDK 差异

典型调用者：
- `src/voice/manager.py`：在 Streamlit 场景下，把音频输入转写为文本。
"""

import logging
import os
from typing import BinaryIO, Literal, cast

logger = logging.getLogger(__name__)

Provider = Literal["openai", "deepgram"]


class SpeechToText:
    """
    STT 工厂类：对外提供统一的“转写”能力，对内委托给具体 provider 实现。

    Example:
        >>> stt = SpeechToText(provider="openai")
        >>> text = stt.transcribe(audio_file)
        >>>
        >>> # Or from environment
        >>> stt = SpeechToText.from_env()
        >>> if stt:
        ...     text = stt.transcribe(audio_file)
    """

    def __init__(self, provider: Provider = "openai", api_key: str | None = None, **config):
        """初始化 STT（按 provider 选择具体实现）。

        Args:
            provider: provider 名称（"openai" / "deepgram" 等）
            api_key: API key（不传则尝试从环境变量读取）
            **config: provider 特有的额外配置

        Raises:
            ValueError: provider 不被支持
        """
        self._provider_name = provider

        # API key 的解析优先级：显式参数 > 环境变量（便于在 Notebook/测试里临时覆盖）。
        resolved_api_key = self._get_api_key(provider, api_key)

        # 按 provider 动态加载实现类并实例化（避免把所有 provider 的依赖一次性 import 进来）。
        self._provider = self._load_provider(provider, resolved_api_key, config)

        logger.info(f"SpeechToText created with provider={provider}")

    def _get_api_key(self, provider: Provider, api_key: str | None) -> str | None:
        """获取 API key：优先用参数，其次读环境变量。

        Args:
            provider: provider 名称
            api_key: 调用方传入的 key（优先级最高）

        Returns:
            解析后的 key（可能为 None，取决于 provider SDK 是否支持“从环境变量自动读取”）
        """
        # 显式传入则直接使用。
        if api_key:
            return api_key

        # 否则按 provider 约定读取环境变量。
        match provider:
            case "openai":
                return os.getenv("OPENAI_API_KEY")
            case "deepgram":
                return os.getenv("DEEPGRAM_API_KEY")
            case _:
                return None

    def _load_provider(self, provider: Provider, api_key: str | None, config: dict):
        """加载具体的 STT provider 实现并返回实例。

        Args:
            provider: provider 名称
            api_key: 已解析的 key
            config: provider 特有配置

        Returns:
            provider 实例（需要实现 `transcribe(audio_file)`）

        Raises:
            ValueError: provider 不被支持
            NotImplementedError: provider 预留但尚未实现
        """
        match provider:
            case "openai":
                from voice.providers.openai_stt import OpenAISTT

                return OpenAISTT(api_key=api_key, **config)

            case "deepgram":
                # 未来扩展示例：如果要支持 Deepgram，需要新增对应 provider 实现并在此处实例化。
                # from voice.providers.deepgram_stt import DeepgramSTT
                # return DeepgramSTT(api_key=api_key, **config)
                raise NotImplementedError("Deepgram STT provider not yet implemented")

            case _:
                # 兜底：未知 provider 直接抛错，避免静默降级导致“看起来没声音但其实配置错了”。
                raise ValueError(f"Unknown STT provider: {provider}. Available providers: openai")

    @property
    def provider(self) -> str:
        """获取当前 provider 名称。"""
        return self._provider_name

    @classmethod
    def from_env(cls) -> "SpeechToText | None":
        """从环境变量创建 SpeechToText（未配置则返回 None）。

        读取 `VOICE_STT_PROVIDER` 来决定使用哪个 provider。
        若未配置该环境变量，则认为“语音转写功能关闭”，返回 None。

        Returns:
            SpeechToText 实例或 None

        Example:
            >>> # In .env: VOICE_STT_PROVIDER=openai
            >>> stt = SpeechToText.from_env()
            >>> if stt:
            ...     text = stt.transcribe(audio_file)
        """
        provider = os.getenv("VOICE_STT_PROVIDER")

        # 未配置 provider => 语音能力关闭（保持应用可用，而不是强依赖语音能力）。
        if not provider:
            logger.debug("VOICE_STT_PROVIDER not set, STT disabled")
            return None

        try:
            # 用环境变量创建实例：会校验 provider 合法性。
            return cls(provider=cast(Provider, provider))
        except Exception as e:
            # 记录错误但不让应用崩溃：语音属于可选能力，失败时应“优雅降级”。
            logger.error(f"Failed to create STT provider: {e}", exc_info=True)
            return None

    def transcribe(self, audio_file: BinaryIO) -> str:
        """转写音频为文本（对外统一入口）。

        实现方式：
        - 直接委托给底层 provider（例如 `OpenAISTT.transcribe`）。

        Args:
            audio_file: 二进制音频文件（file-like）

        Returns:
            转写文本（失败通常返回空字符串，具体取决于 provider 实现）
        """
        return self._provider.transcribe(audio_file)
