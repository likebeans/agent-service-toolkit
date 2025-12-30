"""
本文件在整个项目中的角色：TTS（Text-to-Speech）工厂 / Provider 选择器。

它解决的核心问题是什么？
- 对上层暴露一个稳定的接口：
  - `TextToSpeech.generate(text) -> bytes | None`
  - `TextToSpeech.get_format() -> str`（返回 MIME type，方便 UI 播放）
- 内部按配置选择具体 provider（当前实现 OpenAI；预留 ElevenLabs 等扩展点）
- 将“读取环境变量、选择 provider、构造 provider 实例”的复杂度收敛在一个地方，
  让 UI/业务代码不必理解不同 TTS SDK 的差异与参数细节

典型调用者：
- `src/voice/manager.py:VoiceManager.render_message()`：在 Streamlit 中为 AI 回复生成语音并播放。
"""

import logging
import os
from typing import Literal, cast

logger = logging.getLogger(__name__)

Provider = Literal["openai", "elevenlabs"]


class TextToSpeech:
    """
    TTS 工厂类：对外提供统一的“语音合成”能力，对内委托给具体 provider 实现。

    Example:
        >>> tts = TextToSpeech(provider="openai", voice="nova")
        >>> audio = tts.generate("Hello world")
        >>>
        >>> # Or from environment
        >>> tts = TextToSpeech.from_env()
        >>> if tts:
        ...     audio = tts.generate("Hello world")
    """

    def __init__(self, provider: Provider = "openai", api_key: str | None = None, **config):
        """初始化 TTS（按 provider 选择具体实现）。

        Args:
            provider: provider 名称（"openai" / "elevenlabs" 等）
            api_key: API key（不传则尝试从环境变量读取）
            **config: provider 特有配置
                OpenAI: voice="alloy", model="tts-1"
                ElevenLabs: voice_id="...", model_id="..."

        Raises:
            ValueError: provider 不被支持
        """
        self._provider_name = provider

        # API key 的解析优先级：显式参数 > 环境变量。
        resolved_api_key = self._get_api_key(provider, api_key)

        # 动态加载 provider 实现并实例化（避免引入未使用 provider 的依赖）。
        self._provider = self._load_provider(provider, resolved_api_key, config)

        logger.info(f"TextToSpeech created with provider={provider}")

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
            case "elevenlabs":
                return os.getenv("ELEVENLABS_API_KEY")
            case _:
                return None

    def _load_provider(self, provider: Provider, api_key: str | None, config: dict):
        """加载具体的 TTS provider 实现并返回实例。

        Args:
            provider: provider 名称
            api_key: 已解析的 key
            config: provider 特有配置

        Returns:
            provider 实例（需要实现 `generate(text)` 与 `get_format()`）

        Raises:
            ValueError: provider 不被支持
            NotImplementedError: provider 预留但尚未实现
        """
        match provider:
            case "openai":
                from voice.providers.openai_tts import OpenAITTS

                # 提取 OpenAI 特有配置，并提供默认值（保持使用门槛低）。
                voice = config.get("voice", "alloy")
                model = config.get("model", "tts-1")

                return OpenAITTS(api_key=api_key, voice=voice, model=model)

            case "elevenlabs":
                # 未来扩展示例：如果要支持 ElevenLabs，需要新增对应 provider 实现并在此处实例化。
                # from voice.providers.elevenlabs_tts import ElevenLabsTTS
                # voice_id = config.get("voice_id")
                # model_id = config.get("model_id", "eleven_monolingual_v1")
                # return ElevenLabsTTS(api_key=api_key, voice_id=voice_id, model_id=model_id)
                raise NotImplementedError("ElevenLabs TTS provider not yet implemented")

            case _:
                # 兜底：未知 provider 直接抛错，避免静默降级导致“没有声音但不知为何”。
                raise ValueError(f"Unknown TTS provider: {provider}. Available providers: openai")

    @property
    def provider(self) -> str:
        """获取当前 provider 名称。"""
        return self._provider_name

    @classmethod
    def from_env(cls) -> "TextToSpeech | None":
        """从环境变量创建 TextToSpeech（未配置则返回 None）。

        读取 `VOICE_TTS_PROVIDER` 来决定使用哪个 provider。
        若未配置该环境变量，则认为“语音合成功能关闭”，返回 None。

        Returns:
            TextToSpeech 实例或 None

        Example:
            >>> # In .env: VOICE_TTS_PROVIDER=openai
            >>> tts = TextToSpeech.from_env()
            >>> if tts:
            ...     audio = tts.generate("Hello world")
        """
        provider = os.getenv("VOICE_TTS_PROVIDER")

        # 未配置 provider => 语音能力关闭（保持应用可用，而不是强依赖语音能力）。
        if not provider:
            logger.debug("VOICE_TTS_PROVIDER not set, TTS disabled")
            return None

        try:
            # 用环境变量创建实例：会校验 provider 合法性。
            return cls(provider=cast(Provider, provider))
        except Exception as e:
            # 记录错误但不让应用崩溃：语音属于可选能力，失败时应“优雅降级”。
            logger.error(f"Failed to create TTS provider: {e}", exc_info=True)
            return None

    def generate(self, text: str) -> bytes | None:
        """根据文本生成语音（对外统一入口）。

        Args:
            text: 要合成的文本

        Returns:
            音频 bytes（格式取决于 provider）；失败返回 None
        """
        return self._provider.generate(text)

    def get_format(self) -> str:
        """获取当前 provider 生成音频的 MIME type（供 UI 播放使用）。

        Returns:
            MIME type（例如 "audio/mp3"）
        """
        return self._provider.get_format()
