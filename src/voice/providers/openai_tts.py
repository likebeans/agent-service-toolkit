"""
本文件在整个项目中的角色：OpenAI 的 TTS Provider 实现。

它解决的核心问题是什么？
- 把 OpenAI 的语音合成 API 封装成一个最小接口：
  - `generate(text) -> bytes | None`
  - `get_format() -> str`（MIME type）
- 在 provider 内部处理：
  - 文本长度约束（避免无意义/超限请求）
  - 参数校验（voice/model 合法性）
  - 异常处理（记录日志并返回 None，便于上层“优雅降级”）

典型调用者：
- `src/voice/tts.py:TextToSpeech`（工厂）实例化并委托调用
- `src/voice/manager.py:VoiceManager.render_message()` 在 UI 中生成并播放音频
"""

import logging

from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAITTS:
    """OpenAI TTS provider（最小实现：提供 `generate()` 与 `get_format()`）。"""

    # OpenAI TTS API 的输入长度约束（超过会被拒绝或报错）
    MAX_TEXT_LENGTH = 4096
    MIN_TEXT_LENGTH = 3

    # OpenAI TTS 可选配置项（用于参数校验）
    VALID_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    VALID_MODELS = ["tts-1", "tts-1-hd"]

    def __init__(self, api_key: str | None = None, voice: str = "alloy", model: str = "tts-1"):
        """初始化 OpenAI TTS provider。

        Args:
            api_key: OpenAI API key（不传则由 OpenAI SDK 自行从环境变量读取）
            voice: 声线名称（alloy/echo/fable/onyx/nova/shimmer）
            model: 模型名称（tts-1 或 tts-1-hd）

        Raises:
            ValueError: voice/model 不合法
            Exception: OpenAI client 初始化失败时抛出（属于“启动即失败”的配置问题）
        """
        # 参数校验：尽早失败，避免请求发出去才报错。
        if voice not in self.VALID_VOICES:
            raise ValueError(f"Invalid voice '{voice}'. Must be one of {self.VALID_VOICES}")

        if model not in self.VALID_MODELS:
            raise ValueError(f"Invalid model '{model}'. Must be one of {self.VALID_MODELS}")

        # 创建 OpenAI 客户端：显式传入 api_key 便于在测试/多账号场景中覆盖。
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.voice = voice
        self.model = model

        logger.info(f"OpenAI TTS initialized: voice={voice}, model={model}")

    def _validate_and_prepare_text(self, text: str) -> str | None:
        """校验并规范化输入文本，确保满足 TTS API 约束。

        Args:
            text: 原始输入文本

        Returns:
            可用于 TTS 的文本；若过短则返回 None

        Note:
            - 去掉首尾空白
            - 少于 MIN_TEXT_LENGTH 的文本直接跳过（不值得一次 API 调用）
            - 超过 MAX_TEXT_LENGTH 的文本会截断（避免请求被拒绝）
        """
        # 去掉首尾空白，避免只有空格时也触发合成。
        text = text.strip()

        # 文本过短直接跳过：通常用于避免 streaming 场景下频繁合成极短片段。
        if len(text) < self.MIN_TEXT_LENGTH:
            logger.debug(f"OpenAI TTS: skipping short text ({len(text)} chars)")
            return None

        # 超过 API 限制则截断，并记录 warning 方便你意识到“输出被截断”。
        if len(text) > self.MAX_TEXT_LENGTH:
            logger.warning(
                f"OpenAI TTS: truncating from {len(text)} to {self.MAX_TEXT_LENGTH} chars"
            )
            text = text[: self.MAX_TEXT_LENGTH]

        return text

    def generate(self, text: str) -> bytes | None:
        """根据文本生成语音（返回音频 bytes）。

        Args:
            text: 要合成的文本

        Returns:
            MP3 音频 bytes；若文本过短或生成失败则返回 None

        Note:
            - 过短文本返回 None
            - 超长文本会截断
            - 异常只记录日志，不向上抛出（便于上层优雅降级）
        """
        # 先做输入校验与规范化，避免无效请求。
        prepared_text = self._validate_and_prepare_text(text)
        if not prepared_text:
            return None

        try:
            # 调用 OpenAI TTS API（response_format="mp3"）。
            response = self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=prepared_text,
                response_format="mp3",
            )

            # OpenAI SDK 返回的 response.content 即音频 bytes。
            audio_bytes = response.content
            logger.info(f"OpenAI TTS: generated {len(audio_bytes)} bytes")
            return audio_bytes

        except Exception as e:
            # 记录完整堆栈便于排障（但不让异常向上传播）。
            logger.error(f"OpenAI TTS failed: {e}", exc_info=True)
            # 返回 None 让上层做降级（例如只显示文字、不播放音频）。
            return None

    def get_format(self) -> str:
        """获取生成音频的 MIME type。

        Returns:
            MIME type（用于 Streamlit 的 `st.audio(..., format=...)`）
        """
        return "audio/mp3"
