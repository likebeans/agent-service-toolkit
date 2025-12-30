"""
本文件在整个项目中的角色：VoiceManager（Streamlit 集成层）。

它解决的核心问题是什么？
- 让“语音能力（STT/TTS）”能以符合 Streamlit 交互模型的方式落地：
  - 输入侧：`st.chat_input(..., accept_audio=True)` + 自动转写
  - 输出侧：在渲染消息时可选生成语音，并用 `st.audio(...)` 播放
- 将 Streamlit 依赖隔离在本模块内：
  - `SpeechToText` / `TextToSpeech` 不依赖 Streamlit，可以在其它框架/脚本中复用
  - 只有 VoiceManager 负责 UI spinner、错误提示、session_state 等“前端体验”

典型调用者：
- `src/streamlit_app.py`：在 UI 里创建 VoiceManager，然后调用 `get_chat_input()` / `render_message()`。
"""

import logging
from typing import Optional

import streamlit as st

from voice.stt import SpeechToText
from voice.tts import TextToSpeech

logger = logging.getLogger(__name__)


class VoiceManager:
    """
    Streamlit 场景下的语音能力“门面对象”（Facade）。

    设计意图：
    - 让 UI 侧只关心两个动作：
      1) 获取用户输入（文本或语音转写）：`get_chat_input()`
      2) 渲染 AI 回复（可选附带语音）：`render_message()`
    - 真正的语音处理（转写/合成）由 `SpeechToText` / `TextToSpeech` 实现，这里只负责：
      - 用户体验：spinner、错误提示、占位符
      - 状态：把生成的音频缓存到 `st.session_state` 以跨 rerun 保留

    Example:
        >>> voice = VoiceManager.from_env()
        >>>
        >>> if voice:
        ...     user_input = voice.get_chat_input()
        ...     if user_input:
        ...         with st.chat_message("ai"):
        ...             voice.render_message("Hello!")
    """

    def __init__(self, stt: SpeechToText | None = None, tts: TextToSpeech | None = None):
        """初始化 VoiceManager。

        Args:
            stt: SpeechToText 实例（None 表示关闭 STT）
            tts: TextToSpeech 实例（None 表示关闭 TTS）
        """
        self.stt = stt
        self.tts = tts

        logger.info(
            f"VoiceManager: STT={'enabled' if stt else 'disabled'}, "
            f"TTS={'enabled' if tts else 'disabled'}"
        )

    @classmethod
    def from_env(cls) -> Optional["VoiceManager"]:
        """从环境变量创建 VoiceManager。

        读取：
        - `VOICE_STT_PROVIDER`：决定 STT provider（不设置则禁用 STT）
        - `VOICE_TTS_PROVIDER`：决定 TTS provider（不设置则禁用 TTS）

        Returns:
            若 STT/TTS 至少启用一个，则返回 VoiceManager；否则返回 None（即“语音功能未启用”）

        Example:
            >>> # In .env:
            >>> # VOICE_STT_PROVIDER=openai
            >>> # VOICE_TTS_PROVIDER=openai
            >>>
            >>> voice = VoiceManager.from_env()
            >>> # Returns configured VoiceManager or None if disabled
        """
        # 从环境变量创建 STT/TTS：内部会自行校验 provider 名称并处理异常。
        stt = SpeechToText.from_env()
        tts = TextToSpeech.from_env()

        # 如果两者都未配置，直接返回 None（上层可据此隐藏语音 UI）。
        if not stt and not tts:
            logger.debug("Voice features not configured")
            return None

        return cls(stt=stt, tts=tts)

    def _transcribe_audio(self, audio) -> str | None:
        """带 UI 反馈的语音转写封装。

        Args:
            audio: 来自 Streamlit chat_input 的音频对象（可能是文件-like 对象）

        Returns:
            转写后的文本；失败则返回 None（并在 UI 给出提示）
        """
        # 防御式校验：正常情况下只有在 stt 存在时才会调用。
        if not self.stt:
            st.error("⚠️ Speech-to-text not configured.")
            return None

        # 转写过程可能耗时（网络调用），用 spinner 提升交互体验。
        with st.spinner("🎤 Transcribing audio..."):
            transcribed = self.stt.transcribe(audio)

        # STT 失败时（返回空字符串），这里统一转成 UI 可理解的错误提示。
        if not transcribed:
            st.error("⚠️ Transcription failed. Please try again or type your message.")
            return None

        return transcribed

    def get_chat_input(self, placeholder: str = "Your message") -> str | None:
        """获取聊天输入：支持文本输入 + 可选语音输入（转写后返回文本）。

        Args:
            placeholder: 输入框占位符

        Returns:
            用户输入的文本（若是语音则返回转写结果）；没有输入则返回 None
        """
        # 未启用 STT：退化为普通的 `st.chat_input`（仅文本）。
        if not self.stt:
            return st.chat_input(placeholder)

        # 启用 STT：使用支持音频的 chat_input。
        chat_value = st.chat_input(placeholder, accept_audio=True)

        if not chat_value:
            return None

        # 如果返回的是 str，说明用户输入的是纯文本。
        if isinstance(chat_value, str):
            return chat_value

        # Streamlit 在 accept_audio=True 时，可能返回对象或 dict（不同版本/环境存在差异）。
        # 这里同时兼容 attribute 与 dict 两种访问方式。

        # 提取 text（如果用户同时输入了文本）
        text_content = None
        if hasattr(chat_value, "text"):
            text_content = chat_value.text
        elif isinstance(chat_value, dict):
            text_content = chat_value.get("text", "")

        # 提取 audio（如果用户提供了语音）
        audio_content = None
        if hasattr(chat_value, "audio"):
            audio_content = chat_value.audio
        elif isinstance(chat_value, dict):
            audio_content = chat_value.get("audio")

        # 优先处理语音：有音频就转写并返回（转写失败则返回 None）。
        if audio_content:
            return self._transcribe_audio(audio_content)

        # 没有音频，则返回文本内容。
        if text_content:
            return text_content

        # 既没有音频也没有文本：返回 None。
        return None

    def render_message(self, content: str, container=None, audio_only: bool = False) -> None:
        """渲染一条消息，并可选生成/播放 TTS 语音。

        Args:
            content: 要显示的消息文本
            container: Streamlit 容器（默认使用当前 `st` 上下文）
            audio_only: 为 True 时只渲染音频（常用于“文本已在 streaming 过程中显示”的场景）
        """
        if container is None:
            container = st

        # 默认渲染文本；如果处于 streaming 等场景，可能上层已经渲染过文本，则可设置 audio_only=True。
        if not audio_only:
            container.write(content)

        # 只有在启用 TTS 且文本非空时才生成音频（避免无意义的 API 调用）。
        if self.tts and content.strip():
            # 生成音频是网络调用：用 placeholder + caption 作为“生成中”反馈。
            placeholder = container.empty()
            with placeholder:
                st.caption("🎙️ Generating audio...")

            # 调用 TTS provider 生成音频 bytes（失败则返回 None）。
            audio = self.tts.generate(content)

            # 将最后一条 AI 消息的音频缓存到 session_state：
            # Streamlit 的 rerun 会重建 UI 树，但 session_state 能保留数据，避免音频丢失。
            if audio:
                st.session_state.last_audio = {"data": audio, "format": self.tts.get_format()}

            # 用音频播放器替换占位符；失败则展示降级提示。
            if audio:
                placeholder.audio(audio, format=self.tts.get_format())
            else:
                placeholder.caption("🔇 Audio generation unavailable")
