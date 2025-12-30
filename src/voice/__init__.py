"""
本包在整个项目中的角色：语音输入/输出（Voice I/O）能力的聚合入口。

它解决的核心问题是什么？
- 在聊天式 Agent 服务中提供可选的语音能力：
  - STT（Speech-to-Text）：把用户语音转成文本输入
  - TTS（Text-to-Speech）：把 Agent 的文本回复转成语音播放

设计意图（对学习者最重要）：
- 将“业务逻辑（STT/TTS）”与“UI 框架（Streamlit）”解耦：
  - `SpeechToText` / `TextToSpeech`：纯能力模块，可在任意 Python 环境使用
  - `VoiceManager`：只在这里依赖 Streamlit，负责 UI 交互与状态管理
- 多 provider 扩展：通过工厂类按配置加载不同 provider 实现（例如 OpenAI / 未来的 Deepgram/ElevenLabs）。

快速开始（Streamlit 场景）：
    >>> from voice import VoiceManager
    >>>
    >>> # 推荐方式：从环境变量创建（不配置则返回 None，相当于关闭语音功能）
    >>> voice = VoiceManager.from_env()
    >>>
    >>> if voice:
    ...     user_input = voice.get_chat_input()
    ...     # ... 用 user_input 继续走 agent 调用链路 ...
    ...     with st.chat_message("ai"):
    ...         voice.render_message(response)

进阶用法（手动组合 provider）：
    >>> from voice import SpeechToText, TextToSpeech, VoiceManager
    >>>
    >>> stt = SpeechToText(provider="openai")
    >>> tts = TextToSpeech(provider="openai", voice="nova")
    >>> voice = VoiceManager(stt=stt, tts=tts)
"""

from voice.manager import VoiceManager
from voice.stt import SpeechToText
from voice.tts import TextToSpeech

__all__ = ["VoiceManager", "SpeechToText", "TextToSpeech"]
