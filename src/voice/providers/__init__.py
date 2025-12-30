"""
本包在整个项目中的角色：Voice Provider 的实现集合。

设计意图：
- 上层（`SpeechToText` / `TextToSpeech`）通过动态 import 选择具体 provider；
- provider 层只负责“调用第三方语音 API/SDK”并返回统一格式的结果：
  - STT：`transcribe(audio_file) -> str`
  - TTS：`generate(text) -> bytes | None` + `get_format() -> str`

扩展点：
- 新增 provider 时，通常只需要：
  1) 在此目录新增实现文件（例如 `deepgram_stt.py` / `elevenlabs_tts.py`）
  2) 在 `src/voice/stt.py` 或 `src/voice/tts.py` 的 `_load_provider` 中加入分支
  （是否在这里 re-export 取决于你希望对外暴露哪些类）
"""

from voice.providers.openai_stt import OpenAISTT
from voice.providers.openai_tts import OpenAITTS

# 未来扩展的 provider 可以在这里导入并加入 __all__：
# from voice.providers.deepgram_stt import DeepgramSTT
# from voice.providers.elevenlabs_tts import ElevenLabsTTS

__all__ = ["OpenAISTT", "OpenAITTS"]
