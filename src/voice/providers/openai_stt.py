"""
本文件在整个项目中的角色：OpenAI Whisper 的 STT Provider 实现。

它解决的核心问题是什么？
- 把 OpenAI 的语音转写 API 封装成一个最小接口：`transcribe(audio_file) -> str`
- 屏蔽 SDK 细节与异常处理策略，让上层（`SpeechToText` / `VoiceManager`）只关心：
  - 成功：返回转写文本
  - 失败：返回空字符串（上层可据此做 UI 提示/降级）

注意（工程取舍）：
- 本实现选择“记录错误但不抛异常”，以避免语音能力影响主链路（聊天仍可继续用文本输入）。
"""

import logging
from typing import BinaryIO

from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAISTT:
    """OpenAI Whisper STT provider（最小实现：提供 `transcribe()`）。"""

    def __init__(self, api_key: str | None = None):
        """初始化 OpenAI STT provider。

        Args:
            api_key: OpenAI API key（不传则由 OpenAI SDK 自行从环境变量读取）

        Raises:
            Exception: OpenAI client 初始化失败时抛出（属于“启动即失败”的配置问题）
        """
        # 创建 OpenAI 客户端：显式传入 api_key 便于在测试/多账号场景中覆盖。
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        logger.info("OpenAI STT initialized")

    def transcribe(self, audio_file: BinaryIO) -> str:
        """调用 OpenAI Whisper 进行语音转写。

        Args:
            audio_file: 二进制音频文件（file-like）

        Returns:
            转写后的文本（失败返回空字符串）

        Note:
            - 错误只记录日志，不向上抛出：上层可基于空字符串做“优雅降级”。
            - 这对用户侧应用更友好：语音失败不应影响整体对话流程。
        """
        try:
            # 保险起见重置文件指针：音频对象可能在别处被读取过。
            audio_file.seek(0)

            # 调用 OpenAI Whisper API（response_format="text" 返回纯文本）。
            result = self.client.audio.transcriptions.create(
                model="whisper-1", file=audio_file, response_format="text"
            )

            # 做一次简单清理（去掉首尾空白），并记录日志便于观测效果。
            transcribed = result.strip()
            logger.info(f"OpenAI STT: transcribed {len(transcribed)} chars")
            return transcribed

        except Exception as e:
            # 记录完整堆栈便于排障（但不让异常向上传播）。
            logger.error(f"OpenAI STT failed: {e}", exc_info=True)
            # 返回空字符串以允许上层进行 UI 提示/降级处理。
            return ""
