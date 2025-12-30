"""
本文件在整个项目中的角色：为 `bg_task_agent` 提供一个最小的“任务状态/进度”封装，并通过 custom 事件流式上报。

为什么这个文件存在？
- `src/agents/bg_task_agent/bg_task_agent.py` 想演示“后台任务进度”如何被 UI 感知；
  但如果把状态编码、消息格式、writer 发送细节都写在节点函数里，会让学习者抓不住重点。
- 因此这里抽象出 `Task`：
  - 负责维护 task 的生命周期状态（new/running/complete + success/error）
  - 把状态与附加数据打包成 `TaskData`（`src/schema/task_data.py`）
  - 再用 `agents.utils.CustomData` 统一编码成 `role="custom"` 的 LangChain message 并通过 StreamWriter 推送

端到端链路（与 UI 的协作）：
Task.start()/write_data()/finish()
  -> 生成 custom message（携带 TaskData 的 dict）
  -> 服务端流式透传
  -> Streamlit 端 `TaskDataStatus.add_and_draw_task_data()` 解析并渲染成状态块

注意（不改变行为的约束）：
- 本文件只是对消息封装与派发的“薄封装”，我们只加注释，不改变任何字段/状态机/返回值。
"""

from typing import Literal
from uuid import uuid4

from langchain_core.messages import BaseMessage
from langgraph.types import StreamWriter

from agents.utils import CustomData
from schema.task_data import TaskData


class Task:
    """
    一个“可被流式展示”的后台任务抽象。

    设计意图：
    - 把任务生命周期的状态变更（start/running/finish）与消息派发绑定在一起，
      调用方（图节点）只需要按业务节奏调用这些方法即可。

    生命周期（典型）：
    - start() -> write_data() (可多次) -> finish()

    与其它模块协作：
    - `schema.task_data.TaskData`：定义 UI 侧可解析的结构化数据模型
    - `agents.utils.CustomData`：把结构化 dict 编码为 LangChain custom 消息
    - `langgraph.types.StreamWriter`：把消息写入流（供 service/UI 消费）
    """

    def __init__(self, task_name: str, writer: StreamWriter | None = None) -> None:
        self.name = task_name
        self.id = str(uuid4())
        self.state: Literal["new", "running", "complete"] = "new"
        self.result: Literal["success", "error"] | None = None
        self.writer = writer

    def _generate_and_dispatch_message(self, writer: StreamWriter | None, data: dict):
        """
        生成并（可选）派发一条任务状态消息。

        为什么这是“内部方法”？
        - start/write_data/finish 三个公开方法只是更新状态字段不同，但编码/派发逻辑完全一致；
          提取公共逻辑可避免学习者被重复代码干扰。

        返回值：
        - 返回一个 LangChain message（BaseMessage），便于调用方在必要时也能把消息写入 state。
          （当前 bg_task_agent 主要依赖 writer 流式派发，这个返回值更多是扩展点）
        """
        writer = writer or self.writer
        task_data = TaskData(name=self.name, run_id=self.id, state=self.state, data=data)
        if self.result:
            task_data.result = self.result
        # 注意：CustomData 的职责只是“把 dict 装进 role=custom 的消息里”，
        # 具体结构由 TaskData 决定；UI 侧会用 TaskData.model_validate(...) 解析该 dict。
        task_custom_data = CustomData(
            type=self.name,
            data=task_data.model_dump(),
        )
        if writer:
            task_custom_data.dispatch(writer)
        return task_custom_data.to_langchain()

    def start(self, writer: StreamWriter | None = None, data: dict = {}) -> BaseMessage:
        """
        任务开始（进入 new 状态）。

        设计取舍：
        - 这里把 start 设为 "new" 而不是 "running"，是为了在 UI 上区分“刚开始”与“持续运行中”的展示。
        """
        self.state = "new"
        task_message = self._generate_and_dispatch_message(writer, data)
        return task_message

    def write_data(self, writer: StreamWriter | None = None, data: dict = {}) -> BaseMessage:
        """
        任务运行中输出增量数据（进入 running 状态）。

        关键约束：
        - 已完成的任务不允许再写数据（否则 UI 会出现状态回退或语义混乱），因此这里做显式校验。
        """
        if self.state == "complete":
            raise ValueError("Only incomplete tasks can output data.")
        self.state = "running"
        task_message = self._generate_and_dispatch_message(writer, data)
        return task_message

    def finish(
        self,
        result: Literal["success", "error"],
        writer: StreamWriter | None = None,
        data: dict = {},
    ) -> BaseMessage:
        """
        任务结束（进入 complete 状态），并记录结果 success/error。

        业务语义：
        - result 用于让 UI 决定最终是 green/ red 状态。
        - data 可携带输出（例如计算结果、错误详情等）。
        """
        self.state = "complete"
        self.result = result
        task_message = self._generate_and_dispatch_message(writer, data)
        return task_message
