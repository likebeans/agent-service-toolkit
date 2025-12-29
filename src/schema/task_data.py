"""
任务数据模型模块

本模块定义了用于追踪和显示 Agent 任务执行状态的数据模型。
主要用于 Streamlit 前端界面中展示任务的实时状态更新。

主要组件：
- TaskData: 表示单个任务实例的数据模型
- TaskDataStatus: 管理多个任务状态并在 Streamlit 中渲染的辅助类
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class TaskData(BaseModel):
    """
    任务数据模型
    
    表示 Agent 执行过程中一个任务实例的状态和数据。
    用于追踪任务的生命周期（新建 → 运行中 → 完成）和执行结果。
    
    Attributes:
        name: 任务名称，用于在 UI 中显示
        run_id: 任务运行 ID，用于将状态更新与特定任务实例关联
        state: 任务当前状态，可选值：
               - "new": 新创建的任务
               - "running": 正在执行中
               - "complete": 已完成
        result: 任务执行结果，可选值：
                - "success": 成功完成
                - "error": 执行出错
        data: 任务生成的额外数据，如输入参数、输出结果等
        
    Example:
        >>> task = TaskData(
        ...     name="检查输入安全性",
        ...     run_id="847c6285-8fc9-4560-a83f-4e6285809254",
        ...     state="running"
        ... )
    """
    name: str | None = Field(
        description="任务名称",
        default=None,
        examples=["检查输入安全性"]
    )
    run_id: str = Field(
        description="任务运行 ID，用于将状态更新与特定任务实例关联",
        default="",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    state: Literal["new", "running", "complete"] | None = Field(
        description="任务的当前状态",
        default=None,
        examples=["running"],
    )
    result: Literal["success", "error"] | None = Field(
        description="任务的执行结果",
        default=None,
        examples=["success"],
    )
    data: dict[str, Any] = Field(
        description="任务生成的额外数据",
        default={},
    )

    def completed(self) -> bool:
        """
        检查任务是否已完成
        
        Returns:
            如果任务状态为 "complete" 则返回 True，否则返回 False
        """
        return self.state == "complete"

    def completed_with_error(self) -> bool:
        """
        检查任务是否以错误状态完成
        
        Returns:
            如果任务已完成且结果为 "error" 则返回 True，否则返回 False
        """
        return self.state == "complete" and self.result == "error"


class TaskDataStatus:
    """
    任务状态管理器
    
    用于在 Streamlit 界面中管理和渲染多个任务的状态更新。
    使用 Streamlit 的 status 组件来展示任务的实时进度。
    
    Attributes:
        status: Streamlit 的 status 组件实例
        current_task_data: 当前追踪的所有任务数据，以 run_id 为键
        
    注意：
        此类依赖 Streamlit，只能在 Streamlit 应用中使用。
    """
    
    def __init__(self) -> None:
        """
        初始化任务状态管理器
        
        创建一个空的 Streamlit status 组件和任务数据字典。
        """
        import streamlit as st

        self.status = st.status("")
        self.current_task_data: dict[str, TaskData] = {}

    def add_and_draw_task_data(self, task_data: TaskData) -> None:
        """
        添加任务数据并在 Streamlit 中渲染状态更新
        
        根据任务的状态生成相应的状态消息，并更新 Streamlit 的 status 组件。
        同时追踪所有任务的完成状态，以确定整体进度。
        
        Args:
            task_data: 要添加和渲染的任务数据
            
        状态显示规则：
            - new: 显示 "任务已启动"（蓝色）
            - running: 显示 "任务写入"
            - complete + success: 显示 "成功完成"（绿色）
            - complete + error: 显示 "出错结束"（红色）
        """
        status = self.status
        
        # 根据任务状态生成状态消息
        status_str = f"任务 **{task_data.name}** "
        match task_data.state:
            case "new":
                status_str += "已 :blue[启动]。输入："
            case "running":
                status_str += "写入："
            case "complete":
                if task_data.result == "success":
                    status_str += ":green[成功完成]。输出："
                else:
                    status_str += ":red[出错结束]。输出："
        
        # 渲染状态信息和任务数据
        status.write(status_str)
        status.write(task_data.data)
        status.write("---")
        
        # 状态标签始终显示最后一个新启动的任务
        if task_data.run_id not in self.current_task_data:
            status.update(label=f"""任务: {task_data.name}""")
        
        # 更新任务追踪字典
        self.current_task_data[task_data.run_id] = task_data
        
        # 确定整体状态
        if all(entry.completed() for entry in self.current_task_data.values()):
            # 如果有任何任务出错，整体状态为 "error"
            if any(entry.completed_with_error() for entry in self.current_task_data.values()):
                state = "error"
            # 如果所有任务都成功完成，整体状态为 "complete"
            else:
                state = "complete"
        # 在所有任务完成之前，整体状态为 "running"
        else:
            state = "running"
        
        status.update(state=state)  # type: ignore[arg-type]
