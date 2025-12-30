"""
本文件在整个项目中的角色：演示 `langgraph-supervisor` 的“分层/嵌套（hierarchy）”多 Agent 协作模式。

为什么这个文件存在？
- 有些复杂任务需要多层编排：
  - 顶层 supervisor 负责总体路由与用户交互
  - 中层 supervisor/agent 负责某个领域（例如研究）
  - 底层 agent 负责更细的专长（例如数学）
- 这个示例把 `langgraph_supervisor_agent.py` 的两层结构扩展成三层：
  - 顶层 Supervisor -> 研究子 Agent（其内部又是一个 Supervisor） -> 数学子 Agent

端到端可观测性：
- `add_handoff_back_messages=True` + `output_mode="full_history"` 让 UI 能稳定显示：
  - transfer_to_xxx / transfer_back_to_xxx 这样的交接消息
  - 子 Agent 的完整输出历史（用于回放会话）

注意（不改变行为的约束）：
- prompt/system_prompt 字符串会直接发给 LLM；本文件只加中文解释，不改动其内容。
"""

from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor

from agents.langgraph_supervisor_agent import add, multiply, web_search
from core import get_model, settings

# 与 `langgraph_supervisor_agent.py` 一样：这里在 import 阶段固定绑定 DEFAULT_MODEL（示例取舍）。
model = get_model(settings.DEFAULT_MODEL)


def workflow(chosen_model):
    """
    构建一个“嵌套 supervisor”工作流。

    设计意图：
    - 把 chosen_model 显式作为参数传入，便于测试时替换 Fake 模型（见 `tests/service/test_service_message_generator.py`）。
    - 返回的是一个 supervisor builder，调用方再 `.compile()` 得到最终可执行图。
    """
    math_agent = create_agent(
        model=chosen_model,
        tools=[add, multiply],
        name="sub-agent-math_expert",  # 标记该节点为 sub-agent，便于服务端/UI 识别
        system_prompt="You are a math expert. Always use one tool at a time.",
    ).with_config(tags=["skip_stream"])

    research_agent = (
        create_supervisor(
            [math_agent],
            model=chosen_model,
            tools=[web_search],
            prompt="You are a world class researcher with access to web search. Do not do any math, you have a math expert for that. ",
            supervisor_name="supervisor-research_expert",  # 标记该节点为 supervisor（管理 math_agent）
        )
        .compile(
            name="sub-agent-research_expert"
        )  # 将该子图命名为 sub-agent，作为顶层 supervisor 的一个子 agent
        .with_config(tags=["skip_stream"])
    )  # 对 sub-agents 忽略 token 流（UI 只展示结构化消息/工具交接），避免噪音与渲染复杂度

    # 创建顶层 supervisor：只管理 research_agent（但 research_agent 内部又能调用 math_agent）
    return create_supervisor(
        [research_agent],
        model=chosen_model,
        prompt=(
            "You are a team supervisor managing a research expert with math capabilities."
            "For current events, use research_agent. "
        ),
        add_handoff_back_messages=True,
        # UI 侧依赖该开关来可靠判断“handoff back”边界，避免前端猜测何时交还控制权
        output_mode="full_history",  # 否则会话回放/重载时可能缺失子 Agent 的历史消息
    )  # 顶层 supervisor 默认名为 "supervisor"。


langgraph_supervisor_hierarchy_agent = workflow(model).compile()
