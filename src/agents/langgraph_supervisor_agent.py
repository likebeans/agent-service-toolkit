"""
本文件在整个项目中的角色：演示 `langgraph-supervisor` 的“多 Agent 协作/任务分派（Supervisor）”能力。

为什么这个文件存在？
- 单一 Agent 很快会遇到两个工程痛点：
  1) 工具/技能过多导致提示词复杂、决策不稳定
  2) 不同任务类型（研究/数学/编码/规划）需要不同“专长配置”
- `langgraph-supervisor` 提供一个 Supervisor 图：
  - Supervisor 负责决定把任务交给哪个子 Agent（sub-agent）
  - 子 Agent 只关注自己的工具与专长（更简单、更稳定）

本文件构建了两个子 Agent：
- research_agent：只能做 web_search（示例写死返回值）
- math_agent：只能做 add/multiply
然后用 supervisor 把任务分派给它们。

典型调用者是谁？
- `src/agents/agents.py`：注册 `"langgraph-supervisor-agent"`
- 服务端/前端：流式时会看到“handoff / handback”的工具调用消息（服务端有特殊处理逻辑，见 `src/service/service.py:message_generator()`）

注意（不改变行为的约束）：
- `system_prompt`/`prompt` 字符串会直接发给 LLM；本文件只加中文解释，不改动其内容。
- `add/multiply/web_search` 的 docstring 会作为“工具描述”影响 LLM 选择工具；因此不翻译这些 docstring。
"""

from typing import Any

from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor

from core import get_model, settings

# 注意：这里在 import 阶段固定绑定了 DEFAULT_MODEL。
# 这意味着它不像其它 Agent 那样“按请求从 config 读取 model”；这是示例取舍（简单优先）。
model = get_model(settings.DEFAULT_MODEL)


def add(a: float, b: float) -> float:
    # 注意：docstring 会作为工具描述发给 LLM；不翻译以避免改变行为。
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float) -> float:
    # 注意：docstring 会作为工具描述发给 LLM；不翻译以避免改变行为。
    """Multiply two numbers."""
    return a * b


def web_search(query: str) -> str:
    # 注意：docstring 会作为工具描述发给 LLM；不翻译以避免改变行为。
    """Search the web for information."""
    return (
        "Here are the headcounts for each of the FAANG companies in 2024:\n"
        "1. **Facebook (Meta)**: 67,317 employees.\n"
        "2. **Apple**: 164,000 employees.\n"
        "3. **Amazon**: 1,551,000 employees.\n"
        "4. **Netflix**: 14,000 employees.\n"
        "5. **Google (Alphabet)**: 181,269 employees."
    )


# 子 Agent 1：数学专家（只允许使用 add/multiply 工具）
math_agent: Any = create_agent(
    model=model,
    tools=[add, multiply],
    name="sub-agent-math_expert",
    system_prompt="You are a math expert. Always use one tool at a time.",
).with_config(tags=["skip_stream"])

# 子 Agent 2：研究专家（只允许使用 web_search 工具）
research_agent: Any = create_agent(
    model=model,
    tools=[web_search],
    name="sub-agent-research_expert",
    system_prompt="You are a world class researcher with access to web search. Do not do any math.",
).with_config(tags=["skip_stream"])


# 创建 supervisor 工作流：
# - prompt 告诉 supervisor 如何在两个专家之间分派任务
# - add_handoff_back_messages=True 会在“交接/交还”时插入工具消息，便于 UI 可靠判断边界
workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. "
        "For math problems, use math_agent."
    ),
    add_handoff_back_messages=True,
    # UI 侧依赖该开关来可靠判断“handoff back”边界，避免前端猜测何时交还控制权
    output_mode="full_history",  # 否则会话回放/重载时可能缺失子 Agent 的历史消息
)

langgraph_supervisor_agent = workflow.compile()
