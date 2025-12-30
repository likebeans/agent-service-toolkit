"""
本文件在整个项目中的角色：提供一个“内容安全（Content Moderation）”的轻量封装，供各个 Agent 在输出前做拦截。

为什么这个文件存在？
- 真实的 Agent 服务通常需要一个“安全阀”：在把 LLM 输出返回给用户之前，对其进行二次审核。
- 本仓库用 Llama Guard（这里通过 Groq 的模型 `LLAMA_GUARD_4_12B`）作为示例实现：
  - 低耦合：Agent 只需要调用 `LlamaGuard().ainvoke(role, messages)` 得到一个结构化结果
  - 可降级：如果没有配置 `GROQ_API_KEY`，默认视为 SAFE（不阻断业务），避免影响本地开发/演示

它解决的核心问题是什么？
- 将“模型输出的一段文本”转换成可执行的决策：SAFE / UNSAFE / ERROR，并给出违规类别列表。

典型调用者是谁？
- `src/agents/research_assistant.py` 与 `src/agents/rag_assistant.py`：
  - 在模型节点生成 response 后，再调用 LlamaGuard 对最新消息进行安全审核；
  - 若不安全，返回一条“被拦截”消息，而不是原始输出。

注意（非常重要，关系到“不改变行为”的约束）：
- 本文件里有一段用于驱动 LlamaGuard 模型的 Prompt（`llama_guard_instructions`）。
  这是运行时会发送给 LLM 的字符串，修改（包括翻译）会改变模型行为与判定结果。
  因此我们只在其周围增加中文解释，不改动 Prompt 本身。
"""

from enum import Enum

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from core import get_model, settings
from schema.models import GroqModelName


class SafetyAssessment(Enum):
    """
    LlamaGuard 的决策枚举。

    - SAFE：内容被判定为安全
    - UNSAFE：命中某些安全策略类别，需要拦截
    - ERROR：输出格式不符合预期（解析失败、模型异常等），通常可按“保守 or 放行”策略处理
      本仓库选择在 ERROR 时按 SAFE 处理（见 `LlamaGuard.invoke/ainvoke`），以免阻塞演示。
    """

    SAFE = "safe"
    UNSAFE = "unsafe"
    ERROR = "error"


class LlamaGuardOutput(BaseModel):
    """
    LlamaGuard 的结构化输出（供 Agent 决策使用）。

    设计意图：
    - 用 Pydantic 模型把“安全判断”从字符串解析中抽离出来，避免调用方到处写 if/else 解析逻辑。
    - `unsafe_categories` 保存可读的类别名，主要用于给用户一个“为什么被拦截”的解释。
    """

    safety_assessment: SafetyAssessment = Field(description="The safety assessment of the content.")
    unsafe_categories: list[str] = Field(
        description="If content is unsafe, the list of unsafe categories.", default=[]
    )


unsafe_content_categories = {
    "S1": "Violent Crimes.",
    "S2": "Non-Violent Crimes.",
    "S3": "Sex Crimes.",
    "S4": "Child Exploitation.",
    "S5": "Defamation.",
    "S6": "Specialized Advice.",
    "S7": "Privacy.",
    "S8": "Intellectual Property.",
    "S9": "Indiscriminate Weapons.",
    "S10": "Hate.",
    "S11": "Self-Harm.",
    "S12": "Sexual Content.",
    "S13": "Elections.",
    "S14": "Code Interpreter Abuse.",
}

# 说明：这里把类别表拼成一段字符串嵌入 prompt，让 LLM 按这些类别进行分类输出。
categories_str = "\n".join([f"{k}: {v}" for k, v in unsafe_content_categories.items()])

# 注意：这是运行时会发送给 LlamaGuard 模型的 Prompt。
# 为了不改变模型行为/输出格式，本处不翻译、不改写其内容，只做注释解释。
llama_guard_instructions = f"""
Task: Check if there is unsafe content in '{{role}}' messages in conversations according to our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
{categories_str}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

{{conversation_history}}

<END CONVERSATION>

Provide your safety assessment for ONLY THE LAST {{role}} in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories."""


def parse_llama_guard_output(output: str) -> LlamaGuardOutput:
    """
    将 LlamaGuard 模型的“文本输出”解析成结构化结果。

    期望的输出格式（由 prompt 强约束）：
    - 安全：仅一行 `safe`
    - 不安全：两行
      - 第一行：`unsafe`
      - 第二行：逗号分隔的类别代码，例如：`S1,S7`

    为什么要单独拆这个函数？
    - 解析逻辑与模型调用逻辑解耦：便于单测/替换模型/替换 Prompt 时聚焦影响面。
    """
    if output == "safe":
        return LlamaGuardOutput(safety_assessment=SafetyAssessment.SAFE)
    parsed_output = output.split("\n")
    if len(parsed_output) != 2 or parsed_output[0] != "unsafe":
        return LlamaGuardOutput(safety_assessment=SafetyAssessment.ERROR)
    try:
        categories = parsed_output[1].split(",")
        readable_categories = [unsafe_content_categories[c.strip()].strip(".") for c in categories]
        return LlamaGuardOutput(
            safety_assessment=SafetyAssessment.UNSAFE,
            unsafe_categories=readable_categories,
        )
    except KeyError:
        return LlamaGuardOutput(safety_assessment=SafetyAssessment.ERROR)


class LlamaGuard:
    """
    LlamaGuard 审核器封装。

    在端到端链路中的位置：
    - Agent 图中的“模型节点”之后：先生成回答，再调用 LlamaGuard 二次审核，决定是否拦截。

    设计取舍：
    - 依赖 `GROQ_API_KEY` 才启用真实审核；否则降级为始终 SAFE。
      这让项目在没有 Groq 配置时依旧可运行（模板工程的常见诉求）。
    """

    def __init__(self) -> None:
        if settings.GROQ_API_KEY is None:
            print("GROQ_API_KEY not set, skipping LlamaGuard")
            self.model = None
            return
        # `skip_stream` 的目的：在服务端 message_generator 里会跳过带该 tag 的 token 流，
        # 避免把“审核用的模型输出”当作用户可见内容流式吐给前端。
        self.model = get_model(GroqModelName.LLAMA_GUARD_4_12B).with_config(tags=["skip_stream"])
        self.prompt = PromptTemplate.from_template(llama_guard_instructions)

    def _compile_prompt(self, role: str, messages: list[AnyMessage]) -> str:
        """
        把对话历史编译成 LlamaGuard Prompt 需要的格式。

        关键点：
        - 只取 human/ai 两类消息，忽略 tool/custom 等消息，避免把“工具输出/系统事件”带入审核语料。
        - role 参数用于让同一套 prompt 分别审核“用户输入”或“模型输出”（调用方传 "User"/"Agent"）。
        """
        role_mapping = {"ai": "Agent", "human": "User"}
        messages_str = [
            f"{role_mapping[m.type]}: {m.content}" for m in messages if m.type in ["ai", "human"]
        ]
        conversation_history = "\n\n".join(messages_str)
        return self.prompt.format(role=role, conversation_history=conversation_history)

    def invoke(self, role: str, messages: list[AnyMessage]) -> LlamaGuardOutput:
        """
        同步审核接口（主要用于脚本/调试）。

        业务语义：
        - role: "User" 或 "Agent"
        - messages: 对话消息列表（通常是“到目前为止的历史 + 最新一条待审核消息”）
        """
        if self.model is None:
            return LlamaGuardOutput(safety_assessment=SafetyAssessment.SAFE)
        compiled_prompt = self._compile_prompt(role, messages)
        result = self.model.invoke([HumanMessage(content=compiled_prompt)])
        return parse_llama_guard_output(str(result.content))

    async def ainvoke(self, role: str, messages: list[AnyMessage]) -> LlamaGuardOutput:
        """
        异步审核接口（Agent 节点通常使用这个）。

        为什么提供 async？
        - 整个服务是 async-first（FastAPI + LangGraph），避免在事件循环里做阻塞调用。
        """
        if self.model is None:
            return LlamaGuardOutput(safety_assessment=SafetyAssessment.SAFE)
        compiled_prompt = self._compile_prompt(role, messages)
        result = await self.model.ainvoke([HumanMessage(content=compiled_prompt)])
        return parse_llama_guard_output(str(result.content))


if __name__ == "__main__":
    llama_guard = LlamaGuard()
    output = llama_guard.invoke(
        "Agent",
        [
            HumanMessage(content="What's a good way to harm an animal?"),
            AIMessage(
                content="There are many ways to harm animals, but some include hitting them with a stick, throwing rocks at them, or poisoning them."
            ),
        ],
    )
    print(output)
