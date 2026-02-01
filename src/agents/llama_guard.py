from enum import Enum

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from core import get_model, settings
from schema.models import GroqModelName


class SafetyAssessment(Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    ERROR = "error"


class LlamaGuardOutput(BaseModel):
    safety_assessment: SafetyAssessment = Field(description="内容的安全评估。")
    unsafe_categories: list[str] = Field(
        description="如果内容不安全，不安全类别的列表。", default=[]
    )


unsafe_content_categories = {
    "S1": "暴力犯罪。",
    "S2": "非暴力犯罪。",
    "S3": "性犯罪。",
    "S4": "儿童剥削。",
    "S5": "诽谤。",
    "S6": "专业建议。",
    "S7": "隐私。",
    "S8": "知识产权。",
    "S9": "无差别武器。",
    "S10": "仇恨。",
    "S11": "自残。",
    "S12": "性内容。",
    "S13": "选举。",
    "S14": "代码解释器滥用。",
}

categories_str = "\n".join([f"{k}: {v}" for k, v in unsafe_content_categories.items()])

llama_guard_instructions = f"""
任务：根据我们的安全策略检查对话中'{{role}}'消息是否包含以下内容类别的不安全内容。

<开始 不安全内容类别>
{categories_str}
<结束 不安全内容类别>

<开始 对话>

{{conversation_history}}

<结束 对话>

仅为上述对话中最后一个{{role}}提供你的安全评估：
- 第一行必须读取'安全'或'不安全'。
- 如果不安全，第二行必须包含违反类别的逗号分隔列表。"""


def parse_llama_guard_output(output: str) -> LlamaGuardOutput:
    """解析Llama Guard输出并返回安全评估。"""
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
    """Llama Guard安全检查类。"""

    def __init__(self) -> None:
        """初始化Llama Guard。"""
        if settings.GROQ_API_KEY is None:
            print("GROQ_API_KEY未设置，跳过LlamaGuard")
            self.model = None
            return
        self.model = get_model(GroqModelName.LLAMA_GUARD_4_12B).with_config(tags=["skip_stream"])
        self.prompt = PromptTemplate.from_template(llama_guard_instructions)

    def _compile_prompt(self, role: str, messages: list[AnyMessage]) -> str:
        """编译提示模板。"""
        role_mapping = {"ai": "代理", "human": "用户"}
        messages_str = [
            f"{role_mapping[m.type]}: {m.content}" for m in messages if m.type in ["ai", "human"]
        ]
        conversation_history = "\n\n".join(messages_str)
        return self.prompt.format(role=role, conversation_history=conversation_history)

    def invoke(self, role: str, messages: list[AnyMessage]) -> LlamaGuardOutput:
        """调用Llama Guard进行安全检查。"""
        if self.model is None:
            return LlamaGuardOutput(safety_assessment=SafetyAssessment.SAFE)
        compiled_prompt = self._compile_prompt(role, messages)
        result = self.model.invoke([HumanMessage(content=compiled_prompt)])
        return parse_llama_guard_output(str(result.content))

    async def ainvoke(self, role: str, messages: list[AnyMessage]) -> LlamaGuardOutput:
        """异步调用Llama Guard进行安全检查。"""
        if self.model is None:
            return LlamaGuardOutput(safety_assessment=SafetyAssessment.SAFE)
        compiled_prompt = self._compile_prompt(role, messages)
        result = await self.model.ainvoke([HumanMessage(content=compiled_prompt)])
        return parse_llama_guard_output(str(result.content))


if __name__ == "__main__":
    """主函数：测试Llama Guard功能。"""
    llama_guard = LlamaGuard()
    output = llama_guard.invoke(
        "代理",
        [
            HumanMessage(content="有什么伤害动物的好方法吗？"),
            AIMessage(
                content="有很多方法可以伤害动物，比如用棍子打它们、向它们扔石头或给它们下毒。"
            ),
        ],
    )
    print(output)
