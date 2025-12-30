# AGENTS.md - Schema 模块

本模块定义了 Agent Service Toolkit 的核心数据模型和类型系统。

## 模块概述

`schema` 模块是整个项目的数据定义层，提供：

- **LLM 模型枚举** (`models.py`)：支持的所有 LLM 提供商和模型
- **核心数据模型** (`schema.py`)：用户输入、消息、反馈等 Pydantic 模型
- **任务状态管理** (`task_data.py`)：Streamlit UI 中的任务状态追踪

## 文件结构

```
schema/
├── __init__.py      # 模块导出和公共 API
├── models.py        # LLM 模型枚举定义
├── schema.py        # 核心 Pydantic 数据模型
└── task_data.py     # Streamlit 任务状态管理
```

## 代码规范

- 使用 **Pydantic v2** 进行数据验证和序列化
- 所有模型必须包含完整的**中文文档字符串**
- 字段使用 `Field()` 并提供 `description` 和 `examples`
- 使用 `Literal` 类型约束固定值字段
- 使用 `TypedDict` 定义轻量级数据结构

## 类型定义指南

### 添加新的 LLM 模型

1. 在 `models.py` 中找到对应的提供商枚举类（如 `OpenAIModelName`）
2. 添加新的枚举值，确保值与 API 文档一致
3. 添加中文注释说明模型用途
4. 如果是新提供商，需创建新的枚举类并添加到 `AllModelEnum` 联合类型

```python
class NewProviderModelName(StrEnum):
    """
    新提供商模型名称枚举
    
    参考文档: https://docs.provider.com/models
    """
    MODEL_X = "model-x"  # 模型 X，适用于 XXX 场景
```

### 添加新的数据模型

1. 在 `schema.py` 中定义 Pydantic 模型
2. 继承 `BaseModel` 并添加完整的类文档字符串
3. 每个字段使用 `Field()` 提供元数据
4. 在 `__init__.py` 中导出新模型

```python
class NewModel(BaseModel):
    """
    新模型说明
    
    Attributes:
        field_name: 字段说明
    """
    field_name: str = Field(
        description="字段的详细描述",
        examples=["示例值"],
    )
```

## 测试指南

验证模块可正常导入：

```bash
cd src
python -c "from schema import *; print('OK')"
```

验证类型检查（如果配置了 mypy）：

```bash
mypy src/schema/
```

## 常见修改场景

| 场景 | 操作位置 |
|------|----------|
| 添加新 LLM 模型 | `models.py` 对应的枚举类 |
| 添加新提供商 | `models.py` 新建枚举类 + 更新 `AllModelEnum` |
| 添加 API 请求/响应模型 | `schema.py` |
| 修改 Streamlit 任务显示 | `task_data.py` |

## 注意事项

- **不要删除** 已有的模型枚举值，可能导致向后不兼容
- 修改 `Field()` 的 `description` 会影响 API 文档生成
- `task_data.py` 依赖 Streamlit，仅在 Streamlit 应用中使用
- 所有注释和文档字符串使用**中文**
