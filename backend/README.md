# Deep Thinking Backend

多智能体推理后端。

## 功能特性

### 🧠 推理模式

- **ReAct** (Reason-Act-Observe): 迭代推理循环，适合需要工具或研究的任务
- **Chain of Thought**: 逐步推理，适合逻辑问题
- **Tree of Thoughts**: 探索性推理，适合创造性任务
- **Debate**: 多视角辩论，适合比较观点

### 🎯 思考深度

| 深度 | 模型层级 | 说明 |
|------|---------|------|
| off | small | 直接响应，无深度思考 |
| quick | small | 快速响应，最小推理 |
| standard | medium | 平衡的逐步推理 |
| deep | large | 多视角深度分析 |

### 🔧 核心能力

- ✅ 多智能体并行执行
- ✅ 任务自动分解
- ✅ 结果智能合成
- ✅ 反思质量提升
- ✅ 工具调用系统
- ✅ 流式响应支持

## 快速开始

### 安装

```bash
cd deep-thinking-backend
pip install -r requirements.txt
```

### 配置

复制环境变量示例文件并配置 API 密钥：

```bash
cp env.example.txt .env
```

编辑 `.env` 文件，设置至少一个提供商的 API 密钥：

```env
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
DEEPSEEK_API_KEY=sk-xxx
```

### 运行

```bash
python main.py
```

服务将在 http://localhost:8001 启动

## API 端点

### 基础对话

```http
POST /api/chat
Content-Type: application/json

{
  "messages": [{"role": "user", "content": "你的问题"}],
  "thinking_depth": "standard"
}
```

### 深度思考

```http
POST /api/thinking/think
Content-Type: application/json

{
  "query": "复杂问题",
  "depth": "deep",
  "mode": "auto"
}
```

### 流式深度思考

```http
POST /api/thinking/think/stream
Content-Type: application/json

{
  "query": "复杂问题",
  "depth": "standard",
  "mode": "react"
}
```

### 任务分解

```http
POST /api/thinking/decompose
Content-Type: application/json

{
  "query": "复杂多步骤任务",
  "available_tools": ["web_search", "calculator"]
}
```

### 结果合成

```http
POST /api/thinking/synthesize
Content-Type: application/json

{
  "query": "原始问题",
  "results": [{"response": "结果1"}, {"response": "结果2"}],
  "style": "comprehensive"
}
```

### OpenAI 兼容

```http
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "deep-thinking-deep",
  "messages": [{"role": "user", "content": "问题"}]
}
```

## 推理模式详解

### ReAct 模式

适用于需要多步骤推理和工具调用的任务：

```python
{
  "mode": "react",
  "depth": "standard"
}
```

工作流程：
1. THOUGHT: 分析当前状态
2. ACTION: 决定下一步行动
3. OBSERVATION: 观察结果
4. 重复直到得出答案

### Chain of Thought 模式

适用于需要逻辑推理的问题：

```python
{
  "mode": "cot",
  "depth": "standard"
}
```

工作流程：
1. 理解问题
2. 分步推理
3. 得出结论

### Tree of Thoughts 模式

适用于探索性和创造性任务：

```python
{
  "mode": "tot",
  "depth": "deep"
}
```

工作流程：
1. 生成多个思路分支
2. 评估每个分支
3. 剪枝低质量分支
4. 选择最佳路径

### Debate 模式

适用于需要多视角分析的问题：

```python
{
  "mode": "debate",
  "depth": "deep"
}
```

工作流程：
1. 多个智能体各持立场
2. 多轮辩论
3. 综合各方观点

## 架构

```
deep-thinking-backend/
├── app/
│   ├── api/
│   │   ├── chat.py          # 基础聊天 API
│   │   ├── health.py        # 健康检查
│   │   └── thinking.py      # 深度思考 API
│   ├── patterns/
│   │   ├── base.py          # 基础类和接口
│   │   ├── react.py         # ReAct 模式
│   │   ├── chain_of_thought.py  # CoT 模式
│   │   ├── tree_of_thoughts.py  # ToT 模式
│   │   ├── debate.py        # 辩论模式
│   │   ├── reflection.py    # 反思模式
│   │   ├── parallel.py      # 并行执行
│   │   ├── decomposition.py # 任务分解
│   │   └── synthesis.py     # 结果合成
│   ├── providers/
│   │   ├── base.py          # 提供商基类
│   │   ├── openai_provider.py
│   │   ├── anthropic_provider.py
│   │   └── deepseek_provider.py
│   ├── tools/
│   │   ├── base.py          # 工具基类
│   │   ├── builtin.py       # 内置工具
│   │   └── executor.py      # 工具执行器
│   ├── config.py            # 配置管理
│   ├── llm_manager.py       # LLM 管理器
│   └── thinking_engine.py   # 思考引擎
├── config/
│   └── models.yaml          # 模型配置
├── main.py                  # 应用入口
└── requirements.txt
```

## 许可证

MIT License
