# 🏘️ 桃源镇 - AI 小镇模拟

基于 **LangGraph** 的多 Agent 小镇生活模拟项目。灵感来自斯坦福论文 [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)。

## 🎯 项目亮点

- **LangGraph 状态图编排** — perceive → plan → act → reflect 循环
- **多 Agent 协作** — 5 个性格各异的小镇居民自主生活
- **三层记忆架构** — 观察记忆 / 反思记忆 / 计划记忆
- **动态交互** — 同一地点的居民会自然对话
- **Web 可视化** — 实时查看小镇动态

## 🏗️ 架构

```
LangGraph StateGraph 工作流:

  ┌──────────┐    ┌──────┐    ┌──────┐    ┌─────────┐
  │ perceive │───>│ plan │───>│ act  │───>│ reflect │──> END
  └──────────┘    └──────┘    └──┬───┘    └─────────┘
       ^                        │
       │     (day not ended)    │
       └────────────────────────┘
```

### 核心模块

| 模块 | 说明 |
|------|------|
| `agents/memory.py` | 三层记忆系统（观察/反思/计划） |
| `agents/resident.py` | 居民 Agent 定义 |
| `agents/profiles.py` | 5 个预设角色档案 |
| `simulation/engine.py` | LangGraph 模拟引擎（核心） |
| `simulation/interactions.py` | LLM 交互逻辑（对话、计划、反思） |
| `town/environment.py` | 小镇环境与地点管理 |
| `web/app.py` | FastAPI Web 服务 + WebSocket |

## 🚀 快速开始

### 1. 安装依赖

```bash
cd langgraph-multi-agent
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 OPENAI_API_KEY
```

支持任何 OpenAI 兼容 API（DeepSeek、智谱、本地 Ollama 等），修改 `.env` 中的 `OPENAI_API_BASE` 和 `MODEL_NAME` 即可。

### 3. 运行

```bash
# Web 可视化模式（推荐）
python main.py web

# 终端模式 - 运行一天
python main.py console

# 终端单步模式
python main.py step
```

## 🏘️ 小镇居民

| 居民 | 职业 | 性格 |
|------|------|------|
| 老王 | 面馆老板 | 热情豪爽，爱聊天 |
| 小李 | 远程程序员 | 内向安静，技术宅 |
| 张医生 | 卫生所医生 | 沉稳可靠，暖心 |
| 陈老师 | 小学语文老师 | 温柔耐心，爱读书 |
| 赵大姐 | 超市老板娘 | 热心肠，消息灵通 |

## 🧠 技术要点（面试用）

### 1. LangGraph 状态图
- 使用 `StateGraph` 定义 perceive → plan → act → reflect 工作流
- 条件边（conditional edges）控制是继续下一步还是结束一天
- 状态在节点间以 `TypedDict` 传递

### 2. 记忆系统
- **观察记忆** — 短期，记录所见所闻
- **反思记忆** — 从多条观察中提炼高层次洞察
- **计划记忆** — 当天行动计划
- 检索使用 **重要性 + 时近性 + 相关性** 三维评分

### 3. Agent 设计模式
- 每个 Agent 有独立的性格、记忆、人际关系
- Agent 通过 LLM 自主决策（去哪里、做什么、和谁聊天）
- 对话基于双方性格和关系动态生成

### 4. 可扩展性
- 轻松添加新居民（修改 `profiles.py`）
- 轻松添加新地点（修改 `environment.py`）
- 支持切换不同 LLM 后端

## 📄 License

MIT
