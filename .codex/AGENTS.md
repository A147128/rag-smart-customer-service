# AGENTS.md

## 项目简介
基于 LangChain + LangGraph 双引擎驱动的智能穿搭顾问系统，提供两种交互模式：

- **RAG 问答模式**：基于 Chroma 向量库的混合检索（BM25 + 向量），配合 Redis 响应缓存，
  实现高效精准的知识库问答和连贯的多轮对话。
- **Agent 导购模式**：基于 LangGraph 有状态工作流，通过需求收集 → 用户画像 → 并行检索 →
  方案生成 → 反馈优化的闭环流程，结合 HITL（人在回路中）机制，提供个性化的穿搭推荐体验。

## 技术栈基线（不允许擅自升级）
框架
├── FastAPI + uvicorn          # API 服务
├── Streamlit                  # 前端 UI
└── LangGraph                  # Agent 工作流编排

AI/LLM
├── langchain + community      # LLM 调用链
├── dashscope (通义千问)        # 底层模型
└── langgraph                  # 有状态图工作流

检索层
├── Chroma                     # 向量数据库
├── BM25 (jieba 分词)          # 关键词检索
└── 混合检索策略                # 向量 + 关键词融合

缓存层
├── Redis                      # 分布式缓存
└── 文件缓存                    # 本地降级

持久化
├── MySQL (SQLAlchemy + pymysql)  # 历史记录
└── SQLite (LangGraph checkpoint) # Agent 状态快照

工具链
├── loguru                     # 结构化日志
├── pytest + pytest-asyncio    # 测试
├── ruff + mypy                # 代码检查
└── pre-commit                 # 提交前钩子

## 快速导航
| 你想做什么 | 去哪里看 |
|-----------|---------|
| 了解系统架构 | docs/architecture/overview.md |


## 硬性规则
1. 依赖方向：config → retrieval/cache/infrastructure → service → agent → api/ui
2. 禁止 print() 裸输出，统一 logging / loguru 结构化日志
3. 单文件 ≤ 300 行；单函数 ≤ 50 行
4. 入口文件 ≤ 100 行（main.py / app.py）
5. 所有函数必须有类型注解，返回值不能裸 -> Any
6. .py 文件名全部 snake_case，禁止中文文件名
7. 新增功能必须有 pytest 测试，核心模块覆盖率 ≥ 80%
8. 依赖必须锁定版本，requirements.txt 禁止 >= 裸宽松约束
9. 每轮 git commit 只做一件事，message 用 type: description 格式
10. pre-commit 必须配置 ruff + mypy，提交前自动检查
11. 编写代码必须考虑中文兼容问题，使用 utf-8 编码。
