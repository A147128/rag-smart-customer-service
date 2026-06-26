# 架构总览
## 包结构
P4_RAG项目案例/
├── domain/              # 领域模型（不依赖任何业务包，当前未实际使用）
│   └── model/          # Pydantic 数据模型（BaseModel）
├── config/             # 配置管理（零依赖）
│   └── config_data.py  # 全局配置常量（大模型、Redis、MySQL 等）
├── retrieval/          # 检索层
│   ├── vector_stores.py     # 向量存储服务
│   └── hybrid_retriever.py  # 混合检索（BM25 + 向量）
├── cache/              # 缓存层（零依赖）
│   ├── cache_compat.py     # Redis 缓存（同步接口）
│   ├── cache_manager.py    # 异步缓存管理器
│   └── cache_store.py      # 缓存存储抽象
├── service/            # 业务逻辑层
│   ├── rag_enhanced.py         # RAG 执行引擎（核心）
│   ├── knowledge_base.py        # 知识库管理
│   └── mysql_history_store.py   # MySQL 对话历史持久化
├── agent/              # LangGraph Agent 工作流
│   ├── state.py        # OutfitState 状态定义
│   ├── nodes.py        # 7 个图节点函数
│   ├── edges.py        # 条件路由 + HITL 反馈处理
│   ├── graph.py        # 图构建（同步 build_graph + 异步 build_async_graph）
│   └── tools.py        # 知识库检索 + 库存查询工具
├── api/                # 接口层
│   ├── main.py          # FastAPI 入口
│   └── agent_routes.py  # Agent API 路由（chat + resume）
├── ui/                 # 前端界面
│   ├── app_qa.py        # Streamlit 问答界面（RAG + Agent 双模式）
│   └── app_file_uploader.py  # 文件上传界面（已弃用）
├── infrastructure/     # 横切关注点
│   ├── benchmark.py      # 性能测试
│   └── database.py       # 数据库连接管理
└── tests/              # 测试
    └── test_agent.py    # Agent 模块测试

## 依赖规则
domain                → 不依赖任何业务包
config                → 不依赖任何业务包
retrieval             → 仅依赖 config
cache                 → 不依赖任何业务包
service               → 仅依赖 config、retrieval、cache、infrastructure
agent                 → 仅依赖 config、service
api                   → 仅依赖 service、agent（不依赖 retrieval、cache）
ui                    → 仅依赖 config、service、agent（不依赖 retrieval、cache）
infrastructure        → 仅依赖 config、service
tests                 → 仅依赖 agent
