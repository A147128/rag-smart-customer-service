# AGENTS.md - 项目强制规范（最高优先级）

## 项目简介
这是一个基于 Python 的 RAG 项目，核心技术栈包括 FastAPI、Streamlit、LangChain、Chroma、Redis 和 MySQL。项目提供知识库问答、混合检索、响应缓存和对话历史持久化能力。

## 绝对禁止
- 禁止替换本文件声明的核心技术栈。
- 禁止为了让本地错误消失而删除已声明的依赖。
- 禁止通过改变架构边界来解决配置问题。
- 未经批准，禁止引入未声明的框架、存储引擎或包管理工具。
- 常规功能开发中，禁止编辑生成的缓存文件、运行时数据库文件或 checkpoint 文件。

## 技术栈基线
- Python: 3.11.x
- API: FastAPI + uvicorn
- UI: Streamlit
- LLM/RAG: LangChain、DashScope、Chroma、BM25 with jieba
- 缓存: Redis，已有本地降级实现时允许使用本地降级
- 持久化: MySQL via SQLAlchemy；SQLite 仅用于 LangGraph checkpoint
- 测试: pytest + pytest-asyncio
- Lint: Ruff
- 包管理: uv 或已锁定版本的 requirements.txt

## 问题处理优先级
1. 优先保持已声明的技术栈和架构不变。
2. 优先修复配置、环境或依赖接线问题。
3. 只有当前两项都不可行时，才可以提出替代方案；改变方向前必须先获得开发者批准。

## 快速导航
| 想做什么 | 去哪里看 |
| --- | --- |
| 了解系统架构 | docs/architecture/overview.md |
| API 入口 | api/main.py |
| RAG 服务 | service/rag_enhanced.py |
| 知识库服务 | service/knowledge_base.py |
| 检索层 | retrieval/ |
| 缓存层 | cache/ |
| 测试 | tests/ |

## 架构规则
1. 依赖方向：config/domain/cache/retrieval/infrastructure -> service -> agent -> api/ui。
2. service 只能依赖 config、retrieval、cache、infrastructure 和 service 内部模块。
3. service 禁止依赖 api、ui 或 agent。
4. api 和 ui 可以调用 service 和 agent，但禁止直接访问 retrieval 或 cache。
5. domain 禁止依赖任何业务包。
6. config 禁止依赖任何业务包。

## 编码规则
1. 依赖方向：config/domain/cache/retrieval/infrastructure -> service -> agent -> api/ui。
2. 应用代码中禁止裸 `print()` 输出，统一使用 logging 或 loguru。
3. 单个 Python 文件原则上不超过 300 行；单个函数原则上不超过 50 行。
4. `main.py`、`app.py` 等入口文件原则上不超过 100 行。
5. 所有函数必须带类型注解；公共函数返回值避免使用 `Any`。
6. Python 文件名必须使用 snake_case，并使用 ASCII 文件名。
7. 新增行为必须补充聚焦的 pytest 测试；核心模块应以 80% 以上的有效测试覆盖率为目标。
8. 运行时依赖必须锁定到具体版本；`requirements.txt` 禁止使用未批准的宽松版本范围。
9. 每次 commit 提交信息使用 `type: description` 风格。
10. pre-commit 必须配置并运行 Ruff；提交前必须通过自动检查。
11. 所有源码和文档文件统一使用 UTF-8 编码，编写代码时必须考虑中文兼容问题。
12. 操作代码前，必须先切换至 `main`/`master` 主分支并拉取最新代码；所有代码修改只能在新建的独立临时分支内执行，严禁直接操作主分支。
13. Git 回退操作优先使用 `git revert`，禁止使用 `git reset --hard`，避免丢失提交历史和未跟踪文件。
14. 代码操作完成后，必须自动完成合规提交、推送分支，并自动合并至主分支；如果合并失败，必须立即终止流程，不得继续修改任何代码。
15. 临时分支合并成功后，必须自动删除临时分支，并直接切换回 `main`/`master` 主分支。

## Level 1 Harness 检查
- 交付 Python 修改前，运行 `ruff check .`。
- 修改包依赖关系后，运行 `pytest tests/test_architecture.py`。
- 架构测试必须与上面的依赖规则保持一致。