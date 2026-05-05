# MindPilot 多模态智能科研助手 Agent 系统 — 项目深度解读文档

---

## 目录

1. [项目概述](#1-项目概述)
2. [目录结构全览](#2-目录结构全览)
3. [核心架构设计](#3-核心架构设计)
4. [完整执行流程（六步流水线）](#4-完整执行流程六步流水线)
5. [各模块详细解读](#5-各模块详细解读)
   - 5.1 [入口层：main.py](#51-入口层mainpy)
   - 5.2 [配置中心：config.py](#52-配置中心configpy)
   - 5.3 [中央编排器：orchestrator/orchestrator.py](#53-中央编排器orchestratororchestratorpy)
   - 5.4 [任务规划 Agent：agents/planning_agent.py](#54-任务规划-agentagentsplanning_agentpy)
   - 5.5 [文献检索 Agent：agents/literature_agent.py](#55-文献检索-agentagentsliterature_agentpy)
   - 5.6 [代码生成 Agent：agents/code_agent.py](#56-代码生成-agentagentscode_agentpy)
   - 5.7 [数据分析 Agent：agents/analysis_agent.py](#57-数据分析-agentagentsanalysis_agentpy)
   - 5.8 [评估反思 Agent：agents/evaluation_agent.py](#58-评估反思-agentagentsevaluation_agentpy)
   - 5.9 [工具层 tools/](#59-工具层-tools)
   - 5.10 [框架层 framework/](#510-框架层-framework)
   - 5.11 [记忆系统 memory/](#511-记忆系统-memory)
   - 5.12 [评估基准 evaluation/](#512-评估基准-evaluation)
6. [数据流与依赖关系图](#6-数据流与依赖关系图)
7. [LLM 调用机制详解](#7-llm-调用机制详解)
8. [Mock 模式与降级策略](#8-mock-模式与降级策略)
9. [修改指南：各模块如何入手修改](#9-修改指南各模块如何入手修改)
10. [关键设计决策与权衡](#10-关键设计决策与权衡)
11. [测试体系](#11-测试体系)

---

## 1. 项目概述

MindPilot 是一个 **多 Agent 协作的智能科研助手系统**。用户输入一个科研问题（如"研究 Transformer 注意力机制的计算复杂度优化方法"），系统会自动完成：

1. **任务规划** — 使用 Tree-of-Thought (ToT) + ReAct 策略，生成研究路径并分解为子任务
2. **文献检索** — 对接 ArXiv API 检索学术论文，构建知识图谱，生成结构化摘要
3. **实验设计** — 基于文献综述，自动生成完整的实验设计方案（假设、指标、基线）
4. **代码实现** — 自动生成 Python 代码并在沙箱中执行，支持自动调试闭环
5. **数据分析** — 自动 EDA、统计检验、图表生成
6. **评估反思 + 报告生成** — LLM-as-Judge 评分、Self-Reflection 迭代改进、输出 Word/MD/HTML 三种格式报告

**技术栈**：Python 3.11+, OpenAI-compatible API (百炼/OpenAI/DeepSeek 等), ArXiv API, NumPy, Pandas, Matplotlib, scikit-learn, scipy, FAISS (可选), sentence-transformers (可选), python-docx。

---

## 2. 目录结构全览

```
mindpilot/
├── main.py                    # 程序主入口（交互模式 / 命令行模式）
├── config.py                  # 全局配置中心（所有参数集中管理）
├── __init__.py
├── requirements.txt           # Python 依赖
├── .env                       # 环境变量（API Key、模型名等，不进版本控制）
├── config.example.env         # 环境变量示例
│
├── orchestrator/              # 中央编排器（核心调度中枢）
│   ├── __init__.py
│   └── orchestrator.py        # MindPilotOrchestrator — 六步流水线调度
│
├── agents/                    # 五大 Agent（各司其职）
│   ├── __init__.py
│   ├── planning_agent.py      # 模块① 任务规划 Agent (ToT + ReAct)
│   ├── literature_agent.py    # 模块② 文献检索 Agent (ArXiv + 知识图谱)
│   ├── code_agent.py          # 模块③ 代码生成 Agent (自动调试闭环)
│   ├── analysis_agent.py      # 模块④ 数据分析 Agent (EDA + 统计 + 可视化)
│   └── evaluation_agent.py    # 模块⑥ 评估反思 Agent (LLM-Judge + Reflection + 报告)
│
├── tools/                     # 工具层（各 Agent 依赖的底层工具）
│   ├── __init__.py
│   ├── llm_client.py          # 统一 LLM 客户端（OpenAI兼容接口 + Mock降级）
│   ├── arxiv_search.py        # ArXiv 文献检索工具（含中英文自动翻译）
│   ├── code_executor.py       # 代码沙箱执行器（AST安全检测 + 子进程隔离）
│   ├── visualizer.py          # 智能可视化工具（自动图表推荐 + matplotlib绘制）
│   └── report_generator.py    # 多格式报告生成器（Word + Markdown + HTML）
│
├── framework/                 # 框架层（通信、日志、调度等基础设施）
│   ├── __init__.py
│   ├── communication.py       # Agent 间通信协议（消息总线 + 重试 + 人机协同）
│   ├── logger.py              # 统一日志系统（彩色控制台 + JSONL结构化日志）
│   └── scheduler.py           # 并发调度器（DAG拓扑排序 + 同步/异步执行）
│
├── memory/                    # 记忆系统
│   ├── __init__.py
│   ├── memory_store.py        # 短期+长期记忆存储（FAISS向量检索 / 关键词回退）
│   └── store/
│       └── long_term.jsonl    # 长期记忆持久化文件
│
├── evaluation/                # 评估基准
│   ├── __init__.py
│   └── benchmark.py           # 20条基准测试用例 + 指标计算器
│
├── examples/
│   └── demo.py                # 5个演示脚本（完整流程 / 单模块演示 / 基准测试）
│
├── tests/                     # 单元测试
│   ├── test_planning.py       # 规划 Agent 测试
│   ├── test_literature.py     # 文献 Agent 测试
│   └── test_code_eval.py      # 代码 Agent + 评估 Agent 测试
│
├── logs/                      # 运行日志输出目录
│   ├── session_*.jsonl        # 每次会话的详细JSONL日志
│   └── summary_*.json         # 每次会话的汇总统计
│
└── outputs/                   # 报告与图表输出目录
    ├── final_report_*.docx    # Word 格式报告
    ├── final_report_*.md      # Markdown 格式报告
    ├── final_report_*.html    # HTML 格式报告
    ├── analysis_report_*.*    # 分析阶段报告
    └── *.png                  # 可视化图表
```

---

## 3. 核心架构设计

### 3.1 分层架构

```
┌─────────────────────────────────────────────────┐
│                   用户交互层                       │
│              main.py (CLI 交互/命令行)             │
├─────────────────────────────────────────────────┤
│                  编排调度层                        │
│    orchestrator.py (六步流水线、Agent 调度)        │
├─────────────────────────────────────────────────┤
│                  Agent 业务层                      │
│  PlanningAgent │ LiteratureAgent │ CodeAgent     │
│  AnalysisAgent │ EvaluationAgent                 │
├─────────────────────────────────────────────────┤
│                   工具层                          │
│  LLMClient │ ArXivSearch │ CodeExecutor          │
│  Visualizer │ ReportGenerator                    │
├─────────────────────────────────────────────────┤
│                  框架层                           │
│  Logger │ MessageBus │ Scheduler │ MemoryStore   │
├─────────────────────────────────────────────────┤
│                  外部依赖                          │
│  OpenAI API │ ArXiv API │ FAISS │ matplotlib     │
└─────────────────────────────────────────────────┘
```

### 3.2 核心设计原则

- **单一编排者模式**：`MindPilotOrchestrator` 是唯一的调度中枢，按顺序调用各 Agent，各 Agent 之间不直接通信。
- **Agent 无状态**：每个 Agent 接收输入、产出输出，状态通过 MemoryStore 跨步骤传递。
- **优雅降级**：LLM 不可用时自动切换 Mock 模式；ArXiv 不可达时返回 Mock 论文；FAISS 不可用时回退关键词搜索。
- **配置驱动**：所有参数集中在 `config.py` + `.env`，无硬编码。
- **安全沙箱**：生成的代码经 AST 静态检测后在子进程中执行，禁止危险操作。

---

## 4. 完整执行流程（六步流水线）

当用户输入科研问题后，`orchestrator.run(query)` 按以下六步顺序执行：

```
用户输入 query
       │
       ▼
┌──────────────────────────────────────────────────┐
│  Step 1: 任务规划  (PlanningAgent.run)            │
│  ① ToT 生成 3 条研究路径                          │
│  ② ToT 评分选出最优路径                           │
│  ③ ReAct 分解为 5~7 个 SubTask                    │
│  ④ DAG 验证（去除无效依赖）                       │
│  → 输出: ResearchPlan (任务列表 + 推理 + 路径)     │
└──────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│  Step 2: 文献检索  (LiteratureAgent.run)          │
│  ① 中文查询 → 英文翻译                           │
│  ② ArXiv API 检索论文                             │
│  ③ TF-IDF 重排序                                  │
│  ④ 每篇论文生成结构化摘要 (LLM)                   │
│  ⑤ 构建知识图谱 (论文-作者-类别)                   │
│  ⑥ LLM 生成文献综述段落                           │
│  → 输出: {papers, knowledge_graph, review, metrics}│
└──────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│  Step 3: 实验设计  (EvaluationAgent.design_exp.)  │
│  ① 提取文献中的方法作为参考                       │
│  ② LLM 生成实验方案 JSON                          │
│     (假设/数据集/基线/指标/流程/预期结果)           │
│  → 输出: {hypothesis, baselines, metrics, ...}     │
└──────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│  Step 4: 代码实现  (CodeAgent.run)                │
│  ① 从记忆检索相似代码                             │
│  ② LLM 生成初始代码 (code_model)                  │
│  ③ 循环（最多 5 轮）:                             │
│     a. AST 安全检测                               │
│     b. 子进程执行代码                             │
│     c. 成功 → break；失败 → LLM 自动调试          │
│  ④ LLM 生成单元测试                               │
│  → 输出: {success, final_code, stdout, pass_at_1}  │
└──────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│  Step 5: 数据分析  (AnalysisAgent.run)            │
│  ① NL → 分析意图解析                              │
│  ② 准备数据（代码输出/模拟数据）                   │
│  ③ 自动 EDA (describe, missing, correlation)       │
│  ④ 统计检验 (Shapiro-Wilk, Pearson, t-test 等)    │
│  ⑤ 智能图表推荐 + matplotlib 绘制                  │
│  ⑥ LLM 生成分析结论                               │
│  ⑦ 输出分析报告 (MD + HTML)                        │
│  → 输出: {summary, tests, charts, conclusion}      │
└──────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│  Step 6: 评估反思+报告  (EvaluationAgent.run)     │
│  ① 整合所有 Agent 输出，构建报告结构               │
│  ② LLM 逐章扩写（背景/综述/设计/方法/结果/结论）  │
│  ③ LLM-as-Judge 评分                              │
│  ④ Self-Reflection 循环（评分<0.65时触发）         │
│  ⑤ 生成最终报告: Word + Markdown + HTML            │
│  → 输出: {final_score, report_files}               │
└──────────────────────────────────────────────────┘
       │
       ▼
  收尾: 保存长期记忆 → 保存日志摘要 → 打印调用链
       │
       ▼
  返回 final_result（包含所有阶段结果 + 报告路径）
```

---

## 5. 各模块详细解读

### 5.1 入口层：main.py

**文件**：`main.py` (77行)
**职责**：程序入口，提供两种运行模式。

**运行方式**：
```bash
# 交互模式（循环读取用户输入）
python main.py

# 命令行模式（直接传入问题）
python main.py "研究 Transformer 注意力机制"
```

**核心逻辑**：
1. `interactive_mode()` — 循环 `input()` → `orchestrator.run(query)` → 打印报告路径
2. `single_run(query)` — 一次性执行

**关键代码路径**：
```python
orchestrator = MindPilotOrchestrator()   # 初始化整个系统
result = orchestrator.run(query)          # 执行六步流水线
```

**修改建议**：
- 如需添加 Web API 接口，可在此层添加 FastAPI/Flask 路由，调用 `single_run(query)` 即可。
- 如需修改交互逻辑（如添加参数选择、指定模块运行），修改 `interactive_mode()` 函数。

---

### 5.2 配置中心：config.py

**文件**：`config.py` (121行)
**职责**：集中管理所有配置参数，从 `.env` 文件读取环境变量。

**配置类结构**：

| 配置类 | 控制的模块 | 关键参数 | 说明 |
|--------|-----------|---------|------|
| `LLMConfig` | LLM调用 | `api_key`, `base_url`, `model`, `code_model`, `temperature`, `timeout` | 支持双模型：通用模型 + 代码专用模型 |
| `PlanningConfig` | 任务规划 | `branching_factor=3`, `max_depth=3`, `max_subtasks=8` | ToT分支因子=3意味着6~9次LLM调用 |
| `LiteratureConfig` | 文献检索 | `arxiv_max_results=10`, `retrieval_top_k=5` | 控制检索数量和排序 |
| `CodeConfig` | 代码执行 | `max_debug_rounds=5`, `execution_timeout=30`, `forbidden_modules` | 调试轮数和安全策略 |
| `AnalysisConfig` | 数据分析 | `significance_level=0.05`, `report_formats` | 统计检验显著性水平 |
| `CommunicationConfig` | 框架通信 | `max_concurrent_agents=4`, `message_timeout=300` | 并发和超时控制 |
| `EvaluationConfig` | 评估反思 | `score_threshold=0.65`, `max_reflection_rounds=3` | 触发反思的分数阈值 |
| `MindPilotConfig` | 顶层聚合 | `output_dir`, `memory_dir`, `mock_mode`, `verbose` | 聚合所有子配置 |

**Mock 模式判定**（`_is_mock()` 函数）：
```python
# 以下情况自动进入 Mock 模式：
# 1. LLM_API_KEY 未设置
# 2. LLM_API_KEY 以 "sk-your-" 开头（占位符）
# 3. LLM_API_KEY 等于 "mock"
```

**全局单例**：
```python
CONFIG = MindPilotConfig()  # 模块级别创建，所有组件共享同一个 CONFIG 实例
```

**修改建议**：
- 添加新参数：在对应的 `@dataclass` 中添加字段，默认值从 `os.getenv()` 读取。
- 切换模型：修改 `.env` 中的 `LLM_MODEL` 和 `CODE_MODEL`。
- 调整性能：修改 `PlanningConfig.branching_factor`（降低可加速，提高可改善规划质量）。

---

### 5.3 中央编排器：orchestrator/orchestrator.py

**文件**：`orchestrator/orchestrator.py` (219行)
**职责**：系统调度中枢，初始化所有组件，按顺序执行六步流水线。

**类**：`MindPilotOrchestrator`

**`__init__` 初始化流程**（35-69行）：

```
MindPilotOrchestrator.__init__()
  ├─ 创建 session_id (UUID前12位)
  ├─ 创建 MindPilotLogger (日志系统)
  ├─ 创建 MessageBus (消息总线，当前未在主流程使用)
  ├─ 创建 HumanInTheLoop (人机协同，默认禁用)
  ├─ 创建 LLMClient (统一LLM客户端)
  ├─ 创建 MemoryStore (记忆系统)
  ├─ 创建 ArXivSearchTool (ArXiv检索)
  ├─ 创建 CodeExecutor (代码沙箱)
  ├─ 创建 AutoVisualizer (可视化)
  ├─ 创建 ReportGenerator (报告生成)
  ├─ 创建 PlanningAgent (规划Agent)
  ├─ 创建 LiteratureAgent (文献Agent)
  ├─ 创建 CodeAgent (代码Agent)
  ├─ 创建 AnalysisAgent (分析Agent)
  └─ 创建 EvaluationAgent (评估Agent)
```

**`run(query)` 方法是整个系统的核心（71-174行）**：

```python
def run(self, query: str) -> dict:
    # Step 1: 任务规划
    plan = self.planner.run(query)

    # Step 2: 文献检索 — 从 plan 中找到 LiteratureAgent 任务获取描述
    lit_result = self.lit_agent.run(desc, query)

    # Step 3: 实验设计 — 基于文献结果
    exp_design = self.eval_agent.design_experiment(query, lit_result)

    # Step 4: 代码实现 — 携带文献方法和实验设计作为上下文
    code_result = self.code_agent.run(code_desc, context={
        "top_papers": ..., "exp_design": ...,
        "baselines": ..., "metrics": ...
    })

    # Step 5: 数据分析 — 使用代码的 stdout 输出
    analysis_result = self.analysis_agent.run(ana_desc, code_output=code_stdout)

    # Step 6: 评估+报告 — 聚合所有结果
    eval_result = self.eval_agent.run(query, aggregated)

    # 收尾
    self.memory.save_long_term()
    self.logger.save_summary()
```

**上下文传递链**：
```
query → PlanningAgent → plan.tasks (包含各Agent的描述)
                    ↓
query + plan → LiteratureAgent → lit_result (论文、综述、图谱)
                              ↓
query + lit_result → EvaluationAgent.design_experiment → exp_design
                                                      ↓
plan + lit_result + exp_design → CodeAgent → code_result (代码、stdout)
                                          ↓
plan + code_result.stdout → AnalysisAgent → analysis_result (统计、图表、结论)
                                          ↓
所有结果聚合 → EvaluationAgent.run → eval_result (评分、报告文件)
```

**修改建议**：
- 添加新 Agent：在 `__init__` 中创建实例，在 `run()` 中添加相应步骤。
- 修改步骤顺序：直接调整 `run()` 中的步骤顺序和数据传递。
- 并行执行独立步骤：将 `SyncScheduler` 替换为 `DAGScheduler` 并在 async 环境中运行。

---

### 5.4 任务规划 Agent：agents/planning_agent.py

**文件**：`agents/planning_agent.py` (275行)
**职责**：将科研问题分解为可执行的子任务 DAG。

**内含三个核心类**：

#### (1) `TreeOfThoughtPlanner` (ToT, 46-121行)

**功能**：生成多条研究路径并评分选优。

**流程**：
```
search(query)
  ├─ _generate_paths(query)
  │     调用 LLM 生成 branching_factor(=3) 条研究路径
  │     每条路径包含：id, description, steps
  │     若 LLM 返回解析失败 → 使用 3 条默认模板路径
  │
  └─ _score_paths(query, paths)
        对每条路径调用 LLM 评分（0-1）
        考虑：可行性、完整性、创新性
        若解析失败 → random.uniform(0.6, 0.9)
        选出 score 最高的路径
```

**LLM 调用次数**：`1 (生成路径) + 3 (评分每条路径) = 4 次`

#### (2) `ReActPlanner` (124-204行)

**功能**：将科研问题分解为 5~7 个具体 SubTask。

**核心 prompt 策略**：
- 告知 LLM 可用的 4 种 Agent（`LiteratureAgent`, `CodeAgent`, `AnalysisAgent`, `EvaluationAgent`）
- 要求输出 JSON 格式的任务列表，包含 id、name、agent、description、depends_on、priority
- 结合 ToT 选出的最优研究路径作为上下文

**Fallback**：解析失败时使用 `_default_tasks()` 返回 5 个默认任务。

#### (3) `PlanningAgent` (207-275行)

**功能**：整合 ToT + ReAct，完成规划全流程。

**`run(query)` 流程**：
```
1. memory.search(query) — 检索相关历史记录
2. tot.search(query) — ToT 生成并评分路径
3. react.decompose(query, best_path) — ReAct 分解子任务
4. _validate_dag(tasks) — 验证依赖关系（去除无效引用）
5. 构建 ResearchPlan 对象
6. memory.add(...) — 存入记忆
7. logger.finish_call(...) — 记录完成
```

**数据结构**：

```python
@dataclass
class SubTask:
    task_id: str           # "T1", "T2", ...
    name: str              # "文献检索与综述"
    agent: str             # "LiteratureAgent"
    description: str       # 任务详细描述
    depends_on: list[str]  # ["T1", "T2"] 前置依赖
    priority: int          # 数字越大越重要
    estimated_time: str    # "5min"
    status: str            # "pending"

@dataclass
class ResearchPlan:
    plan_id: str
    query: str
    tasks: list[SubTask]
    reasoning: str         # 分解思路
    selected_path: str     # ToT 选出的最优路径描述
```

**修改建议**：
- 调整规划粒度：修改 ReAct prompt 中的任务数量要求（当前 5~7 个）。
- 添加新 Agent 类型：在 `ReActPlanner.AVAILABLE_AGENTS` 字典中添加新 Agent 描述。
- 修改 ToT 策略：调整 `PlanningConfig.branching_factor` 或修改 `_generate_paths` 的 prompt。

---

### 5.5 文献检索 Agent：agents/literature_agent.py

**文件**：`agents/literature_agent.py` (263行)
**职责**：检索学术文献、构建知识图谱、生成结构化摘要和综述。

**内含三个辅助类**：

#### (1) `LightKnowledgeGraph` (32-98行)

纯 Python 实现的轻量知识图谱（无需 Neo4j/NetworkX）：
- **节点类型**：paper, author, category
- **边类型**：authored_by, belongs_to
- **核心方法**：`add_paper()` 自动创建论文节点、作者节点、类别节点及关系边
- **多跳查询**：`multi_hop_query(start_label, hops=2)` BFS 遍历

#### (2) `StructuredSummarizer` (101-138行)

调用 LLM 将论文摘要压缩为结构化 JSON：
```json
{"method": "...", "conclusion": "...", "limitation": "..."}
```
Fallback：LLM 解析失败时，按句子切分摘要。

#### (3) `LiteratureAgent` (141-263行)

**`run(task_description, query)` 流程**：
```
1. ArXiv 检索论文列表
2. _rerank(papers, query) — TF-IDF 近似重排序
   原始相关性权重 0.6 + TF-IDF 权重 0.4
3. 为每篇论文调用 StructuredSummarizer → paper.structured_summary
4. 为每篇论文调用 kg.add_paper() → 构建知识图谱
5. _compute_recall_at_k() — 计算 Recall@5 和 Recall@10
6. _generate_review() — LLM 生成 200 字文献综述
7. memory.add() — 存入记忆
```

**LLM 调用次数**：`N (摘要, 每篇论文1次) + 1 (综述) = 约 6~11 次`

**修改建议**：
- 切换检索源：替换 `ArXivSearchTool` 为其他学术数据库（如 Semantic Scholar API）。
- 增强重排序：用 sentence-transformers 替代 TF-IDF。
- 扩展知识图谱：添加 method、dataset 等节点类型。

---

### 5.6 代码生成 Agent：agents/code_agent.py

**文件**：`agents/code_agent.py` (267行)
**职责**：代码生成 → 安全检测 → 执行 → 自动调试闭环。

**类**：`CodeAgent`

**`run(task_description, context)` 流程**：
```
1. memory.search() — 检索历史代码
2. _generate_code() — LLM 生成初始代码 (使用 code_model)
3. 调试循环 (最多 max_debug_rounds=5 轮):
   │
   ├─ AST 安全检测 (executor.checker.check)
   │   ├─ 语法错误 → 二次提取 → 仍失败 → _regenerate_clean_code()
   │   └─ 安全问题 → _fix_safety_issues()
   │
   ├─ 子进程执行 (executor.execute_with_subprocess)
   │
   ├─ 成功 → 记录 pass_at_1, break
   └─ 失败 → _debug_code() (LLM根据错误信息修复) → 下一轮
4. _generate_tests() — LLM 自动生成单元测试
5. memory.add() — 存入记忆
```

**关键点**：
- 使用 `llm.chat_code()` 调用代码专用模型（`CODE_MODEL`），而非通用模型
- 代码从 LLM 输出中提取时，使用 `executor.extract_code()` 处理 markdown 代码块
- 文献方法和实验设计作为上下文传入，指导代码生成

**代码执行的两种方式**：
1. `execute()` — 在当前进程的受限命名空间中执行（更快，但隔离性差）
2. `execute_with_subprocess()` — 子进程执行（**实际使用的方式**，更安全）

**修改建议**：
- 增加调试轮数：修改 `CodeConfig.max_debug_rounds`。
- 添加代码优化步骤：在成功执行后调用 LLM 优化代码质量。
- 支持其他语言：扩展 `CodeExecutor` 支持 R/Julia 等。

---

### 5.7 数据分析 Agent：agents/analysis_agent.py

**文件**：`agents/analysis_agent.py` (367行)
**职责**：自然语言 → 分析指令转换 + 自动 EDA + 统计检验 + 可视化 + 报告。

**内含两个类**：

#### (1) `NLToAnalysis` (26-62行)

自然语言意图分析器，基于关键词匹配，识别 8 种分析类型：
```
distribution | comparison | correlation | significance_test
regression   | trend      | clustering  | eda
```

还包含 `select_test()` 方法，根据意图自动选择统计检验方法。

#### (2) `AnalysisAgent` (65-367行)

**`run(task_description, data, code_output)` 流程**：
```
1. nl_parser.parse() — 识别分析意图
2. _prepare_data() — 准备 DataFrame
   ├─ 用户传入数据 → 直接使用
   ├─ 从 code_output 中正则提取数字
   └─ 生成符合任务的模拟数据
3. _run_eda() — 自动 EDA
   ├─ shape, columns, describe
   ├─ missing values
   └─ correlation matrix
4. _run_statistical_analysis() — 统计检验
   ├─ Shapiro-Wilk 正态性检验（始终执行）
   ├─ Pearson 相关（若 intent=correlation/regression 且>=2列）
   └─ 独立样本 t 检验（若 intent=comparison 且>=2列）
5. visualizer.infer_chart_type() — 推荐图表类型
6. _generate_charts() — matplotlib 绘图
7. _generate_conclusion() — LLM 生成分析结论
8. report_gen.generate() — 输出 MD + HTML 报告
9. memory.add() — 存入记忆
```

**数据准备的三级策略**：
1. 优先使用传入数据
2. 从代码输出中正则提取数字（`re.findall(r"[-+]?\d+\.?\d*", code_output)`）
3. 根据任务描述关键词生成模拟数据（回归/对比/通用三种模板）

**修改建议**：
- 添加新统计检验：在 `_run_statistical_analysis()` 中添加分支。
- 增强 NL 解析：用 LLM 替代关键词匹配实现意图识别。
- 支持真实数据源：扩展 `_prepare_data()` 支持 CSV/数据库输入。

---

### 5.8 评估反思 Agent：agents/evaluation_agent.py

**文件**：`agents/evaluation_agent.py` (484行)
**职责**：双重角色 — ① 实验设计生成 ② 评估反思+报告生成。

**内含四个辅助类**：

#### (1) `LLMJudge` (36-91行)

LLM-as-Judge 评分器：
- 调用 LLM 对输出进行四维评分：accuracy, completeness, format_quality, overall
- 评分 < `score_threshold`(0.65) 时标记 `needs_reflection=True`
- 附带 `compute_rouge_l()` ROUGE-L F1 计算方法

#### (2) `SelfReflector` (93-116行)

自我反思机制：
- 根据评审意见调用 LLM 补充和修正报告
- 重点补充缺失内容，不删减已有内容

#### (3) `BenchmarkEvaluator` (119-178行)

基准对比评估器：
- 10 个 AI 领域标准问题 + 关键词参考答案
- 对比三个系统：MindPilot vs LLM-only vs RAG-only
- 评估指标：关键词召回率

#### (4) `EvaluationAgent` (181-484行)

**两个核心方法**：

**`design_experiment(query, literature_result)`** (195-265行)：
```
1. 提取文献方法作为参考
2. LLM 生成实验设计 JSON:
   {hypothesis, dataset, baselines, metrics, procedure, expected_results, full_description}
```

**`run(query, outputs)`** (268-346行) — 评估 + 报告主流程：
```
1. _build_rich_report() — 构建六章节学术报告
   ├─ 一、研究背景 (LLM 扩写, >=300字)
   ├─ 二、文献综述 (LLM 扩写, >=400字)
   ├─ 三、实验设计 (直接使用或LLM扩写)
   │   ├─ 3.1 实验假设
   │   ├─ 3.2 评估指标
   │   └─ 3.3 基线方法
   ├─ 四、核心方法实现 (LLM 扩写, >=200字)
   ├─ 五、实验结果与分析 (LLM 扩写, >=300字)
   └─ 六、结论与展望 (LLM 扩写, >=200字)

2. LLMJudge 评分

3. Self-Reflection 循环 (最多 3 轮):
   while score < 0.65:
     revised = reflector.reflect_and_revise()
     new_score = judge.score(revised)
     if new_score > score: accept; else: break

4. report_gen.generate() — 输出 Word + MD + HTML
5. memory.add() — 存入记忆
```

**LLM 调用次数（Step 6）**：`6 (章节扩写) + 1 (评分) + 0~3×2 (反思) = 7~13 次`

**修改建议**：
- 调整报告章节：修改 `_build_rich_report()` 中的 sections 列表。
- 修改评分标准：调整 `LLMJudge` 的 prompt 和 `EvaluationConfig.score_threshold`。
- 添加 PDF 格式：在 `ReportGenerator` 中添加 `_to_pdf()` 方法。

---

### 5.9 工具层 tools/

#### 5.9.1 LLM 客户端：tools/llm_client.py (309行)

**类**：`LLMClient`
**核心职责**：封装 OpenAI-compatible API 调用，提供统一接口 + Mock 降级。

**初始化连接探测流程**：
```
_init_client()
  ├─ 若指定代理 (proxy_url) → 直接用代理连接
  ├─ 否则自动探测:
  │   ├─ 先尝试直连 (trust_env=False，绕过系统代理)
  │   └─ 直连失败 → 尝试系统代理 (trust_env=True)
  └─ 全部失败 → mock_mode = True
```

**双客户端设计**：
- `_client` — 通用对话模型（如 `glm-5`）
- `_client_code` — 代码专用模型（如 `qwen3-coder-plus`）

**对外 API**：
- `chat(messages, ...)` — 通用对话
- `chat_code(messages, ...)` — 代码生成（自动使用 code_model）

**Mock 模式** (233-309行)：
根据 system prompt 中的关键词，返回不同的模拟响应：
- 含"规划/plan" → 返回默认任务列表 JSON
- 含"路径/path" → 返回评分 JSON
- 含"代码/python" → 返回线性回归代码
- 含"摘要/summary" → 返回结构化摘要 JSON
- 含"评估/judge" → 返回评分 JSON

#### 5.9.2 ArXiv 检索：tools/arxiv_search.py (318行)

**类**：`ArXivSearchTool`
**核心特性**：
- 使用 ArXiv 官方 Atom API（无需 API Key）
- **中英文自动翻译**：内置 `_CN_TO_EN` 字典（约 60 个常见 AI 术语对照），自动将中文查询转为英文
- 相关性评分：基于查询词在标题+摘要中的出现频率
- Mock fallback：网络不可达时返回 5 篇模板论文

**数据结构**：`Paper` dataclass 包含 arxiv_id, title, authors, abstract, published, categories, relevance_score, structured_summary。

#### 5.9.3 代码执行器：tools/code_executor.py (395行)

**两个核心类**：

**`ASTSafetyChecker`** (47-119行)：
- AST 静态分析 + 正则检测
- 禁止的调用：`os.system`, `subprocess.run`, `eval`, `exec`, `socket` 等
- 禁止的导入：`socket`, `ctypes`, `multiprocessing` 等
- 检测文件写操作：`open(..., 'w')`, `.write()`

**`CodeExecutor`** (122-395行)：
- `extract_code(text)` — 从 LLM 输出提取 Python 代码（处理7种 markdown 变体）
- `execute(code)` — 在受限命名空间中执行（允许 numpy/pandas/matplotlib/sklearn/scipy）
- `execute_with_subprocess(code)` — **实际使用**，子进程执行，更高安全隔离
- 安全全局变量构建：只暴露白名单 builtins + 科学计算库

#### 5.9.4 可视化工具：tools/visualizer.py (198行)

**类**：`AutoVisualizer`
**支持图表类型**：histogram, lineplot, barplot, scatter, scatter_with_fit, heatmap, pie
**推荐逻辑**：
1. 优先匹配关键词（"分布"→histogram, "趋势"→lineplot, "相关"→scatter 等）
2. 启发式规则（有分类+数值列→barplot, 双数值列→scatter, 默认→histogram）

#### 5.9.5 报告生成器：tools/report_generator.py (378行)

**类**：`ReportGenerator`
**三种输出格式**：
- **Word (.docx)**：使用 `python-docx`，包含封面、摘要、正文章节、代码块、参考文献、评估信息，带格式化排版
- **Markdown (.md)**：结构化标题层级、代码块、文献列表、评估表格
- **HTML**：从 Markdown 用正则转换，包含完整 CSS 样式

---

### 5.10 框架层 framework/

#### 5.10.1 通信协议：framework/communication.py (207行)

**核心组件**：

- **`Message` dataclass** — 统一消息格式（msg_id, type, sender, receiver, payload, error_code, timestamp 等）
- **`ErrorCode` 枚举** — OK, TIMEOUT, LLM_ERROR, TOOL_ERROR, EXECUTION_ERROR 等
- **`MessageBus`** — 内存级消息总线（asyncio.Queue），支持 register/send/receive/subscribe
- **`with_retry` 装饰器** — 同步重试（支持退避策略）
- **`HumanInTheLoop`** — 人机协同门控（可在关键节点暂停等待用户确认）

> **注意**：当前主流程 (`orchestrator.run()`) 是同步串行调用各 Agent，**未使用** MessageBus 和 async 调度。这些组件是为未来异步并发扩展预留的。

#### 5.10.2 日志系统：framework/logger.py (221行)

**类**：`MindPilotLogger`
**功能**：
- **控制台彩色输出**：每个 Agent 有独立颜色标识
- **JSONL 结构化日志**：`logs/session_{id}.jsonl`
- **调用链记录**：`start_call()` / `finish_call()` / `fail_call()`，记录每次 Agent 调用的耗时、输入输出
- **会话摘要**：`save_summary()` 输出到 `logs/summary_{id}.json`
- **调用链回放**：`print_call_chain()` 打印完整调用序列

**第三方日志抑制**：将 openai, httpx, httpcore, urllib3 的日志级别设为 ERROR/CRITICAL，避免刷屏。

#### 5.10.3 并发调度器：framework/scheduler.py (219行)

**两个调度器**：

- **`DAGScheduler`（异步版）**：
  - 对任务 DAG 进行拓扑分层
  - 同层任务并发执行（`asyncio.gather`）
  - 使用 `Semaphore` 控制最大并发数
  - 前置任务失败时自动跳过依赖任务

- **`SyncScheduler`（同步版）**：
  - DFS 拓扑排序后串行执行
  - 当前编排器中 import 了但未在主流程中实际使用

> **注意**：当前编排器是手动按步骤顺序调用各 Agent。若要启用 DAG 并发调度，需将各 Agent 封装为 Task 并交由 DAGScheduler 执行。

---

### 5.11 记忆系统 memory/

**文件**：`memory/memory_store.py` (188行)
**类**：`MemoryStore`

**双层记忆架构**：
- **短期记忆** (`_short_term`)：当前会话的 in-memory list
- **长期记忆** (`_long_term`)：持久化到 `memory/store/long_term.jsonl`

**记忆条目**（`MemoryEntry`）：
```python
entry_id, session_id, agent, content, payload, timestamp, tags, importance
```

**检索策略（两级）**：
1. **向量检索**（优先）：`sentence-transformers` 编码 + FAISS IndexFlatIP（内积）
2. **关键词检索**（fallback）：词集合交集 / importance 加权

**生命周期**：
- `add()` — 同时写入短期+长期记忆
- `search()` — 在长期记忆中检索
- `get_recent()` — 获取短期记忆最近 N 条
- `save_long_term()` — 持久化到 JSONL 文件
- `_load_long_term()` — 启动时从 JSONL 加载

**所有 Agent 的记忆使用模式**：
```python
# 任务开始时 — 检索相关历史
similar = self.memory.search(query, top_k=5)

# 任务完成后 — 存入记忆
self.memory.add(content="...", agent=self.AGENT_NAME, payload={...}, tags=[...])
```

---

### 5.12 评估基准 evaluation/

**文件**：`evaluation/benchmark.py` (129行)

**`BENCHMARK_CASES`**：20 条标准科研任务（7 文献 + 6 代码 + 4 分析 + 3 综合），每条包含参考关键词。

**`MetricsCalculator`** — 四种评估指标：
- `keyword_recall(output, keywords)` — 关键词召回率
- `recall_at_k(retrieved, keywords, k)` — Top-K 召回率
- `rouge_l(hypothesis, reference)` — ROUGE-L F1
- `pass_at_k(results, k)` — Pass@K 代码正确率

---

## 6. 数据流与依赖关系图

### 6.1 组件依赖图

```
main.py
  └── MindPilotOrchestrator
        ├── MindPilotLogger
        ├── MessageBus (预留)
        ├── HumanInTheLoop (默认禁用)
        ├── LLMClient ────────────────── OpenAI API
        ├── MemoryStore ─────────────── FAISS / sentence-transformers (可选)
        ├── ArXivSearchTool ─────────── ArXiv API
        ├── CodeExecutor ────────────── subprocess
        ├── AutoVisualizer ──────────── matplotlib
        ├── ReportGenerator ─────────── python-docx
        │
        ├── PlanningAgent
        │     ├── TreeOfThoughtPlanner → LLMClient
        │     ├── ReActPlanner → LLMClient
        │     └── MemoryStore
        │
        ├── LiteratureAgent
        │     ├── ArXivSearchTool
        │     ├── StructuredSummarizer → LLMClient
        │     ├── LightKnowledgeGraph
        │     └── MemoryStore
        │
        ├── CodeAgent
        │     ├── CodeExecutor (AST检测 + 子进程执行)
        │     ├── LLMClient (code_model)
        │     └── MemoryStore
        │
        ├── AnalysisAgent
        │     ├── NLToAnalysis (意图解析)
        │     ├── AutoVisualizer
        │     ├── ReportGenerator
        │     ├── LLMClient
        │     └── MemoryStore
        │
        └── EvaluationAgent
              ├── LLMJudge → LLMClient
              ├── SelfReflector → LLMClient
              ├── BenchmarkEvaluator → LLMClient
              ├── ReportGenerator
              └── MemoryStore
```

### 6.2 单次完整执行的 LLM 调用总量估算

| 步骤 | Agent | 调用次数 | 使用模型 |
|------|-------|---------|---------|
| Step 1: ToT 生成路径 | PlanningAgent | 1 | 通用模型 |
| Step 1: ToT 评分 | PlanningAgent | 3 | 通用模型 |
| Step 1: ReAct 分解 | PlanningAgent | 1 | 通用模型 |
| Step 2: 结构化摘要 | LiteratureAgent | 5~10 | 通用模型 |
| Step 2: 文献综述 | LiteratureAgent | 1 | 通用模型 |
| Step 3: 实验设计 | EvaluationAgent | 1 | 通用模型 |
| Step 4: 代码生成 | CodeAgent | 1 | 代码模型 |
| Step 4: 调试修复 | CodeAgent | 0~4 | 代码模型 |
| Step 4: 单元测试 | CodeAgent | 1 | 代码模型 |
| Step 5: 分析结论 | AnalysisAgent | 1 | 通用模型 |
| Step 6: 章节扩写 | EvaluationAgent | 6 | 通用模型 |
| Step 6: 评分 | EvaluationAgent | 1 | 通用模型 |
| Step 6: 反思修订 | EvaluationAgent | 0~6 | 通用模型 |
| **合计** | | **约 22~36 次** | |

---

## 7. LLM 调用机制详解

### 7.1 双模型架构

系统使用两个模型：
- **通用模型** (`LLM_MODEL`, 默认 `glm-5`)：规划、摘要、评估、报告扩写
- **代码模型** (`CODE_MODEL`, 默认 `qwen3-coder-plus`)：代码生成、调试、测试生成

调用方式：
```python
# 通用模型
self.llm.chat(messages)

# 代码模型
self.llm.chat_code(messages)
# 等价于 self.llm.chat(messages, use_code_model=True)
```

### 7.2 连接探测

启动时自动探测 API 连通性：
```
1. 若设置了 LLM_PROXY_URL → 直接使用指定代理
2. 否则先尝试直连（绕过系统代理）
3. 直连失败 → 尝试系统代理
4. 全部失败 → 切换 Mock 模式
```

### 7.3 错误处理

API 调用失败时：
- 打印友好的错误说明（连接拒绝/超时/401/403/404 等）
- 对 400 模型名错误给出具体建议
- 降级为 Mock 响应（不中断流程）

---

## 8. Mock 模式与降级策略

系统的"优雅降级"贯穿所有层：

| 组件 | 正常模式 | 降级条件 | 降级行为 |
|------|---------|---------|---------|
| LLMClient | 调用真实 API | API Key 无效/网络不通 | 根据 prompt 关键词返回预设响应 |
| ArXivSearchTool | 调用 ArXiv API | 网络不可达 | 返回 5 篇模板论文 |
| MemoryStore (向量) | FAISS + sentence-transformers | 库未安装 | 关键词匹配检索 |
| CodeExecutor | 子进程执行 | 超时 | 返回 TimeoutError 结果 |
| ReportGenerator (docx) | python-docx 生成 | 库未安装 | 跳过 docx 格式 |
| Visualizer | matplotlib 绘图 | matplotlib 未安装 | 返回空路径 |
| PlanningAgent (ToT) | LLM 生成路径 | LLM 返回解析失败 | 使用 3 条默认模板路径 |
| PlanningAgent (ReAct) | LLM 分解任务 | LLM 返回解析失败 | 使用 5 个默认任务模板 |
| LLMJudge | LLM 评分 | LLM 返回解析失败 | random.uniform(0.62, 0.88) |

---

## 9. 修改指南：各模块如何入手修改

### 9.1 添加新的 Agent

**步骤**：
1. 在 `agents/` 目录创建新文件，如 `agents/data_collection_agent.py`
2. 参照已有 Agent 模板实现：
   ```python
   class DataCollectionAgent:
       AGENT_NAME = "DataCollectionAgent"

       def __init__(self, config, llm_client, memory_store, logger):
           ...

       def run(self, task_description: str, ...) -> dict:
           call = self.logger.start_call(self.AGENT_NAME, "task_id", input)
           try:
               # 业务逻辑
               result = {...}
               self.memory.add(content="...", agent=self.AGENT_NAME, ...)
               self.logger.finish_call(call, result)
               return result
           except Exception as e:
               self.logger.fail_call(call, str(e))
               raise
   ```
3. 在 `orchestrator.py` 的 `__init__` 中创建实例
4. 在 `orchestrator.py` 的 `run()` 方法中添加调用步骤
5. 在 `planning_agent.py` 的 `ReActPlanner.AVAILABLE_AGENTS` 中注册
6. (可选) 在 `config.py` 中添加对应的配置类

### 9.2 更换 LLM 提供商

**只需修改 `.env` 文件**：
```env
LLM_API_KEY=your-new-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o
CODE_MODEL=gpt-4o
```

如果新提供商的 API 不完全兼容 OpenAI，需修改 `tools/llm_client.py`。

### 9.3 修改报告格式/内容

**修改报告章节结构**：
- 编辑 `agents/evaluation_agent.py` 的 `_build_rich_report()` 方法
- 修改 `sections` 列表的 heading 和 body

**修改 Word 排版**：
- 编辑 `tools/report_generator.py` 的 `_to_docx()` 方法
- 调整字体、字号、颜色、缩进等

**添加新报告格式（如 PDF）**：
- 在 `ReportGenerator` 中添加 `_to_pdf()` 方法
- 在 `generate()` 方法的 format 分支中添加 "pdf" 处理

### 9.4 修改代码安全策略

**允许更多模块**：
- 编辑 `tools/code_executor.py` 的 `_build_safe_globals()` 方法
- 在 `ALLOWED_IMPORTS` 集合中添加模块名

**禁止更多操作**：
- 编辑 `ASTSafetyChecker` 的 `FORBIDDEN_CALLS` 和 `FORBIDDEN_IMPORTS`

### 9.5 修改文献检索源

**替换为其他学术数据库**：
1. 创建新的检索工具类（如 `SemanticScholarTool`），实现 `search()` 方法返回 `Paper` 列表
2. 在 `orchestrator.py` 中替换 `ArXivSearchTool` 为新工具
3. 传入 `LiteratureAgent` 构造函数

**扩展中英文翻译词典**：
- 编辑 `tools/arxiv_search.py` 的 `_CN_TO_EN` 字典

### 9.6 修改规划策略

**调整 ToT 参数**：
- `config.py` 中修改 `PlanningConfig.branching_factor` (路径数) 和 `max_depth`
- `branching_factor=1` 可大幅减少 LLM 调用但降低规划质量

**修改 ReAct 分解逻辑**：
- 编辑 `planning_agent.py` 的 `ReActPlanner.decompose()` 方法中的 system prompt

### 9.7 添加新的统计检验

编辑 `agents/analysis_agent.py` 的 `_run_statistical_analysis()` 方法：
```python
if intent == "your_new_intent" and len(num_cols) >= 2:
    stat, p = stats.your_test(data1, data2)
    results.append({
        "test": "Your Test Name",
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "conclusion": "..."
    })
```

### 9.8 修改记忆系统

**切换向量模型**：
- 修改 `LiteratureConfig.embedding_model`（当前 `all-MiniLM-L6-v2`）
- 或在 `MemoryStore.__init__` 中修改 `embedding_model` 参数

**切换持久化后端（如 Redis/PostgreSQL）**：
- 替换 `_load_long_term()` 和 `save_long_term()` 的文件 I/O 为数据库操作
- 替换 `_vector_search()` 为 pgvector 或 Milvus 检索

---

## 10. 关键设计决策与权衡

### 10.1 同步串行 vs 异步并发

**当前**：编排器同步串行调用各 Agent（Step 1 → 2 → 3 → 4 → 5 → 6）。
**原因**：步骤间存在数据依赖（文献结果 → 实验设计 → 代码生成 → 分析）。
**改进方向**：Step 2 和 Step 3 可以并行（文献检索和实验设计部分独立）。

### 10.2 Mock 优先保活 vs 严格失败

**当前**：任何 LLM/网络失败都降级为 Mock 继续运行，不抛异常。
**优点**：开发调试方便，演示不中断。
**缺点**：可能掩盖真实错误。生产环境建议添加严格模式开关。

### 10.3 子进程执行 vs 受限命名空间

**当前**：实际使用 `execute_with_subprocess()`（子进程）。
**优点**：真正隔离，可以 import 任何允许的包。
**缺点**：无法捕获 return_value，只能通过 stdout 传递结果。

### 10.4 知识图谱的轻量实现

**当前**：纯 dict + list 实现（无 NetworkX/Neo4j 依赖）。
**优点**：零依赖，启动快。
**缺点**：不支持复杂图算法。若需 PageRank 等高级分析，建议迁移到 NetworkX。

---

## 11. 测试体系

**测试文件**：

| 文件 | 覆盖模块 | 测试数 |
|------|---------|--------|
| `tests/test_planning.py` | PlanningAgent, SubTask, ResearchPlan | 7 |
| `tests/test_literature.py` | ArXivSearchTool, LightKnowledgeGraph, LiteratureAgent | 10 |
| `tests/test_code_eval.py` | ASTSafetyChecker, CodeExecutor, CodeAgent, LLMJudge, MetricsCalculator | 17 |

**运行测试**：
```bash
cd mindpilot
python -m pytest tests/ -v
# 或
python tests/test_planning.py
```

**注意**：测试依赖 `CONFIG`（可能触发 Mock 模式），不需要真实 API Key 也能运行。

---

## 附录：快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp config.example.env .env
# 编辑 .env 填入 API Key

# 3. 运行
python main.py

# 4. 运行演示
python examples/demo.py --demo 1   # 完整流程
python examples/demo.py --demo 2   # 仅文献检索
python examples/demo.py --demo 3   # 仅代码生成
python examples/demo.py --demo 4   # 仅数据分析
python examples/demo.py --demo 5   # 基准对比测试
```
