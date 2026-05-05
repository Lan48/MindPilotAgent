"""
模块① — 任务规划 Agent
========================
ReAct + Tree-of-Thought。

改进点（v2）：
  1. ToT 路径批量评分：一次 LLM 调用对所有路径打分，减少 N-1 次 API 调用
  2. 循环依赖检测：_validate_dag 增加真正的拓扑排序，防止死锁
  3. 历史记忆复用：将检索到的历史计划注入 LLM 上下文，避免重复规划
  4. 评分 fallback 改为基于路径内容的启发式规则，移除不可靠的 random
  5. AVAILABLE_AGENTS 改为类属性并提供注册接口，便于扩展同步
  6. 新增 PlanningResult 携带 ToT 路径明细，供 Orchestrator 记录
"""

import json
import time
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class SubTask:
    task_id: str
    name: str
    agent: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    priority: int = 0
    estimated_time: str = "unknown"
    status: str = "pending"


@dataclass
class ResearchPlan:
    plan_id: str
    query: str
    tasks: list[SubTask]
    reasoning: str
    selected_path: Optional[str] = None
    all_paths: list[dict] = field(default_factory=list)   # ← 新增：保存所有路径评分明细
    created_at: float = field(default_factory=time.time)
    from_memory: bool = False                              # ← 新增：是否来自历史记忆复用

    def to_dict(self) -> dict:
        return {
            "plan_id":       self.plan_id,
            "query":         self.query,
            "tasks":         [t.__dict__ for t in self.tasks],
            "reasoning":     self.reasoning,
            "selected_path": self.selected_path,
            "all_paths":     self.all_paths,
            "from_memory":   self.from_memory,
        }


class TreeOfThoughtPlanner:
    def __init__(self, llm_client, branching_factor=3, max_depth=3, logger=None):
        self.llm = llm_client
        self.branching_factor = branching_factor
        self.max_depth = max_depth
        self.logger = logger

    def search(self, query: str) -> tuple[list[dict], str]:
        if self.logger:
            self.logger.info("PlanningAgent",
                f"ToT 搜索开始，分支因子={self.branching_factor}")
        paths  = self._generate_paths(query)
        # 改进①：批量评分，一次 LLM 调用替代原来的 N 次
        scored = self._score_paths_batch(query, paths)
        best   = max(scored, key=lambda x: x["score"])
        if self.logger:
            self.logger.info("PlanningAgent",
                f"ToT 完成，{len(paths)} 条路径，最优得分={best['score']:.2f}")
        return scored, best["description"]

    def _generate_paths(self, query: str) -> list[dict]:
        system = (
            "你是资深科研方法论专家。请为以下科研问题生成不同侧重点的研究路径方案，"
            f"生成 {self.branching_factor} 条，每条路径应包含：文献调研、实验设计、"
            "代码实现、数据分析、报告撰写五个环节。"
            f"返回 JSON 数组：[{{\"id\":\"P1\",\"description\":\"...\",\"steps\":[\"...\"]}}]"
        )
        resp = self.llm.chat([
            {"role": "system", "content": system},
            {"role": "user",   "content": f"科研问题：{query}"}
        ])
        try:
            data = json.loads(self._extract_json(resp))
            if isinstance(data, list) and data:
                return data
        except Exception:
            pass
        # fallback 默认路径
        return [
            {"id": "P1",
             "description": "文献驱动：全面调研→理论推导→实验设计→代码实现→结果分析",
             "steps": ["文献检索与综述", "理论分析", "实验设计", "代码实现", "数据分析与报告"]},
            {"id": "P2",
             "description": "实验驱动：快速原型→文献补充→迭代优化→对比实验→报告",
             "steps": ["快速原型实现", "文献对照补充", "对比实验设计", "代码迭代", "结果报告"]},
            {"id": "P3",
             "description": "数据驱动：数据分析→规律发现→假设建立→文献对照→代码验证",
             "steps": ["数据探索", "模式分析", "假设建立", "文献验证", "代码实现与报告"]},
        ]

    # ── 改进①：批量评分，一次调用替换原来 N 次 ──────────────────────────
    def _score_paths_batch(self, query: str, paths: list[dict]) -> list[dict]:
        """
        原实现：对每条路径单独调用一次 LLM → branching_factor 次额外调用。
        新实现：一次调用让 LLM 对所有路径同时打分，节省 N-1 次调用。
        fallback：使用启发式规则打分（不再用 random），保证确定性。
        """
        paths_json = json.dumps(paths, ensure_ascii=False, indent=2)
        system = (
            "你是科研方法评审专家。请对以下所有研究路径打分（score: 0~1），"
            "考虑：可行性、完整性、创新性。"
            "返回 JSON 数组，保留原有字段并追加 score 和 score_reason：\n"
            "[{\"id\":\"P1\", ..., \"score\": 0.85, \"score_reason\": \"...\"}]"
        )
        resp = self.llm.chat([
            {"role": "system", "content": system},
            {"role": "user",
             "content": f"科研问题：{query}\n\n待评路径：\n{paths_json}"}
        ])
        try:
            data = json.loads(self._extract_json(resp))
            if isinstance(data, list) and len(data) == len(paths):
                # 确保每条记录有 score
                for item in data:
                    item["score"] = float(item.get("score", 0.7))
                return data
        except Exception:
            pass
        # ── fallback：启发式规则（移除原来的 random）──
        return self._heuristic_score(query, paths)

    def _heuristic_score(self, query: str, paths: list[dict]) -> list[dict]:
        """
        确定性 fallback：根据路径描述关键词赋予固定分差，
        保证每次结果一致，且不同路径有区分度。
        """
        keywords_bonus = {
            "文献": 0.05, "实验": 0.05, "数据": 0.04,
            "对比": 0.03, "迭代": 0.03, "原型": 0.02,
        }
        base = 0.70
        scored = []
        for i, path in enumerate(paths):
            desc = path.get("description", "")
            bonus = sum(v for k, v in keywords_bonus.items() if k in desc)
            # 给第一条路径微弱偏好（文献驱动通常更全面），但可被关键词覆盖
            order_bonus = 0.02 if i == 0 else 0.0
            score = round(min(base + bonus + order_bonus, 0.95), 2)
            scored.append({**path, "score": score, "score_reason": "启发式评分（LLM不可用）"})
        return scored

    def _extract_json(self, text: str) -> str:
        m = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
        if m:
            return m.group(1).strip()
        m = re.search(r"(\[[\s\S]+\]|\{[\s\S]+\})", text)
        return m.group(1) if m else text


class ReActPlanner:
    # 改进⑤：改为类级别可变字典 + 注册接口，方便动态扩展而不改代码
    _AGENT_REGISTRY: dict[str, str] = {
        "LiteratureAgent":  "文献检索、知识图谱、结构化摘要生成",
        "CodeAgent":        "代码生成、算法实现、自动调试与测试",
        "AnalysisAgent":    "数据分析、统计检验、可视化图表生成",
        "EvaluationAgent":  "实验设计、结果评估、完整报告生成",
    }

    @classmethod
    def register_agent(cls, name: str, description: str):
        """动态注册新 Agent，无需修改源码"""
        cls._AGENT_REGISTRY[name] = description

    def __init__(self, llm_client, logger=None):
        self.llm = llm_client
        self.logger = logger

    def decompose(
        self,
        query: str,
        research_path: str = "",
        memory_context: str = "",      # ← 改进③：接受历史记忆上下文
    ) -> tuple[list[SubTask], str]:
        agents_desc = "\n".join(
            f"- {name}: {desc}" for name, desc in self._AGENT_REGISTRY.items()
        )
        # 改进③：若有历史记忆，注入到 system prompt
        memory_section = ""
        if memory_context:
            memory_section = f"\n\n参考历史相似计划（可借鉴但需结合当前问题调整）：\n{memory_context}\n"

        system = f"""你是科研项目管理专家。将科研问题分解为5~7个具体子任务，
必须包含：文献检索、实验设计、代码实现、数据分析、报告生成五个环节。

可用 Agent:
{agents_desc}
{memory_section}
输出 JSON（严格格式）:
{{
  "tasks": [
    {{"id":"T1","name":"文献检索与综述","agent":"LiteratureAgent",
      "description":"详细描述该任务的具体目标和预期输出","depends_on":[],"priority":3,"estimated_time":"5min"}},
    ...
  ],
  "reasoning": "分解思路详述（200字以上），说明各任务的必要性和衔接逻辑"
}}

规则：
1. description 字段必须详细（50字以上），明确说明预期输出内容
2. 必须包含实验设计任务（分配给 EvaluationAgent）
3. depends_on 引用已存在的任务 ID，无循环依赖
"""
        context = f"科研问题：{query}"
        if research_path:
            context += f"\n推荐研究路径：{research_path}"

        resp = self.llm.chat([
            {"role": "system", "content": system},
            {"role": "user",   "content": context}
        ])
        try:
            m    = re.search(r"\{[\s\S]+\}", resp)
            data = json.loads(m.group(0) if m else resp)
            tasks = []
            for t in data.get("tasks", []):
                tasks.append(SubTask(
                    task_id=t.get("id", f"T{len(tasks)+1}"),
                    name=t.get("name", "未命名"),
                    agent=t.get("agent", "LiteratureAgent"),
                    description=t.get("description", ""),
                    depends_on=t.get("depends_on", []),
                    priority=t.get("priority", 1),
                    estimated_time=t.get("estimated_time", "unknown"),
                ))
            reasoning = data.get("reasoning", "")
            if self.logger:
                self.logger.success("PlanningAgent", f"分解完成：{len(tasks)} 个子任务")
            return tasks, reasoning
        except Exception as e:
            if self.logger:
                self.logger.warning("PlanningAgent", f"解析失败，使用默认模板: {e}")
            return self._default_tasks(query), "使用默认研究流程模板"

    def _default_tasks(self, query: str) -> list[SubTask]:
        return [
            SubTask("T1", "文献检索与综述", "LiteratureAgent",
                    f"检索与「{query[:40]}」相关的学术文献，生成结构化摘要和知识图谱", [], 3, "5min"),
            SubTask("T2", "实验设计", "EvaluationAgent",
                    "基于文献综述，设计完整的实验方案，包括实验目标、对照组、评估指标", ["T1"], 3, "3min"),
            SubTask("T3", "核心代码实现", "CodeAgent",
                    "根据实验设计实现核心算法，包含完整注释、错误处理和单元测试", ["T2"], 2, "8min"),
            SubTask("T4", "数据分析与可视化", "AnalysisAgent",
                    "对代码运行结果进行统计分析，生成图表并给出量化结论", ["T3"], 2, "5min"),
            SubTask("T5", "综合报告生成", "EvaluationAgent",
                    "整合各模块输出，生成包含背景、方法、结果、结论的完整学术报告",
                    ["T1", "T2", "T3", "T4"], 1, "5min"),
        ]


class PlanningAgent:
    AGENT_NAME = "PlanningAgent"

    def __init__(self, config, llm_client, memory_store, logger):
        self.config  = config
        self.llm     = llm_client
        self.memory  = memory_store
        self.logger  = logger
        self.tot     = TreeOfThoughtPlanner(
            llm_client,
            branching_factor=config.planning.branching_factor,
            max_depth=config.planning.max_depth,
            logger=logger,
        )
        self.react = ReActPlanner(llm_client, logger=logger)

    def run(self, query: str) -> ResearchPlan:
        import uuid
        call = self.logger.start_call(self.AGENT_NAME, "planning", query)
        try:
            # ── 改进③：将历史记忆格式化后传入 ReAct ──────────────────
            similar = self.memory.search(query, top_k=self.config.planning.memory_top_k)
            memory_context = ""
            if similar:
                self.logger.info(self.AGENT_NAME, f"记忆检索：找到 {len(similar)} 条相关历史记录")
                snippets = []
                for item in similar[:3]:          # 最多取 3 条，避免 prompt 过长
                    # item 是 MemoryEntry 对象（dataclass），用属性访问而非 .get()
                    payload = item.payload if hasattr(item, "payload") else {}
                    if isinstance(payload, dict) and "tasks" in payload:
                        task_names = [t.get("name", "") for t in payload["tasks"]]
                        snippets.append(
                            f"- 历史问题：{payload.get('query', '')[:60]}\n"
                            f"  任务结构：{' -> '.join(task_names)}"
                        )
                memory_context = "\n".join(snippets)

            paths, best_path = self.tot.search(query)
            # 传入 memory_context
            tasks, reasoning = self.react.decompose(query, best_path, memory_context)
            # ── 改进②：真正的 DAG 验证（含循环依赖检测）──────────────
            tasks, dag_warning = self._validate_dag(tasks)
            if dag_warning and self.logger:
                self.logger.warning(self.AGENT_NAME, dag_warning)

            from_memory = bool(memory_context)    # 标记是否复用了历史计划
            plan = ResearchPlan(
                plan_id=str(uuid.uuid4())[:8],
                query=query,
                tasks=tasks,
                reasoning=reasoning,
                selected_path=best_path,
                all_paths=paths,                  # 保存所有路径明细
                from_memory=from_memory,
            )
            self.memory.add(
                content=f"研究计划: {query}",
                agent=self.AGENT_NAME,
                payload=plan.to_dict(),
                tags=["plan"],
                importance=1.5,
            )
            self.logger.finish_call(call, plan.to_dict())
            return plan
        except Exception as e:
            self.logger.fail_call(call, str(e))
            raise

    # ── 改进②：真正的 DAG 验证 ──────────────────────────────────────────
    def _validate_dag(self, tasks: list[SubTask]) -> tuple[list[SubTask], str]:
        """
        原实现：只过滤不存在的依赖 ID，不检测循环。
        新实现：
          1. 过滤不存在的依赖 ID（原有逻辑）
          2. 拓扑排序检测循环依赖（Kahn 算法）
          3. 若检测到循环，裁剪成环中的反向边，并返回警告信息
        """
        task_ids = {t.task_id for t in tasks}
        warning = ""

        # Step 1：过滤不存在的 ID（原有逻辑）
        for task in tasks:
            task.depends_on = [d for d in task.depends_on if d in task_ids]

        # Step 2：Kahn 拓扑排序检测循环
        in_degree: dict[str, int] = {t.task_id: 0 for t in tasks}
        graph: dict[str, list[str]] = {t.task_id: [] for t in tasks}
        for task in tasks:
            for dep in task.depends_on:
                graph[dep].append(task.task_id)
                in_degree[task.task_id] += 1

        queue = deque([tid for tid, deg in in_degree.items() if deg == 0])
        visited_order: list[str] = []
        while queue:
            node = queue.popleft()
            visited_order.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Step 3：若有节点未被访问，说明存在循环
        cycle_nodes = task_ids - set(visited_order)
        if cycle_nodes:
            warning = (
                f"检测到循环依赖，涉及任务：{cycle_nodes}。"
                "已自动裁剪这些任务的全部依赖以打破循环。"
            )
            for task in tasks:
                if task.task_id in cycle_nodes:
                    task.depends_on = []   # 打破循环：清空这些任务的依赖

        return tasks, warning

    def print_plan(self, plan: ResearchPlan):
        print(f"\n{'━'*60}")
        print(f"  📋 研究计划  [{plan.plan_id}]{'  (复用历史记忆)' if plan.from_memory else ''}")
        print(f"{'━'*60}")
        print(f"  问题: {plan.query[:60]}")
        print(f"  路径: {(plan.selected_path or 'N/A')[:60]}")
        if plan.all_paths:
            print(f"  路径评分明细:")
            for p in sorted(plan.all_paths, key=lambda x: x.get("score", 0), reverse=True):
                marker = "★" if p.get("description") == plan.selected_path else " "
                print(f"    {marker} [{p.get('id','?')}] score={p.get('score', '?'):.2f}  {p.get('description','')[:45]}")
        print(f"{'━'*60}")
        for t in plan.tasks:
            deps = f" (依赖: {', '.join(t.depends_on)})" if t.depends_on else ""
            print(f"  [{t.task_id}] {t.name:18s} → {t.agent:18s}{deps}")
        print(f"{'━'*60}")
        print(f"  💡 {plan.reasoning[:100]}")
        print(f"{'━'*60}\n")
        