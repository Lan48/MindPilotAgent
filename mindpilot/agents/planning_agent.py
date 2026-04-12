"""
模块① — 任务规划 Agent
========================
ReAct + Tree-of-Thought。
强化版：prompt 要求输出完整、详细的科研计划，覆盖实验设计环节。
"""

import json
import time
import re
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
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "query": self.query,
            "tasks": [t.__dict__ for t in self.tasks],
            "reasoning": self.reasoning,
            "selected_path": self.selected_path,
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
        paths   = self._generate_paths(query)
        scored  = self._score_paths(query, paths)
        best    = max(scored, key=lambda x: x["score"])
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

    def _score_paths(self, query: str, paths: list[dict]) -> list[dict]:
        scored = []
        for path in paths:
            system = (
                "你是科研方法评审专家。请对以下研究路径评分（0-1），"
                "考虑：可行性、完整性、创新性。"
                "返回 JSON: {\"score\": 0.85, \"reasoning\": \"...\"}"
            )
            resp = self.llm.chat([
                {"role": "system", "content": system},
                {"role": "user",
                 "content": f"问题：{query}\n路径：{json.dumps(path, ensure_ascii=False)}"}
            ])
            try:
                result = json.loads(self._extract_json(resp))
                score = float(result.get("score", 0.7))
            except Exception:
                import random
                score = round(random.uniform(0.6, 0.9), 2)
            scored.append({**path, "score": score})
        return scored

    def _extract_json(self, text: str) -> str:
        m = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
        if m:
            return m.group(1).strip()
        m = re.search(r"(\[[\s\S]+\]|\{[\s\S]+\})", text)
        return m.group(1) if m else text


class ReActPlanner:
    AVAILABLE_AGENTS = {
        "LiteratureAgent":  "文献检索、知识图谱、结构化摘要生成",
        "CodeAgent":        "代码生成、算法实现、自动调试与测试",
        "AnalysisAgent":    "数据分析、统计检验、可视化图表生成",
        "EvaluationAgent":  "实验设计、结果评估、完整报告生成",
    }

    def __init__(self, llm_client, logger=None):
        self.llm = llm_client
        self.logger = logger

    def decompose(self, query: str, research_path: str = "") -> tuple[list[SubTask], str]:
        agents_desc = "\n".join(
            f"- {name}: {desc}" for name, desc in self.AVAILABLE_AGENTS.items()
        )
        system = f"""你是科研项目管理专家。将科研问题分解为5~7个具体子任务，
必须包含：文献检索、实验设计、代码实现、数据分析、报告生成五个环节。

可用 Agent:
{agents_desc}

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
                    "整合各模块输出，生成包含背景、方法、结果、结论的完整学术报告", ["T1","T2","T3","T4"], 1, "5min"),
        ]


class PlanningAgent:
    AGENT_NAME = "PlanningAgent"

    def __init__(self, config, llm_client, memory_store, logger):
        self.config  = config
        self.llm     = llm_client
        self.memory  = memory_store
        self.logger  = logger
        # 生成三条研究路径，并选出一条得分最高的路径
        self.tot     = TreeOfThoughtPlanner(
            llm_client,
            branching_factor=config.planning.branching_factor,
            max_depth=config.planning.max_depth,
            logger=logger,
        )
        # 根据上述生成的推荐研究路径，分解成5-7个任务流程
        self.react = ReActPlanner(llm_client, logger=logger)

    def run(self, query: str) -> ResearchPlan:
        import uuid
        call = self.logger.start_call(self.AGENT_NAME, "planning", query)
        try:
            similar = self.memory.search(query, top_k=self.config.planning.memory_top_k)
            if similar and self.logger:
                self.logger.info(self.AGENT_NAME, f"记忆检索：找到 {len(similar)} 条相关历史记录")

            paths, best_path = self.tot.search(query)
            tasks, reasoning = self.react.decompose(query, best_path)
            tasks = self._validate_dag(tasks)

            plan = ResearchPlan(
                plan_id=str(uuid.uuid4())[:8],
                query=query,
                tasks=tasks,
                reasoning=reasoning,
                selected_path=best_path,
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

    def _validate_dag(self, tasks: list[SubTask]) -> list[SubTask]:
        task_ids = {t.task_id for t in tasks}
        for task in tasks:
            task.depends_on = [d for d in task.depends_on if d in task_ids]
        return tasks

    def print_plan(self, plan: ResearchPlan):
        print(f"\n{'━'*60}")
        print(f"  📋 研究计划  [{plan.plan_id}]")
        print(f"{'━'*60}")
        print(f"  问题: {plan.query[:60]}")
        print(f"  路径: {(plan.selected_path or 'N/A')[:60]}")
        print(f"{'━'*60}")
        for t in plan.tasks:
            deps = f" (依赖: {', '.join(t.depends_on)})" if t.depends_on else ""
            print(f"  [{t.task_id}] {t.name:18s} → {t.agent:18s}{deps}")
        print(f"{'━'*60}")
        print(f"  💡 {plan.reasoning[:100]}")
        print(f"{'━'*60}\n")
