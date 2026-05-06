"""
模块① 任务规划 Agent — 单元测试
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import MagicMock
from config import CONFIG
from agents.planning_agent import PlanningAgent, SubTask, ResearchPlan
from tools.llm_client import LLMClient
from memory.memory_store import MemoryStore
from framework.logger import MindPilotLogger


def make_agent():
    logger = MindPilotLogger(session_id="test_plan", verbose=False)
    llm = LLMClient(CONFIG)
    memory = MemoryStore(logger=logger)
    return PlanningAgent(CONFIG, llm, memory, logger)


class TestPlanningAgent(unittest.TestCase):

    def setUp(self):
        self.agent = make_agent()

    def test_run_returns_plan(self):
        plan = self.agent.run("研究注意力机制")
        self.assertIsInstance(plan, ResearchPlan)
        self.assertIsNotNone(plan.plan_id)
        self.assertIsNotNone(plan.query)

    def test_plan_has_tasks(self):
        plan = self.agent.run("实现线性回归")
        self.assertGreater(len(plan.tasks), 0)

    def test_tasks_are_subtask_instances(self):
        plan = self.agent.run("分析 BERT 模型")
        for task in plan.tasks:
            self.assertIsInstance(task, SubTask)
            self.assertIn(task.agent, [
                "LiteratureAgent", "CodeAgent", "AnalysisAgent", "EvaluationAgent"
            ])

    def test_dag_has_no_self_loops(self):
        plan = self.agent.run("研究梯度下降")
        for task in plan.tasks:
            self.assertNotIn(task.task_id, task.depends_on,
                             f"任务 {task.task_id} 不应依赖自身")

    def test_depends_on_valid_ids(self):
        plan = self.agent.run("对比优化器性能")
        task_ids = {t.task_id for t in plan.tasks}
        for task in plan.tasks:
            for dep in task.depends_on:
                self.assertIn(dep, task_ids,
                              f"依赖 {dep} 不在任务列表中")

    def test_plan_to_dict(self):
        plan = self.agent.run("文献综述")
        d = plan.to_dict()
        self.assertIn("plan_id", d)
        self.assertIn("tasks", d)
        self.assertIn("reasoning", d)

    def test_memory_stores_plan(self):
        plan = self.agent.run("测试记忆存储")
        stats = self.agent.memory.stats()
        self.assertGreater(stats["short_term"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
