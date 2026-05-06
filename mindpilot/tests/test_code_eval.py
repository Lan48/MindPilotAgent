"""
模块③ 代码生成 + 模块⑥ 评估 — 单元测试
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from config import CONFIG
from tools.code_executor import CodeExecutor, ASTSafetyChecker
from agents.code_agent import CodeAgent
from agents.evaluation_agent import EvaluationAgent, LLMJudge, EvalScore
from tools.llm_client import LLMClient
from tools.report_generator import ReportGenerator
from memory.memory_store import MemoryStore
from framework.logger import MindPilotLogger
from evaluation.benchmark import MetricsCalculator


# ── 模块③ 测试 ─────────────────────────────────────────────

class TestASTSafetyChecker(unittest.TestCase):
    def setUp(self):
        self.checker = ASTSafetyChecker()

    def test_safe_code_passes(self):
        code = "import numpy as np\nx = np.array([1,2,3])\nprint(x.mean())"
        issues = self.checker.check(code)
        self.assertEqual(len(issues), 0)

    def test_os_system_blocked(self):
        code = "import os\nos.system('rm -rf /')"
        issues = self.checker.check(code)
        self.assertGreater(len(issues), 0)

    def test_subprocess_blocked(self):
        code = "import subprocess\nsubprocess.run(['ls'])"
        issues = self.checker.check(code)
        self.assertGreater(len(issues), 0)

    def test_eval_blocked(self):
        code = "eval('print(1)')"
        issues = self.checker.check(code)
        self.assertGreater(len(issues), 0)

    def test_syntax_error_detected(self):
        code = "def foo(:\n  pass"
        issues = self.checker.check(code)
        self.assertGreater(len(issues), 0)


class TestCodeExecutor(unittest.TestCase):
    def setUp(self):
        self.executor = CodeExecutor(timeout=10)

    def test_execute_simple_code(self):
        code = "x = 1 + 1\nprint(x)"
        result = self.executor.execute(code)
        self.assertTrue(result.success)
        self.assertIn("2", result.stdout)

    def test_execute_numpy_code(self):
        code = "import numpy as np\narr = np.array([1,2,3,4,5])\nprint(arr.mean())"
        result = self.executor.execute(code)
        self.assertTrue(result.success)
        self.assertIn("3.0", result.stdout)

    def test_execute_error_code(self):
        code = "x = 1 / 0"
        result = self.executor.execute(code)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_type)

    def test_extract_code_from_markdown(self):
        text = "Here is the code:\n```python\nprint('hello')\n```"
        code = self.executor.extract_code(text)
        self.assertIn("print", code)
        self.assertNotIn("```", code)

    def test_execution_time_recorded(self):
        result = self.executor.execute("import time\ntime.sleep(0.01)\nprint('done')")
        self.assertIsNotNone(result.execution_time)
        self.assertGreater(result.execution_time, 0)


class TestCodeAgent(unittest.TestCase):
    def setUp(self):
        logger = MindPilotLogger(session_id="test_code", verbose=False)
        llm = LLMClient(CONFIG)
        executor = CodeExecutor(timeout=15, logger=logger)
        memory = MemoryStore(logger=logger)
        self.agent = CodeAgent(CONFIG, llm, executor, memory, logger)

    def test_run_returns_dict(self):
        result = self.agent.run("计算 1+1 并打印结果")
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        self.assertIn("final_code", result)

    def test_result_has_iterations(self):
        result = self.agent.run("打印 Hello MindPilot")
        self.assertIn("iterations", result)
        self.assertIsInstance(result["iterations"], list)

    def test_pass_at_1_is_bool(self):
        result = self.agent.run("计算斐波那契数列前10项")
        self.assertIsInstance(result["pass_at_1"], bool)


# ── 模块⑥ 测试 ─────────────────────────────────────────────

class TestLLMJudge(unittest.TestCase):
    def setUp(self):
        logger = MindPilotLogger(session_id="test_eval", verbose=False)
        llm = LLMClient(CONFIG)
        self.judge = LLMJudge(llm, threshold=0.65, logger=logger)

    def test_score_returns_eval_score(self):
        score = self.judge.score("研究注意力机制", "注意力机制通过 Q、K、V 矩阵计算...")
        self.assertIsInstance(score, EvalScore)

    def test_score_range(self):
        score = self.judge.score("test", "some output")
        self.assertGreaterEqual(score.overall, 0.0)
        self.assertLessEqual(score.overall, 1.0)

    def test_needs_reflection_logic(self):
        score = self.judge.score("test", "x")
        self.assertEqual(score.needs_reflection, score.overall < 0.65)

    def test_rouge_l_identical(self):
        r = self.judge.compute_rouge_l("hello world test", "hello world test")
        self.assertAlmostEqual(r, 1.0, places=1)

    def test_rouge_l_empty(self):
        r = self.judge.compute_rouge_l("", "")
        self.assertEqual(r, 0.0)


class TestMetricsCalculator(unittest.TestCase):
    def test_keyword_recall_full(self):
        r = MetricsCalculator.keyword_recall("attention softmax transformer", ["attention", "softmax"])
        self.assertEqual(r, 1.0)

    def test_keyword_recall_partial(self):
        r = MetricsCalculator.keyword_recall("attention only", ["attention", "softmax"])
        self.assertEqual(r, 0.5)

    def test_keyword_recall_empty(self):
        r = MetricsCalculator.keyword_recall("anything", [])
        self.assertEqual(r, 0.0)

    def test_pass_at_k(self):
        results = [True, False, False, True, False]
        p1 = MetricsCalculator.pass_at_k(results, 1)
        p5 = MetricsCalculator.pass_at_k(results, 5)
        self.assertGreaterEqual(p5, p1)

    def test_rouge_l_basic(self):
        r = MetricsCalculator.rouge_l("cat sat on mat", "cat sat on mat")
        self.assertGreater(r, 0.9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
