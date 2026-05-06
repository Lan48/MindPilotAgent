"""
模块③ 代码生成 + 模块⑥ 评估 — 单元测试
"""
import json
import os
import re
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from pathlib import Path
from MindPilotAgent.mindpilot.config import CONFIG
from MindPilotAgent.mindpilot.tools.code_executor import CodeExecutor, ASTSafetyChecker
from MindPilotAgent.mindpilot.agents.code_agent import CodeAgent
from MindPilotAgent.mindpilot.agents.evaluation_agent import EvaluationAgent, LLMJudge, EvalScore
from MindPilotAgent.mindpilot.tools.llm_client import LLMClient
from MindPilotAgent.mindpilot.tools.report_generator import ReportGenerator
from MindPilotAgent.mindpilot.memory.memory_store import MemoryStore
from MindPilotAgent.mindpilot.framework.logger import MindPilotLogger
from MindPilotAgent.mindpilot.evaluation.benchmark import MetricsCalculator


class StubLLM:
    def chat(self, messages, **kwargs):
        system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
        last = messages[-1].get("content", "") if messages else ""
        if "数据集生成器" in system:
            return json.dumps({
                "dataset_name": "stub_experiment_dataset",
                "description": "用于测试的本地实验数据集。",
                "columns": [
                    {"name": "sample_id", "type": "int"},
                    {"name": "task_text", "type": "str"},
                    {"name": "feature_a", "type": "float"},
                    {"name": "feature_b", "type": "float"},
                    {"name": "label", "type": "str"},
                ],
                "rows": [
                    {"sample_id": 1, "task_text": "a", "feature_a": 1.0, "feature_b": 2.0, "label": "train"},
                    {"sample_id": 2, "task_text": "b", "feature_a": 2.0, "feature_b": 3.0, "label": "train"},
                    {"sample_id": 3, "task_text": "c", "feature_a": 3.0, "feature_b": 4.0, "label": "test"},
                    {"sample_id": 4, "task_text": "d", "feature_a": 4.0, "feature_b": 5.0, "label": "test"},
                    {"sample_id": 5, "task_text": "e", "feature_a": 5.0, "feature_b": 6.0, "label": "train"},
                    {"sample_id": 6, "task_text": "f", "feature_a": 6.0, "feature_b": 7.0, "label": "train"},
                    {"sample_id": 7, "task_text": "g", "feature_a": 7.0, "feature_b": 8.0, "label": "train"},
                    {"sample_id": 8, "task_text": "h", "feature_a": 8.0, "feature_b": 9.0, "label": "train"},
                    {"sample_id": 9, "task_text": "i", "feature_a": 9.0, "feature_b": 10.0, "label": "test"},
                    {"sample_id": 10, "task_text": "j", "feature_a": 10.0, "feature_b": 11.0, "label": "test"},
                    {"sample_id": 11, "task_text": "k", "feature_a": 11.0, "feature_b": 12.0, "label": "test"},
                    {"sample_id": 12, "task_text": "l", "feature_a": 12.0, "feature_b": 13.0, "label": "test"},
                ],
            }, ensure_ascii=False)
        return last

    def chat_code(self, messages, **kwargs):
        last = messages[-1].get("content", "") if messages else ""
        match = re.search(r"CSV 路径：([^\n]+)", last)
        csv_path = match.group(1).strip() if match else ""
        return (
            "```python\n"
            "import pandas as pd\n"
            f"df = pd.read_csv(r'{csv_path}')\n"
            "print(len(df))\n"
            "__result__ = len(df)\n"
            "```"
        )


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

    def test_execute_can_read_file(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as tmp_file:
            tmp_file.write("mindpilot-dataset")
            tmp_path = tmp_file.name

        try:
            code = (
                f"with open(r'{tmp_path}', 'r', encoding='utf-8') as f:\n"
                "    content = f.read()\n"
                "print(content)\n"
                "__result__ = content\n"
            )
            result = self.executor.execute(code)
            self.assertTrue(result.success)
            self.assertIn("mindpilot-dataset", result.stdout)
            self.assertEqual(result.return_value, "mindpilot-dataset")
        finally:
            os.unlink(tmp_path)

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
        llm = StubLLM()
        executor = CodeExecutor(timeout=15, logger=logger)
        memory = MemoryStore(logger=logger)
        self.agent = CodeAgent(CONFIG, llm, executor, memory, logger)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.agent.dataset_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dataset_is_created_before_code(self):
        dataset_info = self.agent._prepare_dataset("读取实验数据并计算统计量", "", {}, "task-001")
        self.assertTrue(Path(dataset_info["csv_path"]).exists())
        self.assertTrue(Path(dataset_info["json_path"]).exists())
        self.assertGreaterEqual(dataset_info["row_count"], 12)
        self.assertIn("sample_id", dataset_info["column_names"])

    def test_run_returns_dict(self):
        result = self.agent.run("计算 1+1 并打印结果")
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        self.assertIn("final_code", result)
        self.assertIn("dataset_path", result)

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
