"""
Focused tests for Module 6 EvaluationAgent behavior.

This file keeps EvaluationAgent-specific checks out of test_code_eval.py so
shared code-generation tests stay stable for other contributors.

Run with:
    python mindpilot/tests/test_evaluation_agent.py -q
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.evaluation_agent import EvalScore, EvaluationAgent, LLMJudge
from config import CONFIG


class FakeLLM:
    def __init__(self, response=None):
        self.response = response or (
            '{"overall_score": 0.70, "accuracy": 0.70, "completeness": 0.70, '
            '"format_quality": 0.70, "feedback": "ok", "needs_reflection": false}'
        )

    def chat(self, messages, max_tokens=None):
        return self.response


class FakeLogger:
    def start_call(self, *args, **kwargs):
        return {"args": args, "kwargs": kwargs}

    def finish_call(self, *args, **kwargs):
        return None

    def fail_call(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def success(self, *args, **kwargs):
        return None


class FakeMemory:
    def add(self, *args, **kwargs):
        return None


class FakeReportGenerator:
    def __init__(self):
        self.last_content = None

    def generate(self, content, filename="report", formats=None):
        self.last_content = content
        return {"markdown": "fake.md"}


class FakeJudge:
    def __init__(self, scores):
        self.scores = list(scores)
        self.index = 0

    def score(self, query, output, output_type="report"):
        score = self.scores[min(self.index, len(self.scores) - 1)]
        self.index += 1
        return score


class FakeReflector:
    def __init__(self, payload):
        self.payload = payload

    def reflect_and_revise_report(self, query, report_content, score):
        return self.payload


def build_agent():
    return EvaluationAgent(
        CONFIG,
        llm_client=FakeLLM(),
        report_gen=FakeReportGenerator(),
        memory_store=FakeMemory(),
        logger=FakeLogger(),
    )


def section_map(report):
    return {section["heading"]: section["body"] for section in report["sections"]}


def section_body_contains(report, heading_text):
    for section in report["sections"]:
        if heading_text in section["heading"]:
            return section["body"]
    raise AssertionError(f"Missing section containing: {heading_text}")


class TestLLMJudge(unittest.TestCase):
    def test_score_returns_eval_score(self):
        judge = LLMJudge(FakeLLM(), threshold=0.65, logger=FakeLogger())

        score = judge.score("test query", "test output")

        self.assertIsInstance(score, EvalScore)
        self.assertEqual(score.overall, 0.70)

    def test_needs_reflection_defaults_from_score_threshold(self):
        response = (
            '{"overall_score": 0.50, "accuracy": 0.50, "completeness": 0.50, '
            '"format_quality": 0.50, "feedback": "weak"}'
        )
        judge = LLMJudge(FakeLLM(response), threshold=0.65, logger=FakeLogger())

        score = judge.score("test query", "test output")

        self.assertTrue(score.needs_reflection)

    def test_rouge_l_identical_and_empty(self):
        judge = LLMJudge(FakeLLM(), threshold=0.65, logger=FakeLogger())

        self.assertAlmostEqual(judge.compute_rouge_l("hello world", "hello world"), 1.0, places=1)
        self.assertEqual(judge.compute_rouge_l("", ""), 0.0)


class TestEvaluationAgentReflection(unittest.TestCase):
    def test_reflection_updates_saved_report_content(self):
        agent = build_agent()
        agent._build_rich_report = lambda query, outputs: {
            "title": "Test Report",
            "query": query,
            "abstract": "old abstract",
            "sections": [
                {"heading": "一、背景", "body": "old body", "level": 1},
                {"heading": "二、结论", "body": "old ending", "level": 1},
            ],
        }
        agent.judge = FakeJudge(
            [
                EvalScore(0.50, 0.50, 0.50, 0.50, "needs work", True, "revise"),
                EvalScore(0.90, 0.90, 0.90, 0.90, "much better", False, ""),
            ]
        )
        agent.reflector = FakeReflector(
            {
                "abstract": "new abstract",
                "sections": [
                    {"body": "new body"},
                    {"body": "new ending"},
                ],
            }
        )

        result = agent.run("test query", {})

        self.assertEqual(result["reflection_rounds"], 1)
        self.assertEqual(result["attempted_reflection_rounds"], 1)
        self.assertEqual(agent.report_gen.last_content["abstract"], "new abstract")
        self.assertEqual(agent.report_gen.last_content["sections"][0]["body"], "new body")
        self.assertEqual(agent.report_gen.last_content["evaluation"]["overall_score"], 0.90)

    def test_invalid_reflection_attempt_is_not_counted_as_accepted_round(self):
        agent = build_agent()
        agent._build_rich_report = lambda query, outputs: {
            "title": "Test Report",
            "query": query,
            "abstract": "old abstract",
            "sections": [
                {"heading": "一、背景", "body": "old body", "level": 1},
                {"heading": "二、结论", "body": "old ending", "level": 1},
            ],
        }
        agent.judge = FakeJudge([EvalScore(0.50, 0.50, 0.50, 0.50, "needs work", True, "revise")])
        agent.reflector = FakeReflector({"sections": []})

        result = agent.run("test query", {})

        self.assertEqual(result["reflection_rounds"], 0)
        self.assertEqual(result["attempted_reflection_rounds"], 1)
        self.assertEqual(result["reflection_log"][0]["status"], "invalid_revision")
        self.assertFalse(result["reflection_log"][0]["accepted"])
        self.assertEqual(agent.report_gen.last_content["evaluation"]["reflection_rounds"], 0)
        self.assertEqual(agent.report_gen.last_content["evaluation"]["attempted_reflection_rounds"], 1)


class TestEvaluationAgentReportGeneration(unittest.TestCase):
    def test_report_generation_prompts_reuse_upstream_outputs(self):
        agent = build_agent()
        prompts = []

        def fake_expand_section(prompt, min_words=200):
            prompts.append(prompt)
            return f"section-{len(prompts)}"

        agent._expand_section = fake_expand_section
        agent._build_abstract = lambda query, report_body: "final abstract"

        outputs = {
            "literature_result": {
                "literature_review": "上游综述指出现有方法在鲁棒性和泛化性方面仍存在明显不足。",
                "top_papers": [
                    {
                        "title": "Paper A",
                        "structured_summary": {
                            "method": "使用对比学习增强表征",
                            "conclusion": "在小样本场景下效果更稳定",
                        },
                    }
                ],
            },
            "experiment_design": {
                "research_hypothesis": "引入结构化先验后可提升模型稳定性。",
                "dataset": "Dataset-X，含训练集、验证集和测试集。",
                "baselines": ["Baseline-1", "Baseline-2"],
                "metrics": ["Accuracy", "F1-score"],
                "procedure": ["训练主模型", "与基线比较", "误差分析"],
                "expected_results": "预期在 Accuracy 和 F1-score 上均优于基线。",
                "full_description": "",
            },
            "code_result": {
                "success": True,
                "final_code": "def train_model():\n    return 'ok'\n",
                "stdout": "Accuracy=0.91; F1=0.88",
            },
            "analysis_result": {
                "conclusion": "实验结果表明该方法在核心指标上优于基线。",
                "charts": ["reports/chart_accuracy.png"],
            },
        }

        report = agent._build_rich_report("test query", outputs)

        self.assertEqual(report["abstract"], "final abstract")
        self.assertEqual(len(prompts), 6)
        self.assertTrue(any("上游综述指出现有方法" in prompt for prompt in prompts))
        self.assertTrue(any("Paper A" in prompt for prompt in prompts))
        self.assertTrue(any("Baseline-1" in prompt and "F1-score" in prompt for prompt in prompts))
        self.assertTrue(any("reports/chart_accuracy.png" in prompt for prompt in prompts))
        self.assertTrue(any("def train_model" in prompt for prompt in prompts))
        self.assertTrue(any("结构化先验" in prompt for prompt in prompts))

    def test_experiment_subsections_are_rendered_from_design_fields_only(self):
        agent = build_agent()
        agent._expand_section = lambda prompt, min_words=200: "generated body"
        agent._build_abstract = lambda query, report_body: "final abstract"

        report = agent._build_rich_report(
            "model compression",
            {
                "literature_result": {"literature_review": "lit review", "top_papers": []},
                "experiment_design": {
                    "research_hypothesis": "H1: distilled student keeps perception accuracy.",
                    "metrics": ["mAP", "FPS"],
                    "baselines": ["Vanilla student", "FitNets"],
                    "full_description": "Detailed experiment design from the experiment stage.",
                },
                "code_result": {},
                "analysis_result": {},
            },
        )
        sections = section_map(report)

        self.assertEqual(sections["3.1 实验假设与目标"], "H1: distilled student keeps perception accuracy.")
        self.assertEqual(sections["3.2 评估指标"], "mAP\nFPS")
        self.assertEqual(sections["3.3 基线方法"], "Vanilla student\nFitNets")

    def test_missing_experiment_subsections_are_not_generated_in_report_stage(self):
        agent = build_agent()
        agent._expand_section = lambda prompt, min_words=200: "generated body"
        agent._build_abstract = lambda query, report_body: "final abstract"

        report = agent._build_rich_report(
            "model compression",
            {
                "literature_result": {},
                "experiment_design": {"full_description": "Detailed experiment design from the experiment stage."},
                "code_result": {},
                "analysis_result": {},
            },
        )
        sections = section_map(report)

        self.assertIn("实验设计模块未提供实验假设与目标", sections["3.1 实验假设与目标"])
        self.assertIn("实验设计模块未提供评估指标", sections["3.2 评估指标"])
        self.assertIn("实验设计模块未提供基线方法", sections["3.3 基线方法"])

    def test_failed_code_execution_changes_result_and_conclusion_tone(self):
        agent = build_agent()
        agent._expand_section = lambda prompt, min_words=200: "generated body"
        agent._build_abstract = lambda query, report_body: "final abstract"

        report = agent._build_rich_report(
            "autonomous driving model compression",
            {
                "literature_result": {"literature_review": "literature context", "top_papers": []},
                "experiment_design": {
                    "research_hypothesis": "H1: the compressed student remains accurate.",
                    "metrics": ["mAP", "FPS"],
                    "baselines": ["Vanilla student", "FitNets"],
                    "procedure": ["train", "evaluate"],
                    "expected_results": "The student should be smaller and faster.",
                    "full_description": "Detailed experiment design from the experiment stage.",
                },
                "code_result": {
                    "success": False,
                    "final_code": "print('experiment')",
                    "error": "RuntimeError: dataset not found",
                    "stdout": "",
                },
                "analysis_result": {"charts": []},
            },
        )

        result_body = section_body_contains(report, "实验结果")
        conclusion_body = section_body_contains(report, "结论")

        self.assertIn("代码执行模块返回失败状态", result_body)
        self.assertIn("不能被解释为已经获得的实验结论", result_body)
        self.assertIn("代码执行模块当前返回失败状态", conclusion_body)
        self.assertIn("不能将设计目标或预期结果表述为已经得到验证", conclusion_body)


class TestEvaluationAgentHybridScoring(unittest.TestCase):
    def test_hybrid_scoring_penalizes_ungrounded_report_claims(self):
        agent = build_agent()
        agent.judge = FakeJudge([EvalScore(0.92, 0.92, 0.92, 0.92, "LLM liked it", False, "")])
        report = {
            "title": "Test report",
            "abstract": "summary",
            "sections": [
                {"heading": "1. Background", "body": "已有研究表明该方向已经取得显著进展。"},
                {"heading": "2. Literature", "body": "大量研究支持本文方法。"},
                {"heading": "3. Experiment", "body": "experiment overview"},
                {"heading": "3.1 Hypothesis", "body": "invented hypothesis"},
                {"heading": "3.2 Metrics", "body": "mAP\nFPS"},
                {"heading": "3.3 Baselines", "body": "invented baseline"},
                {"heading": "4. Method", "body": "method"},
                {"heading": "5. Results", "body": "实验结果表明显著提升 10% mAP and FPS."},
                {"heading": "6. Conclusion", "body": "conclusion"},
            ],
        }
        outputs = {
            "literature_result": {"total_found": 0, "top_papers": []},
            "experiment_design": {
                "research_hypothesis": "H1 from upstream design",
                "metrics": [],
                "baselines": [],
            },
            "code_result": {"success": False, "stdout": "", "error": "runtime error"},
            "analysis_result": {},
        }

        score, breakdown = agent._score_report("query", report, outputs)

        self.assertEqual(breakdown["method"], "hybrid_rule_llm")
        self.assertLess(score.overall, 0.75)
        self.assertTrue(score.needs_reflection)
        self.assertEqual(breakdown["scores"]["evidence_grounding"], 0.45)
        self.assertLessEqual(breakdown["scores"]["execution_validity"], 0.35)
        self.assertTrue(any(item["severity"] == "hard" for item in breakdown["findings"]))

    def test_hybrid_scoring_rewards_consistent_upstream_usage(self):
        agent = build_agent()
        agent.judge = FakeJudge([EvalScore(0.82, 0.82, 0.82, 0.82, "solid", False, "")])
        report = {
            "title": "Test report",
            "abstract": "summary",
            "sections": [
                {"heading": "1. Background", "body": "grounded background"},
                {"heading": "2. Literature", "body": "Paper A supports the method."},
                {"heading": "3. Experiment", "body": "experiment overview"},
                {"heading": "3.1 Hypothesis", "body": "H1: student keeps accuracy."},
                {"heading": "3.2 Metrics", "body": "mAP\nFPS"},
                {"heading": "3.3 Baselines", "body": "Vanilla student\nFitNets"},
                {"heading": "4. Method", "body": "method"},
                {"heading": "5. Results", "body": "stdout shows mAP and FPS evidence."},
                {"heading": "6. Conclusion", "body": "conclusion"},
            ],
        }
        outputs = {
            "literature_result": {"total_found": 2, "top_papers": [{"title": "Paper A"}]},
            "experiment_design": {
                "research_hypothesis": "H1: student keeps accuracy.",
                "metrics": ["mAP", "FPS"],
                "baselines": ["Vanilla student", "FitNets"],
            },
            "code_result": {"success": True, "stdout": "mAP=0.41 FPS=32"},
            "analysis_result": {"conclusion": "The run produced measurable results.", "charts": ["chart.png"]},
        }

        score, breakdown = agent._score_report("query", report, outputs)

        self.assertGreater(score.overall, 0.80)
        self.assertFalse(score.needs_reflection)
        self.assertEqual(breakdown["upstream_status"]["literature_total_found"], 2)
        self.assertTrue(breakdown["upstream_status"]["code_success"])


if __name__ == "__main__":
    unittest.main()
