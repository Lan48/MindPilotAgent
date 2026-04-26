"""
中央编排器（Orchestrator）
===========================
完整流程：
  Step 1 → 任务规划（ToT + ReAct）
  Step 2 → 文献检索（ArXiv + 知识图谱）
  Step 3 → 实验设计（基于文献输出）        ← 新增
  Step 4 → 代码实现（沙箱 + 自动调试）
  Step 5 → 数据分析（统计 + 可视化）
  Step 6 → 评估反思 + 报告生成（Word/MD/HTML）
"""

import time
import uuid
from typing import Optional

from config import CONFIG
from framework.logger import MindPilotLogger
from framework.communication import ErrorCode, MessageBus, HumanInTheLoop
from framework.scheduler import SyncScheduler, Task
from memory.memory_store import MemoryStore
from tools.llm_client import LLMClient
from tools.arxiv_search import ArXivSearchTool
from tools.code_executor import CodeExecutor
from tools.visualizer import AutoVisualizer
from tools.report_generator import ReportGenerator
from agents.planning_agent import PlanningAgent
from agents.literature_agent import LiteratureAgent
from agents.code_agent import CodeAgent
from agents.analysis_agent import AnalysisAgent
from agents.evaluation_agent import EvaluationAgent


class MindPilotOrchestrator:
    def __init__(self, config=None, session_id: Optional[str] = None):
        self.config     = config or CONFIG
        self.session_id = session_id or str(uuid.uuid4())[:12]

        self.logger = MindPilotLogger(
            session_id=self.session_id,
            log_dir=self.config.communication.log_dir,
            verbose=self.config.verbose,
        )
        self.bus        = MessageBus()
        self._register_message_endpoints()
        self.human_loop = HumanInTheLoop(enabled=False)

        self.llm = LLMClient(self.config)
        self.memory = MemoryStore(
            store_dir=self.config.memory_dir,
            session_id=self.session_id,
            logger=self.logger,
        )
        self.arxiv = ArXivSearchTool(
            max_results=self.config.literature.arxiv_max_results,
            logger=self.logger,
        )
        self.executor   = CodeExecutor(timeout=self.config.code.execution_timeout, logger=self.logger)
        self.visualizer = AutoVisualizer(output_dir=self.config.output_dir, logger=self.logger)
        self.report_gen = ReportGenerator(output_dir=self.config.output_dir, logger=self.logger)

        self.planner      = PlanningAgent(self.config, self.llm, self.memory, self.logger)
        self.lit_agent    = LiteratureAgent(self.config, self.llm, self.arxiv, self.memory, self.logger)
        self.code_agent   = CodeAgent(self.config, self.llm, self.executor, self.memory, self.logger)
        self.analysis_agent = AnalysisAgent(
            self.config, self.llm, self.visualizer, self.report_gen, self.memory, self.logger)
        self.eval_agent   = EvaluationAgent(
            self.config, self.llm, self.report_gen, self.memory, self.logger)

        self._print_banner()

    def run(self, query: str) -> dict:
        self.logger.info("Orchestrator", "="*52)
        self.logger.info("Orchestrator", f"收到科研问题: {query[:60]}")
        self.logger.info("Orchestrator", "="*52)
        start_time = time.time()

        # ── Step 1: 任务规划 ──────────────────────────────────
        self.logger.info("Orchestrator", "【Step 1/6】 任务规划（ToT + ReAct）...")
        self._send_request("PlanningAgent", "planning", {"query": query})
        plan = self.planner.run(query)
        self._send_response("PlanningAgent", "planning", {"plan_id": plan.plan_id, "tasks": len(plan.tasks)})
        self.planner.print_plan(plan)

        # ── Step 2: 文献检索 ──────────────────────────────────
        self.logger.info("Orchestrator", "【Step 2/6】 文献检索与知识图谱构建...")
        lit_result = {}
        try:
            lit_task = next(
                (t for t in plan.tasks if t.agent == "LiteratureAgent"), None
            )
            desc = lit_task.description if lit_task else f"检索关于「{query}」的学术文献"
            self._send_request("LiteratureAgent", "literature_search", {"description": desc, "query": query})
            lit_result = self.lit_agent.run(desc, query)
            self._send_response("LiteratureAgent", "literature_search", {
                "total_found": lit_result.get("total_found", 0),
                "top_papers": len(lit_result.get("top_papers", [])),
            })
        except Exception as e:
            self._send_error("LiteratureAgent", "literature_search", e)
            self.logger.error("Orchestrator", f"文献检索失败: {e}")

        # ── Step 3: 实验设计 ──────────────────────────────────
        self.logger.info("Orchestrator", "【Step 3/6】 实验设计方案生成...")
        exp_design = {}
        try:
            self._send_request("EvaluationAgent", "experiment_design", {
                "query": query,
                "papers": len(lit_result.get("top_papers", [])),
                "research_path": plan.selected_path,
            })
            exp_design = self.eval_agent.design_experiment(
                query,
                lit_result,
                research_path=plan.selected_path,
            )
            self._send_response("EvaluationAgent", "experiment_design", {
                "baselines": len(exp_design.get("baselines", [])),
                "metrics": len(exp_design.get("metrics", [])),
                "sections": len(exp_design.get("sections", [])),
                "procedure": len(exp_design.get("procedure", [])),
            })
            self._print_exp_design(exp_design)
        except Exception as e:
            self._send_error("EvaluationAgent", "experiment_design", e)
            self.logger.error("Orchestrator", f"实验设计失败: {e}")

        # ── Step 4: 代码实现 ──────────────────────────────────
        self.logger.info("Orchestrator", "【Step 4/6】 代码生成与自动调试...")
        code_result = {}
        try:
            code_task = next(
                (t for t in plan.tasks if t.agent == "CodeAgent"), None
            )
            code_desc = code_task.description if code_task else f"为「{query}」实现核心算法"

            # 把文献方法和实验设计作为代码生成的上下文
            context = {
                "top_papers":   lit_result.get("top_papers", [])[:3],
                "exp_design":   exp_design,
                "baselines":    exp_design.get("baselines", []),
                "metrics":      exp_design.get("metrics", []),
            }
            self._send_request("CodeAgent", "code_generation", {
                "description": code_desc,
                "has_experiment_design": bool(exp_design),
            })
            code_result = self.code_agent.run(code_desc, context=context)
            self._send_response("CodeAgent", "code_generation", {
                "success": code_result.get("success", False),
                "rounds": code_result.get("total_rounds", 0),
            })
        except Exception as e:
            self._send_error("CodeAgent", "code_generation", e)
            self.logger.error("Orchestrator", f"代码生成失败: {e}")

        # ── Step 5: 数据分析 ──────────────────────────────────
        self.logger.info("Orchestrator", "【Step 5/6】 数据分析与可视化...")
        analysis_result = {}
        try:
            ana_task = next(
                (t for t in plan.tasks if t.agent == "AnalysisAgent"), None
            )
            ana_desc = ana_task.description if ana_task else f"分析「{query}」的实验结果"
            code_stdout = code_result.get("stdout", "")
            self._send_request("AnalysisAgent", "data_analysis", {
                "description": ana_desc,
                "has_code_output": bool(code_stdout),
            })
            analysis_result = self.analysis_agent.run(
                ana_desc, code_output=code_stdout
            )
            self._send_response("AnalysisAgent", "data_analysis", {
                "charts": len(analysis_result.get("charts", [])),
                "has_conclusion": bool(analysis_result.get("conclusion", "")),
            })
        except Exception as e:
            self._send_error("AnalysisAgent", "data_analysis", e)
            self.logger.error("Orchestrator", f"数据分析失败: {e}")

        # ── Step 6: 评估反思 + 报告生成 ──────────────────────
        self.logger.info("Orchestrator", "【Step 6/6】 评估反思与报告生成（Word/MD/HTML）...")
        aggregated = {
            "literature_result": lit_result,
            "experiment_design": exp_design,
            "code_result":       code_result,
            "analysis_result":   analysis_result,
            "literature_review": lit_result.get("literature_review", ""),
            "papers":            lit_result.get("top_papers", []),
            "code":              code_result.get("final_code", ""),
            "stdout":            code_result.get("stdout", ""),
            "analysis":          analysis_result.get("conclusion", ""),
            "charts":            analysis_result.get("charts", []),
        }
        self._send_request("EvaluationAgent", "evaluation", {
            "sections": ["literature", "experiment", "code", "analysis"],
        })
        eval_result = self.eval_agent.run(query, aggregated)
        self._send_response("EvaluationAgent", "evaluation", {
            "score": eval_result.get("final_score", {}).get("overall"),
            "reports": list(eval_result.get("report_files", {}).keys()),
        })

        # ── 收尾 ──────────────────────────────────────────────
        self.memory.save_long_term()
        bus_stats = self.bus.get_stats()
        session_summary = self.logger.save_summary(extra={"message_bus": bus_stats})
        self.logger.print_call_chain()

        total_time  = round(time.time() - start_time, 2)
        final_result = {
            "query":        query,
            "session_id":   self.session_id,
            "plan":         plan.to_dict(),
            "literature":   lit_result,
            "experiment":   exp_design,
            "code":         code_result,
            "analysis":     analysis_result,
            "evaluation":   eval_result,
            "report_files": eval_result.get("report_files", {}),
            "total_time_s": total_time,
            "session_log":  str(self.logger.log_file),
            "message_bus":  bus_stats,
        }
        self._print_final_summary(final_result, total_time)
        return final_result

    def _register_message_endpoints(self):
        for agent in [
            "Orchestrator",
            "PlanningAgent",
            "LiteratureAgent",
            "CodeAgent",
            "AnalysisAgent",
            "EvaluationAgent",
        ]:
            self.bus.register(agent)

    def _send_request(self, receiver: str, task_id: str, payload: dict):
        msg = self.bus.request(
            sender="Orchestrator",
            receiver=receiver,
            task_id=task_id,
            payload=payload,
            session_id=self.session_id,
        )
        self.logger.log_message(msg)
        return msg

    def _send_response(self, sender: str, task_id: str, payload: dict):
        msg = self.bus.response(
            sender=sender,
            receiver="Orchestrator",
            task_id=task_id,
            payload=payload,
            session_id=self.session_id,
        )
        self.logger.log_message(msg)
        return msg

    def _send_error(self, sender: str, task_id: str, error: Exception):
        msg = self.bus.error(
            sender=sender,
            receiver="Orchestrator",
            task_id=task_id,
            detail=str(error),
            code=ErrorCode.EXECUTION_ERROR,
            session_id=self.session_id,
        )
        self.logger.log_message(msg)
        return msg

    def _print_exp_design(self, exp: dict):
        print(f"\n{'━'*60}")
        print(f"  🔬 实验设计方案")
        print(f"{'━'*60}")
        hyp = exp.get("research_hypothesis", "")
        if hyp:
            print(f"  假设: {hyp[:70]}")
        baselines = exp.get("baselines", [])
        if baselines:
            print(f"  对照: {' | '.join(str(b)[:25] for b in baselines[:3])}")
        metrics = exp.get("metrics", [])
        if metrics:
            print(f"  指标: {' | '.join(str(m)[:20] for m in metrics[:3])}")
        desc = exp.get("full_description", "")
        if desc:
            print(f"  描述: {desc[:80]}...")
        print(f"{'━'*60}\n")

    def _print_banner(self):
        mode = "Mock 模式" if self.config.mock_mode else f"API [{self.config.llm.model}]"
        print(f"""
╔══════════════════════════════════════════════════════╗
║         MindPilot  多模态智能科研助手 Agent           ║
║         六步全流程：规划→文献→实验→代码→分析→报告    ║
╠══════════════════════════════════════════════════════╣
║  会话 ID : {self.session_id:<42s}║
║  LLM 模式: {mode:<42s}║
╚══════════════════════════════════════════════════════╝
""")

    def _print_final_summary(self, result: dict, total_time: float):
        score   = result["evaluation"].get("final_score", {}).get("overall", "N/A")
        reports = result.get("report_files", {})
        print(f"""
╔══════════════════════════════════════════════════════╗
║                  ✅ 全流程完成！                      ║
╠══════════════════════════════════════════════════════╣
║  总耗时   : {total_time}s""")
        print(f"║  质量评分 : {score}")
        print(f"║  报告文件 :")
        for fmt, path in reports.items():
            print(f"║    [{fmt.upper():8s}] {path}")
        print(f"║  会话日志 : {result['session_log']}")
        print(f"╚══════════════════════════════════════════════════════╝")
