"""
MindPilot 完整演示脚本
=======================
展示系统从问题输入到报告输出的完整流程。
运行: python examples/demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.orchestrator import MindPilotOrchestrator
from agents.evaluation_agent import BenchmarkEvaluator
from tools.llm_client import LLMClient
from framework.logger import MindPilotLogger
from config import CONFIG


# ── 改进⑥：Demo 用进度回调 ──────────────────────────────────
_STEP_LABELS = {
    "planning": "Planning", "literature": "Literature",
    "experiment": "Experiment", "code": "Code",
    "analysis": "Analysis", "evaluation": "Evaluation",
}


def _demo_step_callback(step_name: str, result: dict):
    """Demo 专用的进度回调：简洁打印每步完成状态"""
    label = _STEP_LABELS.get(step_name, step_name)
    step_order = list(_STEP_LABELS.keys())
    idx = step_order.index(step_name) + 1 if step_name in step_order else 0
    total = len(_STEP_LABELS)

    is_degraded = isinstance(result, dict) and (
        result.get("_fallback") or result.get("_timeout") or result.get("_error")
    )
    mark = "[DEGRADED]" if is_degraded else "[OK]"
    print(f"  [{idx}/{total}] {label} {mark}")


def demo_full_pipeline():
    """Demo 1: 完整科研助手流程"""
    print("\n" + "🟦"*30)
    print("Demo 1: 完整科研助手流程")
    print("🟦"*30)

    query = "研究 Transformer 注意力机制的计算复杂度，并用 Python 实现一个简单的缩放点积注意力"

    # 传入进度回调
    orchestrator = MindPilotOrchestrator(on_step_done=_demo_step_callback)
    result = orchestrator.run(query)

    print(f"\n📋 Demo 1 结果摘要：")
    print(f"   会话 ID    : {result['session_id']}")
    print(f"   规划任务数  : {len(result['plan']['tasks'])}")
    lit = result.get('literature', {})
    print(f"   检索论文数  : {lit.get('total_found', 0)}")
    print(f"   知识图谱    : {lit.get('knowledge_graph', {})}")
    code_r = result.get('code', {})
    print(f"   代码生成    : {'✓ 成功' if code_r.get('success') else '✗ 失败'} | 轮次: {code_r.get('total_rounds', 0)}")
    eval_r = result.get('evaluation', {})
    score = eval_r.get('final_score', {}).get('overall', 'N/A')
    print(f"   最终评分    : {score}")
    print(f"   报告文件    : {list(result.get('report_files', {}).values())}")
    return result


def demo_literature_only():
    """Demo 2: 单独演示文献检索模块"""
    print("\n" + "🟩"*30)
    print("Demo 2: 文献检索 Agent 单独演示")
    print("🟩"*30)

    from tools.arxiv_search import ArXivSearchTool
    from agents.literature_agent import LiteratureAgent
    from memory.memory_store import MemoryStore

    logger = MindPilotLogger(session_id="demo2", log_dir="logs")
    llm = LLMClient(CONFIG)
    arxiv = ArXivSearchTool(max_results=5, logger=logger)
    memory = MemoryStore(logger=logger)

    agent = LiteratureAgent(CONFIG, llm, arxiv, memory, logger)
    result = agent.run("attention mechanism in transformer", "attention transformer")

    print(f"\n📚 文献检索结果：")
    print(f"   找到论文数   : {result['total_found']}")
    print(f"   Recall@5    : {result['metrics']['recall@5']}")
    print(f"   知识图谱节点 : {result['knowledge_graph']['nodes']}")
    print(f"   文献综述:\n   {result['literature_review'][:200]}...")
    return result


def demo_code_agent():
    """Demo 3: 单独演示代码生成模块"""
    print("\n" + "🟨"*30)
    print("Demo 3: 代码生成 Agent 单独演示")
    print("🟨"*30)

    from tools.code_executor import CodeExecutor
    from agents.code_agent import CodeAgent
    from memory.memory_store import MemoryStore

    logger = MindPilotLogger(session_id="demo3", log_dir="logs")
    llm = LLMClient(CONFIG)
    executor = CodeExecutor(timeout=30, logger=logger)
    memory = MemoryStore(logger=logger)

    agent = CodeAgent(CONFIG, llm, executor, memory, logger)
    result = agent.run(
        "用 Python 实现线性回归，生成 100 个随机数据点，训练模型并打印 R² 分数和 MSE"
    )

    print(f"\n💻 代码生成结果：")
    print(f"   状态      : {'✓ 成功' if result['success'] else '✗ 失败'}")
    print(f"   Pass@1    : {result['pass_at_1']}")
    print(f"   调试轮次  : {result['total_rounds']}")
    print(f"   执行输出  : {result['stdout'][:150]}")
    return result


def demo_analysis_agent():
    """Demo 4: 单独演示数据分析模块"""
    print("\n" + "🟧"*30)
    print("Demo 4: 数据分析 Agent 单独演示")
    print("🟧"*30)

    import numpy as np
    from agents.analysis_agent import AnalysisAgent
    from tools.visualizer import AutoVisualizer
    from tools.report_generator import ReportGenerator
    from memory.memory_store import MemoryStore

    logger = MindPilotLogger(session_id="demo4", log_dir="logs")
    llm = LLMClient(CONFIG)
    visualizer = AutoVisualizer(output_dir="outputs", logger=logger)
    report_gen = ReportGenerator(output_dir="outputs", logger=logger)
    memory = MemoryStore(logger=logger)

    agent = AnalysisAgent(CONFIG, llm, visualizer, report_gen, memory, logger)

    # 传入真实数据
    np.random.seed(42)
    data = {
        "group_A": np.random.normal(50, 10, 60).tolist(),
        "group_B": np.random.normal(58, 12, 60).tolist(),
    }
    result = agent.run("对比两组数据的显著性差异，并可视化分布", data=data)

    print(f"\n📊 数据分析结果：")
    print(f"   统计检验   : {len(result['statistical_tests'])} 项")
    print(f"   图表文件   : {result['charts']}")
    print(f"   报告格式   : {list(result['report_files'].keys())}")
    print(f"   结论: {result['conclusion'][:150]}...")
    return result


def demo_benchmark():
    """Demo 5: 基准集对比实验"""
    print("\n" + "🟥"*30)
    print("Demo 5: 横向对比基准实验（3题）")
    print("🟥"*30)

    logger = MindPilotLogger(session_id="demo5", log_dir="logs")
    llm = LLMClient(CONFIG)
    evaluator = BenchmarkEvaluator(llm, logger=logger)

    # 简化版 MindPilot runner（不运行完整流程）
    def simple_runner(q):
        return llm.chat([
            {"role": "system", "content": "你是一个科研助手，请详细回答以下问题。"},
            {"role": "user", "content": q}
        ])

    summary = evaluator.run_comparison(system_runner=simple_runner, n_questions=3)

    print(f"\n🏆 基准对比结果：")
    for sys_name, stats in summary.items():
        if isinstance(stats, dict) and "avg" in stats:
            print(f"   {sys_name:15s}: 平均 {stats['avg']:.3f} | 最高 {stats['max']:.3f}")
    print(f"   MindPilot 领先题数: {summary.get('mindpilot_wins', 0)}/{summary.get('total_questions', 0)}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MindPilot Demo")
    parser.add_argument("--demo", type=int, default=0,
                        help="运行指定 Demo（1-5），0 = 运行所有")
    args = parser.parse_args()

    demos = {
        1: demo_full_pipeline,
        2: demo_literature_only,
        3: demo_code_agent,
        4: demo_analysis_agent,
        5: demo_benchmark,
    }

    if args.demo == 0:
        # 默认只运行 Demo 2、3、4（较快），不运行完整流程
        print("\n🚀 MindPilot 功能演示（快速模式：Demo 2~4）")
        print("   运行完整流程: python examples/demo.py --demo 1")
        demo_literature_only()
        demo_code_agent()
        demo_analysis_agent()
    elif args.demo in demos:
        demos[args.demo]()
    else:
        print(f"Demo {args.demo} 不存在，请选择 1-5")
