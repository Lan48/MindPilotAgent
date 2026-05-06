"""
MindPilot — 主入口（交互模式）
================================
运行: python main.py
"""

import sys
import os
import time

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator.orchestrator import MindPilotOrchestrator


EXAMPLE_QUERIES = [
    "研究 Transformer 注意力机制的计算复杂度优化方法，并实现一个简单的线性注意力",
    "对比 SGD 和 Adam 优化器的收敛速度，用可视化图表展示",
    "调研联邦学习的隐私保护方法，实现一个 FedAvg 的简化版本",
    "研究图神经网络在推荐系统中的应用，实现一个简单的 GCN 节点分类",
]

# ── 改进⑥：进度回调函数 ─────────────────────────────────────
# 步骤名称 → 显示名映射（使用英文避免 Windows GBK 终端乱码）
_STEP_LABELS = {
    "planning":    "Planning",
    "literature":  "Literature",
    "experiment":  "Experiment",
    "code":        "Code",
    "analysis":    "Analysis",
    "evaluation":  "Evaluation",
}

# 记录各步骤完成时间，用于计算耗时
_step_start_time = None


def on_step_done(step_name: str, result: dict):
    """
    每个 Step 完成后由编排器自动调用的回调函数。
    功能：
      1. 在控制台打印实时进度条
      2. 显示每个步骤的耗时和关键指标
      3. 标记降级/超时/失败的步骤
    """
    global _step_start_time

    label = _STEP_LABELS.get(step_name, step_name)
    # 步骤序号（1-based）
    step_order = list(_STEP_LABELS.keys())
    step_idx = step_order.index(step_name) + 1 if step_name in step_order else 0
    total_steps = len(_STEP_LABELS)

    # 进度条（使用 ASCII 兼容字符，避免 Windows GBK 编码问题）
    filled = step_idx
    bar = "#" * filled + "-" * (total_steps - filled)
    pct = int(step_idx / total_steps * 100)

    # 状态标记（使用 ASCII 兼容字符，避免 Windows GBK 编码问题）
    is_fallback = isinstance(result, dict) and (
        result.get("_fallback") or result.get("_timeout") or result.get("_error")
    )
    if isinstance(result, dict) and result.get("_timeout"):
        status = "[TIMEOUT]"
    elif isinstance(result, dict) and result.get("_error"):
        status = "[FAILED]"
    elif is_fallback:
        status = "[DEGRADED]"
    else:
        status = "[OK]"

    # 提取关键指标摘要
    summary = _extract_step_summary(step_name, result)

    print(f"\n  [{bar}] {pct}%  Step {step_idx}/{total_steps}: {label} {status}")
    if summary:
        print(f"  {summary}")


def _extract_step_summary(step_name: str, result) -> str:
    """Extract a one-line key metric summary from each step's result"""
    if not isinstance(result, dict):
        return ""

    if step_name == "planning":
        tasks = result.get("tasks", [])
        return f"  -> {len(tasks)} subtasks" if tasks else ""

    if step_name == "literature":
        total = result.get("total_found", 0)
        r5 = result.get("metrics", {}).get("recall@5", "N/A") if isinstance(result.get("metrics"), dict) else "N/A"
        return f"  -> {total} papers | Recall@5: {r5}"

    if step_name == "experiment":
        hyp = result.get("research_hypothesis", "")
        n_bl = len(result.get("baselines", []))
        return f"  -> hypothesis: {hyp[:40]}... | {n_bl} baselines" if hyp else ""

    if step_name == "code":
        ok = result.get("success", False)
        rounds = result.get("total_rounds", 0)
        return f"  -> {'success' if ok else 'failed'} | {rounds} debug rounds"

    if step_name == "analysis":
        n_charts = len(result.get("charts", []))
        return f"  -> {n_charts} charts generated"

    if step_name == "evaluation":
        score = result.get("final_score", {}).get("overall", "N/A") if isinstance(result.get("final_score"), dict) else "N/A"
        n_files = len(result.get("report_files", {}))
        return f"  -> score: {score} | {n_files} reports"

    return ""


def interactive_mode():
    """交互模式：用户输入科研问题"""
    print("\n" + "="*58)
    print("  MindPilot — 多模态智能科研助手 Agent 系统")
    print("  输入 'exit' 退出 | 输入 'example' 查看示例问题")
    print("="*58 + "\n")

    # 传入 on_step_done 回调，启用实时进度显示
    orchestrator = MindPilotOrchestrator(on_step_done=on_step_done)

    while True:
        try:
            query = input("🔬 请输入科研问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 再见！")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q", "退出"):
            print("👋 再见！")
            break
        if query.lower() == "example":
            print("\n示例问题：")
            for i, q in enumerate(EXAMPLE_QUERIES, 1):
                print(f"  {i}. {q}")
            print()
            continue

        try:
            result = orchestrator.run(query)
            print(f"\n✅ 完成！报告文件：")
            for fmt, path in result.get("report_files", {}).items():
                print(f"   [{fmt.upper()}] {path}")
        except Exception as e:
            print(f"\n❌ 执行出错：{e}")
            import traceback
            traceback.print_exc()

        print()


def single_run(query: str):
    """单次运行模式（也启用进度回调）"""
    orchestrator = MindPilotOrchestrator(on_step_done=on_step_done)
    return orchestrator.run(query)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 命令行参数模式：python main.py "your question here"
        query = " ".join(sys.argv[1:])
        single_run(query)
    else:
        interactive_mode()
