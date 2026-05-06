"""
MindPilot — 主入口（交互模式）
================================
运行: python main.py
"""

import sys
import os


# 确保项目根目录的父目录在 Python 路径中，以便能导入
# 顶级包 `MindPilotAgent`（当直接运行脚本时当前路径通常是
# 项目目录，本文件位于 `MindPilotAgent/mindpilot/main.py`，
# 因此需要将项目目录的父目录加入 sys.path）。
project_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_parent)
#sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MindPilotAgent.mindpilot.orchestrator.orchestrator import MindPilotOrchestrator


EXAMPLE_QUERIES = [
    "研究 Transformer 注意力机制的计算复杂度优化方法，并实现一个简单的线性注意力",
    "对比 SGD 和 Adam 优化器的收敛速度，用可视化图表展示",
    "调研联邦学习的隐私保护方法，实现一个 FedAvg 的简化版本",
    "研究图神经网络在推荐系统中的应用，实现一个简单的 GCN 节点分类",
]


def interactive_mode():
    """交互模式：用户输入科研问题"""
    print("\n" + "="*58)
    print("  MindPilot — 多模态智能科研助手 Agent 系统")
    print("  输入 'exit' 退出 | 输入 'example' 查看示例问题")
    print("="*58 + "\n")

    orchestrator = MindPilotOrchestrator()

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
    """单次运行模式"""
    orchestrator = MindPilotOrchestrator()
    return orchestrator.run(query)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 命令行参数模式：python main.py "your question here"
        query = " ".join(sys.argv[1:])
        single_run(query)
    else:
        interactive_mode()
