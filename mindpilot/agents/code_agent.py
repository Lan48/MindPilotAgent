"""
模块③ — 代码生成与执行 Agent
================================
「需求 → 代码 → 执行 → 错误 → 修复」闭环自动调试。
AST 静态安全检测 + 受限沙箱执行。
"""

import ast
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CodeSession:
    """一次代码生成会话的完整记录"""
    task_id: str
    requirement: str
    iterations: list[dict] = field(default_factory=list)   # 每轮代码 + 执行结果
    final_code: str = ""
    final_result: Optional[dict] = None
    success: bool = False
    total_rounds: int = 0
    pass_at_1: bool = False      # 第一轮是否成功


class CodeAgent:
    """
    模块③ — 代码生成与执行 Agent
    """

    AGENT_NAME = "CodeAgent"

    def __init__(self, config, llm_client, code_executor, memory_store, logger):
        self.config = config
        self.llm = llm_client
        self.executor = code_executor
        self.memory = memory_store
        self.logger = logger
        self.max_rounds = config.code.max_debug_rounds
        # debug
        self.debug_mode = os.getenv("MP_DEBUG", "0") == "1"

    def run(self, task_description: str, context: dict = None) -> dict:
        """
        主入口：生成并执行代码，自动调试
        """
        call = self.logger.start_call(self.AGENT_NAME, "code_generation", task_description)
        session = CodeSession(task_id=call.call_id, requirement=task_description)
        context = context or {}

        try:
            # Step 1: 初始代码生成（只基于任务描述）
            self.logger.info(self.AGENT_NAME, "生成初始代码...")
            code = self._generate_code(task_description, context)

            quick_issues = self._quick_validate_code(code)
            if quick_issues:
                self.logger.warning(
                    self.AGENT_NAME,
                    f"初始代码快速校验失败: {quick_issues[0]}"
                )
                code = self._regenerate_clean_code(task_description, code)


            # Step 2: 执行 + 自动调试循环
            for round_num in range(1, self.max_rounds + 1):
                self.logger.info(self.AGENT_NAME, f"第 {round_num}/{self.max_rounds} 轮执行...")

                # ── AST 安全检测 ────────────────────────────────
                issues = self.executor.checker.check(code)
                if issues:
                    # 区分"语法错误"（代码提取失败）和"真正的安全问题"
                    syntax_issues   = [i for i in issues if "语法错误" in i or "SyntaxError" in i]
                    security_issues = [i for i in issues if "语法错误" not in i and "SyntaxError" not in i]

                    if syntax_issues:
                        # 语法错误：说明 extract_code 没能剥离 markdown 标记，
                        # 或 LLM 返回了非 Python 内容。
                        # 策略：再次强制提取 + 让 LLM 重新生成纯代码。
                        self.logger.warning(self.AGENT_NAME,
                            f"代码提取异常（语法错误），尝试重新提取并要求 LLM 输出纯代码...")
                        # 先尝试二次提取
                        code = self.executor.extract_code(code)
                        # 二次提取后再检测
                        issues2 = self.executor.checker.check(code)
                        if any("语法错误" in i for i in issues2):
                            # 仍有语法错误，要求 LLM 重新输出
                            code = self._regenerate_clean_code(task_description, code)

                    if security_issues:
                        # 真正的安全问题（危险函数调用等）
                        self.logger.warning(self.AGENT_NAME,
                            f"安全检测：{len(security_issues)} 个安全问题 → {security_issues[0]}")
                        code = self._fix_safety_issues(code, security_issues)

                # 执行代码（子进程模式，支持完整 import）
                exec_result = self.executor.execute_with_subprocess(code)
                iteration = {
                    "round": round_num,
                    "code": code,
                    "stdout": exec_result.stdout[:500],
                    "stderr": exec_result.stderr[:500],
                    "success": exec_result.success,
                    "duration": exec_result.execution_time,
                    "safety_issues": exec_result.safety_issues,
                }
                session.iterations.append(iteration)

                if exec_result.success:
                    session.success = True
                    session.pass_at_1 = (round_num == 1)
                    session.final_code = code
                    session.final_result = exec_result.to_dict()
                    self.logger.success(self.AGENT_NAME,
                        f"✓ 代码执行成功（第{round_num}轮，Pass@1={session.pass_at_1}）")
                    break
                else:
                    # 未到最后一轮才修复
                    if round_num < self.max_rounds:
                        self.logger.info(self.AGENT_NAME,
                            f"执行失败，自动调试... 错误: {exec_result.error_type}")
                        code = self._debug_code(
                            code, exec_result.stderr, exec_result.error_type, task_description
                        )
                    else:
                        self.logger.warning(self.AGENT_NAME,
                            f"达到最大调试轮数 ({self.max_rounds})，任务未完成")
                        session.final_code = code
                        session.final_result = exec_result.to_dict()

            session.total_rounds = len(session.iterations)

            # 单元测试自动生成
            test_code = self._generate_tests(session.final_code, task_description)

            # 存入记忆
            self.memory.add(
                content=f"代码任务: {task_description[:100]}",
                agent=self.AGENT_NAME,
                payload={"code": session.final_code[:500], "success": session.success},
                tags=["code"],
                importance=1.2 if session.success else 0.8,
            )

            result = {
                "success": session.success,
                "final_code": session.final_code,
                "stdout": session.final_result.get("stdout", "") if session.final_result else "",
                "total_rounds": session.total_rounds,
                "pass_at_1": session.pass_at_1,
                "iterations": session.iterations,
                "test_code": test_code,
            }
            self.logger.finish_call(call, result)
            self._print_session(session)
            return result

        except Exception as e:
            self.logger.fail_call(call, str(e))
            raise

    def _generate_code(self, requirement: str, context: dict = None) -> str:
        """初始代码生成（基于任务描述 + 简要上下文）"""
        context = context or {}

        context_parts = []

        # 1. 文献方法摘要，只取少量，避免 prompt 太长
        top_papers = context.get("top_papers", [])
        if top_papers:
            methods = []
            for p in top_papers[:2]:
                summary = p.get("structured_summary", {}) if isinstance(p, dict) else {}
                method = summary.get("method", "")
                if method:
                    methods.append(method[:300])
            if methods:
                context_parts.append("参考文献方法摘要：\n" + "\n".join(f"- {m}" for m in methods))

        # 2. 实验设计摘要
        exp_design = context.get("exp_design", {})
        if isinstance(exp_design, dict) and exp_design:
            hypothesis = exp_design.get("research_hypothesis", "")
            full_desc = exp_design.get("full_description", "")
            if hypothesis:
                context_parts.append(f"实验假设：{hypothesis[:300]}")
            if full_desc:
                context_parts.append(f"实验设计说明：{full_desc[:500]}")

        # 3. baseline 和 metrics
        baselines = context.get("baselines", [])
        if baselines:
            context_parts.append(
                "实验对照/基线：\n" + "\n".join(f"- {str(b)[:150]}" for b in baselines[:3])
            )

        metrics = context.get("metrics", [])
        if metrics:
            context_parts.append(
                "评价指标：\n" + "\n".join(f"- {str(m)[:120]}" for m in metrics[:5])
            )

        context_text = "\n\n".join(context_parts)

        system = (
            "你是资深 Python 科研工程师。请根据任务描述和给定上下文生成高质量、可直接运行的 Python 代码。\n"
            "要求：\n"
            "1. 任务描述是主要依据，上下文只作为参考，不要输出文献综述或解释文字。\n"
            "2. 添加中文注释。\n"
            "3. 包含完整的错误处理。\n"
            "4. 代码必须是轻量级可运行示例，不要下载外部模型、数据集或权重文件。\n"
            "5. 不要依赖本地不存在的数据路径。\n"
            "6. 如果任务涉及 YOLO、Transformer、CLIP 等大型模型，请用合成数据或简化模型模拟核心优化思想。\n"
            "7. 最后必须 print 可供后续分析的关键数值结果，例如 loss、accuracy、latency、model_size、speedup 等。\n"
            "8. 建议按如下格式打印：\n"
            "print('ANALYSIS_RESULT_START')\n"
            "print('accuracy=0.91')\n"
            "print('loss=0.12')\n"
            "print('latency_ms=12.5')\n"
            "print('ANALYSIS_RESULT_END')\n"
            "9. 只输出纯 Python 代码，不要 markdown，不要解释。"
        )

        if context_text:
            prompt = f"任务描述：{requirement}\n\n参考上下文：\n{context_text}"
        else:
            prompt = f"任务描述：{requirement}"

        resp = self.llm.chat_code([
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ])

        extracted = self.executor.extract_code(resp)

        self._save_debug_file("generate_raw.txt", str(resp))
        self._save_debug_file("generate_extracted.py", extracted)

        return extracted

    def _debug_code(self, code: str, error: str, error_type: str, requirement: str) -> str:
        """根据错误信息自动修复代码"""
        system = (
            "你是 Python 调试专家。根据错误信息修复以下代码。"
            "只输出修复后的纯 Python 完整代码，不要 markdown，不要解释。"
        )
        prompt = (
            f"原始需求：{requirement}\n\n"
            f"错误类型：{error_type}\n"
            f"错误信息：{error[:600]}\n\n"
            f"当前代码：\n```python\n{code}\n```"
        )
        resp = self.llm.chat_code([
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ])

        extracted = self.executor.extract_code(resp)

        self._save_debug_file("debug_raw.txt", str(resp))
        self._save_debug_file("debug_extracted.py", extracted)

        return extracted

    def _fix_safety_issues(self, code: str, issues: list[str]) -> str:
        """修复安全问题"""
        system = (
            "你是代码安全专家。以下代码存在安全问题，请移除危险操作，"
            "替换为安全的等价实现。可以保留正常科研计算库（包括 PyTorch、NumPy、Pandas 等），"
            "只输出修复后的纯 Python 完整代码，不要 markdown，不要解释。"
        )
        prompt = f"安全问题：{'; '.join(issues)}\n\n代码：\n```python\n{code}\n```"
        resp = self.llm.chat_code([
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ])
        return self.executor.extract_code(resp)

    def _regenerate_clean_code(self, requirement: str, bad_code: str) -> str:
        """
        当 LLM 返回了带 markdown 标记或非 Python 内容时，
        明确要求 LLM 重新输出纯 Python 代码（无任何 markdown 格式）。
        """
        system = (
            "你是 Python 专家。请直接输出可运行的 Python 代码，"
            "【绝对不要】包含任何 markdown 标记（不要有 ```python 或 ``` ），"
            "不要有任何解释文字，只输出纯 Python 代码本身。"
            "第一行必须是 import 语句或注释，不能是 ``` 或其他标记。"
        )
        prompt = (
            f"需求：{requirement}\n\n"
            f"之前的输出包含了格式错误，请重新输出纯 Python 代码（不含任何 markdown）：\n"
            f"前次输出片段：{bad_code[:200]}"
        )
        resp = self.llm.chat_code([
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ])
        # 再次提取，双重保险
        return self.executor.extract_code(resp)

    def _generate_tests(self, code: str, requirement: str) -> str:
        """自动生成单元测试"""
        if not code or not code.strip():
            return ""
        system = (
            "为以下 Python 代码生成简单的单元测试（使用 unittest）。"
            "只测试核心逻辑，可使用代码本身依赖的常见科学计算/深度学习库（如 NumPy、Pandas、PyTorch）。"
            "只输出纯 Python 代码，不要 markdown，不要解释。"
        )
        resp = self.llm.chat_code([
            {"role": "system", "content": system},
            {"role": "user", "content": f"需求：{requirement}\n\n代码：\n```python\n{code[:800]}\n```"}
        ])
        return self.executor.extract_code(resp)

    def _print_session(self, session: CodeSession):
        print(f"\n{'━'*58}")
        print(f"  💻 代码生成会话 [{session.task_id}]")
        print(f"{'━'*58}")
        print(f"  需求: {session.requirement[:55]}")
        print(f"  状态: {'✓ 成功' if session.success else '✗ 失败'} | "
            f"轮次: {session.total_rounds} | Pass@1: {session.pass_at_1}")
        for it in session.iterations:
            icon = "✓" if it["success"] else "✗"
            print(f"  轮{it['round']}: {icon} [{it['duration']}s] "
                f"{'OK' if it['success'] else it['stderr'][:40]}")
        print(f"{'━'*58}\n")

    def _save_debug_file(self, filename: str, content: str):
        if not self.debug_mode:
            return

        debug_dir = "debug_outputs"
        os.makedirs(debug_dir, exist_ok=True)

        path = os.path.join(debug_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content if content is not None else "")

    def _quick_validate_code(self, code: str) -> list[str]:
        """对生成代码做轻量校验，返回问题列表"""
        issues = []

        if not code or not code.strip():
            issues.append("生成代码为空")
            return issues

        # 长度过短，通常说明生成失败或提取失败
        if len(code.strip()) < 50:
            issues.append("生成代码过短，疑似不完整")

        # 先检查语法
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(f"语法错误: {e}")
            return issues

        return issues