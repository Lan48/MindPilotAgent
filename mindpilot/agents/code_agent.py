"""
模块③ — 代码生成与执行 Agent
================================
「需求 → 代码 → 执行 → 错误 → 修复」闭环自动调试。
AST 静态安全检测 + 受限沙箱执行。
"""

import json
import re
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

    def run(self, task_description: str, context: dict = None) -> dict:
        """
        主入口：生成并执行代码，自动调试
        """
        call = self.logger.start_call(self.AGENT_NAME, "code_generation", task_description)
        session = CodeSession(task_id=call.call_id, requirement=task_description)
        context = context or {}

        try:
            # Step 1: 从记忆检索相似代码
            similar = self.memory.search(task_description, top_k=3, agent_filter=self.AGENT_NAME)
            context_hint = ""
            if similar:
                self.logger.info(self.AGENT_NAME, f"记忆检索：{len(similar)} 条相似代码记录")
                context_hint = f"\n参考历史：{similar[0].content[:200]}"

            # Step 2: 初始代码生成
            self.logger.info(self.AGENT_NAME, "生成初始代码...")
            code = self._generate_code(task_description, context_hint, context)

            # Step 3: 执行 + 自动调试循环
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

    def _generate_code(self, requirement: str, context_hint: str, extra_ctx: dict) -> str:
        """初始代码生成"""
        papers_hint = ""
        if extra_ctx.get("top_papers"):
            methods = [
                p.get("structured_summary", {}).get("method", "")
                for p in extra_ctx["top_papers"][:2]
                if p.get("structured_summary")
            ]
            if methods:
                papers_hint = f"\n参考文献方法：{'; '.join(m for m in methods if m)}"

        system = (
            "你是资深 Python 科研工程师。请根据需求生成高质量、可直接运行的 Python 代码。\n"
            "要求：\n"
            "1. 只使用标准库、numpy、pandas、matplotlib、sklearn、scipy\n"
            "2. 添加中文注释\n"
            "3. 包含完整的错误处理\n"
            "4. 最后 print 关键结果\n"
            "5. 使用 matplotlib.use('Agg') 避免 GUI 报错\n"
            "只输出代码块，不要解释。"
        )
        prompt = f"需求：{requirement}{papers_hint}{context_hint}"
        resp = self.llm.chat_code([
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ])
        return self.executor.extract_code(resp)

    def _debug_code(self, code: str, error: str, error_type: str, requirement: str) -> str:
        """根据错误信息自动修复代码"""
        system = (
            "你是 Python 调试专家。根据错误信息修复以下代码。"
            "只输出修复后的完整代码块，不要解释。"
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
        return self.executor.extract_code(resp)

    def _fix_safety_issues(self, code: str, issues: list[str]) -> str:
        """修复安全问题"""
        system = (
            "你是代码安全专家。以下代码存在安全问题，请移除危险操作，"
            "替换为安全的等价实现。只输出修复后的代码块。"
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
            "只测试核心逻辑，假设环境中有 numpy、pandas。只输出代码块。"
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
