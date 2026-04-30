"""
模块⑤ — Agent 间消息通信协议
==============================
定义统一的消息格式、错误码体系与超时重试机制。
"""

import uuid
import time
import asyncio
import inspect
from enum import Enum
from typing import Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime


# ── 错误码体系 ──────────────────────────────────────────────
class ErrorCode(str, Enum):
    OK = "OK"
    TIMEOUT = "TIMEOUT"
    LLM_ERROR = "LLM_ERROR"
    TOOL_ERROR = "TOOL_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    REFLECTION_NEEDED = "REFLECTION_NEEDED"
    MAX_RETRIES = "MAX_RETRIES"
    AGENT_UNAVAILABLE = "AGENT_UNAVAILABLE"


# ── 消息类型 ─────────────────────────────────────────────────
class MessageType(str, Enum):
    REQUEST = "REQUEST"          # 任务请求
    RESPONSE = "RESPONSE"        # 正常响应
    ERROR = "ERROR"              # 错误响应
    HEARTBEAT = "HEARTBEAT"      # 心跳检测
    REFLECTION = "REFLECTION"    # 反思请求
    HUMAN_PAUSE = "HUMAN_PAUSE"  # 人机协同暂停


# ── 消息数据结构（JSON Schema 见 docs/message_schema.json）──
@dataclass
class Message:
    """
    统一消息格式规范
    所有 Agent 间通信必须使用此结构
    """
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    msg_type: MessageType = MessageType.REQUEST
    sender: str = ""
    receiver: str = ""
    task_id: str = ""
    session_id: str = ""
    payload: Any = None          # 实际内容
    metadata: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error_code: ErrorCode = ErrorCode.OK
    error_detail: Optional[str] = None
    retry_count: int = 0
    requires_human_approval: bool = False

    def to_dict(self) -> dict:
        return {
            "msg_id": self.msg_id,
            "msg_type": self.msg_type.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "payload": self.payload,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "error_code": self.error_code.value,
            "error_detail": self.error_detail,
            "retry_count": self.retry_count,
            "requires_human_approval": self.requires_human_approval,
        }

    @classmethod
    def ok(cls, sender: str, receiver: str, task_id: str, payload: Any, **kwargs) -> "Message":
        return cls(
            msg_type=MessageType.RESPONSE,
            sender=sender, receiver=receiver,
            task_id=task_id, payload=payload,
            error_code=ErrorCode.OK, **kwargs
        )

    @classmethod
    def error(cls, sender: str, receiver: str, task_id: str,
              code: ErrorCode, detail: str, **kwargs) -> "Message":
        return cls(
            msg_type=MessageType.ERROR,
            sender=sender, receiver=receiver,
            task_id=task_id,
            error_code=code, error_detail=detail, **kwargs
        )


# ── 消息总线（内存版，适合单进程多协程） ───────────────────────
class MessageBus:
    """
    轻量级内存消息总线
    生产环境可替换为 Redis Pub/Sub 或 RabbitMQ
    """

    def __init__(self):
        self._queues: dict[str, asyncio.Queue] = {}
        self._handlers: dict[str, list[Callable]] = {}
        self._message_log: list[Message] = []

    def register(self, agent_name: str):
        """注册一个 Agent 的接收队列"""
        if agent_name not in self._queues:
            self._queues[agent_name] = asyncio.Queue()

    def create_message(
        self,
        sender: str,
        receiver: str,
        task_id: str,
        payload: Any = None,
        msg_type: MessageType = MessageType.REQUEST,
        session_id: str = "",
        metadata: Optional[dict] = None,
    ) -> Message:
        """创建符合统一协议的消息对象，便于同步/异步流程复用。"""
        return Message(
            msg_type=msg_type,
            sender=sender,
            receiver=receiver,
            task_id=task_id,
            session_id=session_id,
            payload=payload,
            metadata=metadata or {},
        )

    async def send(self, msg: Message):
        """发送消息到目标 Agent 的队列"""
        self._message_log.append(msg)
        if msg.receiver in self._queues:
            await self._queues[msg.receiver].put(msg)
        # 触发注册的处理器
        for handler in self._handlers.get(msg.receiver, []):
            await handler(msg)

    def send_sync(self, msg: Message):
        """
        同步流程下的消息发送。
        当前 Orchestrator 是串行同步执行，使用该方法可以保留统一消息协议、
        队列追踪与日志回放，不强制把业务改造成 async。
        """
        self._message_log.append(msg)
        if msg.receiver in self._queues:
            self._queues[msg.receiver].put_nowait(msg)

        for handler in self._handlers.get(msg.receiver, []):
            result = handler(msg)
            if inspect.isawaitable(result):
                raise RuntimeError("send_sync 不支持异步 handler，请改用 await send(msg)")

    def request(
        self,
        sender: str,
        receiver: str,
        task_id: str,
        payload: Any = None,
        session_id: str = "",
        metadata: Optional[dict] = None,
    ) -> Message:
        """记录一次任务请求消息。"""
        msg = self.create_message(
            sender=sender,
            receiver=receiver,
            task_id=task_id,
            payload=payload,
            msg_type=MessageType.REQUEST,
            session_id=session_id,
            metadata=metadata,
        )
        self.send_sync(msg)
        return msg

    def response(
        self,
        sender: str,
        receiver: str,
        task_id: str,
        payload: Any = None,
        session_id: str = "",
        metadata: Optional[dict] = None,
    ) -> Message:
        """记录一次正常响应消息。"""
        msg = self.create_message(
            sender=sender,
            receiver=receiver,
            task_id=task_id,
            payload=payload,
            msg_type=MessageType.RESPONSE,
            session_id=session_id,
            metadata=metadata,
        )
        self.send_sync(msg)
        return msg

    def error(
        self,
        sender: str,
        receiver: str,
        task_id: str,
        detail: str,
        code: ErrorCode = ErrorCode.EXECUTION_ERROR,
        session_id: str = "",
        metadata: Optional[dict] = None,
    ) -> Message:
        """记录一次错误响应消息。"""
        msg = Message.error(
            sender=sender,
            receiver=receiver,
            task_id=task_id,
            code=code,
            detail=detail,
            session_id=session_id,
            metadata=metadata or {},
        )
        self.send_sync(msg)
        return msg

    async def receive(self, agent_name: str, timeout: float = 30.0) -> Optional[Message]:
        """从队列接收消息，支持超时"""
        if agent_name not in self._queues:
            return None
        try:
            return await asyncio.wait_for(
                self._queues[agent_name].get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None

    def subscribe(self, agent_name: str, handler: Callable):
        """订阅消息处理器"""
        self._handlers.setdefault(agent_name, []).append(handler)

    def get_stats(self) -> dict:
        return {
            "total_messages": len(self._message_log),
            "registered_agents": list(self._queues.keys()),
            "by_type": {
                t.value: sum(1 for m in self._message_log if m.msg_type == t)
                for t in MessageType
            },
            "by_agent": {
                agent: sum(
                    1 for m in self._message_log
                    if m.sender == agent or m.receiver == agent
                )
                for agent in self._queues
            },
        }

    def get_message_log(self) -> list[dict]:
        """返回可序列化的消息日志，供报告或调试使用。"""
        return [msg.to_dict() for msg in self._message_log]


# ── 重试装饰器 ────────────────────────────────────────────────
def with_retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    同步重试装饰器，用于封装可能失败的 Agent 操作
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_error = None
            wait = delay
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        time.sleep(wait)
                        wait *= backoff
            raise RuntimeError(
                f"Max retries ({max_attempts}) exceeded: {last_error}"
            ) from last_error
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator


# ── 人机协同门控 ──────────────────────────────────────────────
class HumanInTheLoop:
    """
    Human-in-the-Loop 机制
    在关键决策节点暂停流程，展示中间结果，等待用户确认、选择或修改。

    审核点：
      1. 规划完成后：展示 ToT 路径和任务列表，允许用户选择路径或跳过任务
      2. 实验设计完成后：展示假设/基线/指标，允许用户修改
      3. 代码执行完成后：展示代码和输出，允许用户决定是否继续
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.decisions: list[dict] = []   # 记录所有人工决策，供日志追溯

    def _record(self, checkpoint: str, action: str, detail: str = ""):
        self.decisions.append({
            "checkpoint": checkpoint, "action": action, "detail": detail
        })

    # ── 通用确认（y/n）────────────────────────────────────────
    def request_approval(self, task: str, context: str, agent: str) -> bool:
        if not self.enabled:
            return True
        print(f"\n{'='*58}")
        print(f"  [Human Review] {agent}")
        print(f"{'='*58}")
        print(f"  Task: {task}")
        if context:
            print(f"  Context: {context[:300]}")
        print(f"{'='*58}")
        answer = input("  Approve? [Y/n] ").strip().lower()
        approved = answer in ("", "y", "yes")
        self._record(agent, "approved" if approved else "rejected", task)
        return approved

    # ── 审核点 1：规划方案审核 ────────────────────────────────
    def review_plan(self, plan) -> dict:
        """
        展示规划结果，允许用户：
          - 选择不同的研究路径（输入路径编号）
          - 跳过某些任务（输入要跳过的任务 ID）
          - 直接确认（回车）
        返回: {"action": "approve"|"select_path"|"skip_tasks", ...}
        """
        if not self.enabled:
            return {"action": "approve"}

        print(f"\n{'='*58}")
        print(f"  [Human Review] Research Plan")
        print(f"{'='*58}")
        print(f"  Query: {plan.query[:60]}")

        # 展示所有路径及评分
        if plan.all_paths:
            print(f"\n  Available research paths:")
            for i, p in enumerate(plan.all_paths):
                selected = " <-- current" if p.get("description") == plan.selected_path else ""
                print(f"    [{i+1}] score={p.get('score', '?'):.2f}  "
                      f"{p.get('description', '')[:50]}{selected}")

        # 展示任务列表
        print(f"\n  Task list:")
        for t in plan.tasks:
            deps = f" (depends: {', '.join(t.depends_on)})" if t.depends_on else ""
            print(f"    [{t.task_id}] {t.name:18s} -> {t.agent}{deps}")

        print(f"\n{'='*58}")
        print(f"  Options:")
        print(f"    [Enter]     Approve current plan")
        print(f"    [1-{len(plan.all_paths)}]       Select a different path (re-plan)")
        print(f"    [skip T2,T4] Skip specific tasks")
        print(f"    [abort]     Abort entire run")
        print(f"{'='*58}")

        answer = input("  Your choice: ").strip()

        if not answer:
            self._record("plan_review", "approve")
            return {"action": "approve"}

        if answer.lower() == "abort":
            self._record("plan_review", "abort")
            return {"action": "abort"}

        # 选择路径
        if answer.isdigit():
            idx = int(answer) - 1
            if 0 <= idx < len(plan.all_paths):
                chosen = plan.all_paths[idx]
                self._record("plan_review", "select_path",
                             chosen.get("description", ""))
                return {"action": "select_path", "path_index": idx,
                        "path": chosen.get("description", "")}

        # 跳过任务
        if answer.lower().startswith("skip"):
            parts = answer.replace("skip", "").strip().split(",")
            skip_ids = [p.strip().upper() for p in parts if p.strip()]
            self._record("plan_review", "skip_tasks", str(skip_ids))
            return {"action": "skip_tasks", "skip_ids": skip_ids}

        self._record("plan_review", "approve", "unrecognized input, auto-approve")
        return {"action": "approve"}

    # ── 审核点 2：实验设计审核 ────────────────────────────────
    def review_experiment(self, query: str, exp_design: dict) -> dict:
        """
        展示实验设计，允许用户：
          - 直接确认（回车）
          - 修改假设（输入新假设）
          - 添加/替换基线或指标
        返回: {"action": "approve"|"modify", "modifications": {...}}
        """
        if not self.enabled:
            return {"action": "approve"}

        print(f"\n{'='*58}")
        print(f"  [Human Review] Experiment Design")
        print(f"{'='*58}")
        hyp = exp_design.get("research_hypothesis", "N/A")
        print(f"  Hypothesis: {hyp[:80]}")
        baselines = exp_design.get("baselines", [])
        print(f"  Baselines:  {', '.join(str(b)[:30] for b in baselines[:5])}")
        metrics = exp_design.get("metrics", [])
        print(f"  Metrics:    {', '.join(str(m)[:25] for m in metrics[:5])}")
        desc = exp_design.get("full_description", "")
        if desc:
            print(f"  Description: {desc[:120]}...")

        print(f"\n{'='*58}")
        print(f"  Options:")
        print(f"    [Enter]          Approve")
        print(f"    [h: <text>]      Change hypothesis")
        print(f"    [b: A, B, C]     Replace baselines")
        print(f"    [m: X, Y]        Replace metrics")
        print(f"    [abort]          Abort entire run")
        print(f"{'='*58}")

        answer = input("  Your choice: ").strip()

        if not answer:
            self._record("experiment_review", "approve")
            return {"action": "approve"}

        if answer.lower() == "abort":
            self._record("experiment_review", "abort")
            return {"action": "abort"}

        modifications = {}
        if answer.lower().startswith("h:"):
            modifications["research_hypothesis"] = answer[2:].strip()
        elif answer.lower().startswith("b:"):
            modifications["baselines"] = [
                b.strip() for b in answer[2:].split(",") if b.strip()
            ]
        elif answer.lower().startswith("m:"):
            modifications["metrics"] = [
                m.strip() for m in answer[2:].split(",") if m.strip()
            ]

        if modifications:
            self._record("experiment_review", "modify", str(modifications))
            return {"action": "modify", "modifications": modifications}

        self._record("experiment_review", "approve", "unrecognized input")
        return {"action": "approve"}

    # ── 审核点 3：代码执行结果审核 ────────────────────────────
    def review_code(self, code_result: dict) -> dict:
        """
        展示代码执行结果，允许用户：
          - 确认继续（回车）
          - 要求重新生成代码（retry）
          - 跳过后续分析（skip）
        返回: {"action": "approve"|"retry"|"skip"|"abort"}
        """
        if not self.enabled:
            return {"action": "approve"}

        success = code_result.get("success", False)
        code = code_result.get("final_code", "")
        stdout = code_result.get("stdout", "")
        rounds = code_result.get("total_rounds", 0)

        print(f"\n{'='*58}")
        print(f"  [Human Review] Code Execution Result")
        print(f"{'='*58}")
        print(f"  Status: {'SUCCESS' if success else 'FAILED'}")
        print(f"  Debug rounds: {rounds}")
        if code:
            # 显示代码前 15 行
            lines = code.strip().split("\n")
            print(f"  Code ({len(lines)} lines):")
            for line in lines[:15]:
                print(f"    {line}")
            if len(lines) > 15:
                print(f"    ... ({len(lines)-15} more lines)")
        if stdout:
            print(f"  Output: {stdout[:200]}")

        print(f"\n{'='*58}")
        print(f"  Options:")
        print(f"    [Enter]   Approve, continue to analysis")
        print(f"    [retry]   Regenerate code")
        print(f"    [skip]    Skip analysis, go to report")
        print(f"    [abort]   Abort entire run")
        print(f"{'='*58}")

        answer = input("  Your choice: ").strip().lower()

        if not answer:
            self._record("code_review", "approve")
            return {"action": "approve"}
        if answer == "retry":
            self._record("code_review", "retry")
            return {"action": "retry"}
        if answer == "skip":
            self._record("code_review", "skip")
            return {"action": "skip"}
        if answer == "abort":
            self._record("code_review", "abort")
            return {"action": "abort"}

        self._record("code_review", "approve", "unrecognized input")
        return {"action": "approve"}
