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
    关键决策节点暂停等待用户确认
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.auto_approve_keywords = ["data analysis", "literature search", "visualization"]

    def request_approval(self, task: str, context: str, agent: str) -> bool:
        """
        请求用户批准。
        - 在交互模式下向用户提问
        - 在批处理模式下自动批准低风险任务
        """
        if not self.enabled:
            return True

        # 低风险任务自动批准
        task_lower = task.lower()
        if any(kw in task_lower for kw in self.auto_approve_keywords):
            return True

        print(f"\n{'━'*55}")
        print(f"  🤚 人机协同确认  [Agent: {agent}]")
        print(f"{'━'*55}")
        print(f"  任务: {task}")
        print(f"  上下文: {context[:200]}...")
        print(f"{'━'*55}")
        answer = input("  是否批准执行？[y/N] ").strip().lower()
        return answer in ("y", "yes", "是")
