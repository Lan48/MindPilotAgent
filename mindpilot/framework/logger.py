"""
模块⑤ — 执行日志系统
====================
记录每个 Agent 的调用链、耗时、输入输出，支持可视化回放。
"""

import json
import time
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"


@dataclass
class AgentCall:
    """单次 Agent 调用记录"""
    call_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_name: str = ""
    task_id: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    input_data: Any = None
    output_data: Any = None
    status: str = "running"   # running | success | failed | retrying
    error: Optional[str] = None
    reflection_round: int = 0
    metadata: dict = field(default_factory=dict)

    def finish(self, output: Any, status: str = "success"):
        self.end_time = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)
        self.output_data = output
        self.status = status

    def fail(self, error: str):
        self.finish(None, status="failed")
        self.error = error

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert non-serializable types
        if isinstance(d.get("input_data"), (dict, list, str, int, float, bool, type(None))):
            pass
        else:
            d["input_data"] = str(d["input_data"])
        if isinstance(d.get("output_data"), (dict, list, str, int, float, bool, type(None))):
            pass
        else:
            d["output_data"] = str(d["output_data"])
        return d


class MindPilotLogger:
    """
    MindPilot 统一日志系统
    - 控制台彩色输出
    - JSON 结构化日志文件
    - 完整调用链记录
    """

    COLORS = {
        "DEBUG": "\033[37m",
        "INFO": "\033[36m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "SUCCESS": "\033[32m",
        "RESET": "\033[0m",
        "BOLD": "\033[1m",
    }

    AGENT_COLORS = {
        "Orchestrator": "\033[35m",
        "PlanningAgent": "\033[34m",
        "LiteratureAgent": "\033[32m",
        "CodeAgent": "\033[33m",
        "AnalysisAgent": "\033[36m",
        "EvaluationAgent": "\033[31m",
        "Framework": "\033[37m",
    }

    def __init__(self, session_id: Optional[str] = None, log_dir: str = "logs", verbose: bool = True):
        self.session_id = session_id or str(uuid.uuid4())[:12]
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.call_history: list[AgentCall] = []
        self._current_calls: dict[str, AgentCall] = {}

        # File logger
        self.log_file = self.log_dir / f"session_{self.session_id}.jsonl"
        self.summary_file = self.log_dir / f"summary_{self.session_id}.json"

        # ── Python 日志配置 ──────────────────────────────────────────
        # 关键：只把 MindPilot 自己的 logger 设为 DEBUG/INFO，
        # 根 logger 保持 WARNING，避免 openai / httpx 等第三方库
        # 的 DEBUG/INFO 日志（包括重试 traceback）打印到终端。
        logging.getLogger("openai").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("httpcore").setLevel(logging.ERROR)
        logging.getLogger("urllib3").setLevel(logging.ERROR)

        # 根 logger 只输出 WARNING 及以上（不干扰第三方库）
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.WARNING)
            root_logger.addHandler(handler)
        root_logger.setLevel(logging.WARNING)

        # MindPilot 专属 logger：按 verbose 决定级别
        self._py_logger = logging.getLogger(f"mindpilot.{self.session_id}")
        self._py_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        self._py_logger.propagate = False   # 不向根 logger 传播，避免重复输出

    def _color(self, text: str, color: str) -> str:
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['RESET']}"

    def _agent_color(self, agent: str, text: str) -> str:
        color = self.AGENT_COLORS.get(agent, "\033[37m")
        return f"{color}{text}{self.COLORS['RESET']}"

    def _write_jsonl(self, entry: dict):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log(self, level: str, agent: str, message: str, data: Any = None):
        ts = datetime.now().strftime("%H:%M:%S")
        entry = {
            "ts": datetime.now().isoformat(),
            "session": self.session_id,
            "level": level,
            "agent": agent,
            "message": message,
            "data": data,
        }
        self._write_jsonl(entry)

        if self.verbose:
            level_str = self._color(f"[{level:7s}]", level)
            agent_str = self._agent_color(agent, f"[{agent:18s}]")
            print(f"{self._color(ts, 'DEBUG')} {level_str} {agent_str} {message}")

    def info(self, agent: str, message: str, data: Any = None):
        self.log("INFO", agent, message, data)

    def success(self, agent: str, message: str, data: Any = None):
        self.log("SUCCESS", agent, f"✓ {message}", data)

    def warning(self, agent: str, message: str, data: Any = None):
        self.log("WARNING", agent, f"⚠ {message}", data)

    def error(self, agent: str, message: str, data: Any = None):
        self.log("ERROR", agent, f"✗ {message}", data)

    def debug(self, agent: str, message: str, data: Any = None):
        self.log("DEBUG", agent, message, data)

    def log_message(self, msg: Any):
        """记录符合通信协议的消息，便于把 Agent 调用和消息流对应起来。"""
        data = msg.to_dict() if hasattr(msg, "to_dict") else msg
        msg_type = data.get("msg_type", "UNKNOWN") if isinstance(data, dict) else "UNKNOWN"
        sender = data.get("sender", "") if isinstance(data, dict) else ""
        receiver = data.get("receiver", "") if isinstance(data, dict) else ""
        task_id = data.get("task_id", "") if isinstance(data, dict) else ""
        self.log(
            "DEBUG",
            "Framework",
            f"消息流 {msg_type}: {sender} -> {receiver} task={task_id}",
            data,
        )

    def start_call(self, agent_name: str, task_id: str, input_data: Any) -> AgentCall:
        call = AgentCall(agent_name=agent_name, task_id=task_id, input_data=input_data)
        self._current_calls[call.call_id] = call
        self.call_history.append(call)
        self.info(agent_name, f"▶ 开始执行 task={task_id}", {"call_id": call.call_id})
        return call

    def finish_call(self, call: AgentCall, output: Any):
        call.finish(output)
        self._current_calls.pop(call.call_id, None)
        self.success(call.agent_name, f"完成 [{call.duration_ms}ms] task={call.task_id}")
        self._write_jsonl({"event": "call_complete", **call.to_dict()})

    def fail_call(self, call: AgentCall, error: str):
        call.fail(error)
        self._current_calls.pop(call.call_id, None)
        self.error(call.agent_name, f"失败 task={call.task_id}: {error}")
        self._write_jsonl({"event": "call_failed", **call.to_dict()})

    def save_summary(self, extra: Optional[dict] = None):
        """保存会话总结"""
        total = len(self.call_history)
        success = sum(1 for c in self.call_history if c.status == "success")
        failed = sum(1 for c in self.call_history if c.status == "failed")
        durations = [c.duration_ms for c in self.call_history if c.duration_ms]
        avg_dur = round(sum(durations) / len(durations), 2) if durations else 0

        summary = {
            "session_id": self.session_id,
            "total_calls": total,
            "success": success,
            "failed": failed,
            "avg_duration_ms": avg_dur,
            "calls": [c.to_dict() for c in self.call_history],
        }
        if extra:
            summary.update(extra)
        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        self.info("Framework", f"会话日志已保存: {self.summary_file}")
        return summary

    def print_call_chain(self):
        """打印完整调用链"""
        print(f"\n{'='*60}")
        print(f"  调用链回放  Session: {self.session_id}")
        print(f"{'='*60}")
        for i, call in enumerate(self.call_history):
            status_icon = "✓" if call.status == "success" else "✗"
            dur = f"{call.duration_ms}ms" if call.duration_ms else "running"
            print(f"  {i+1:2d}. {status_icon} [{call.agent_name:18s}] "
                  f"task={call.task_id:20s} {dur}")
        print(f"{'='*60}\n")
