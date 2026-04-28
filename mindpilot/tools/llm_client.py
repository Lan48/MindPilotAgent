"""
统一 LLM 客户端
===============
兼容任何 OpenAI Chat Completions 格式的接口。

百炼 Coding Plan 支持的模型（2026-01）：
  千问：qwen3.5-plus / qwen3-max-2026-01-23 / qwen3-coder-next / qwen3-coder-plus
  智谱：glm-5 / glm-4.7
  Kimi：kimi-k2.5
  MiniMax：MiniMax-M2.5
"""

# 在任何 openai/httpx import 之前静默第三方日志
import logging as _logging
for _lib in ("openai", "httpx", "httpcore", "urllib3", "requests"):
    _l = _logging.getLogger(_lib)
    _l.setLevel(_logging.CRITICAL)
    _l.propagate = False

import json
import time
import random
from typing import Optional

# 百炼 Coding Plan 支持的合法模型名前缀（根据截图）
_VALID_PREFIXES = (
    "qwen3", "qwen3.", "qwen-",      # 千问系列
    "glm-",                           # 智谱系列
    "kimi-",                          # Kimi
    "minimax-", "MiniMax-",           # MiniMax
)
_BAILIAN_HOSTS = ("coding.dashscope.aliyuncs.com", "dashscope.aliyuncs.com")

# 常见错误模型名 → 正确模型名的提示
_MODEL_SUGGESTIONS = {
    "qwen-coder-plus":  "qwen3-coder-plus",
    "qwen-coder-turbo": "qwen3-coder-next",
    "qwen-plus":        "qwen3.5-plus",
    "qwen-max":         "qwen3-max-2026-01-23",
    "qwen-turbo":       "qwen3.5-plus",
    "gpt-4o":           "qwen3.5-plus 或 glm-5",
    "gpt-4":            "qwen3.5-plus 或 glm-5",
}


def _warn_model(model: str, base_url: str):
    """若模型名可能有误，打印警告和建议"""
    if not any(h in base_url for h in _BAILIAN_HOSTS):
        return
    m_lower = model.lower()
    # 已知错误拼写
    if model in _MODEL_SUGGESTIONS:
        print(f"\n[LLMClient] ⚠ 模型名可能有误：'{model}'")
        print(f"            百炼 Coding Plan 中对应的正确名称是：{_MODEL_SUGGESTIONS[model]}")
        print(f"            请修改 .env 中的 LLM_MODEL 或 CODE_MODEL。\n")
        return
    # 完全不认识的前缀
    if not any(m_lower.startswith(p.lower()) for p in _VALID_PREFIXES):
        print(f"\n[LLMClient] ⚠ 未识别的模型名：'{model}'")
        print(f"            百炼 Coding Plan 支持的模型（2026-01）：")
        print(f"              千问：qwen3.5-plus / qwen3-max-2026-01-23 / qwen3-coder-next / qwen3-coder-plus")
        print(f"              智谱：glm-5 / glm-4.7")
        print(f"              其他：kimi-k2.5 / MiniMax-M2.5\n")


class LLMClient:
    def __init__(self, config):
        self.config = config
        self.mock_mode = config.mock_mode
        self._client = None
        self._client_code = None

        if not self.mock_mode:
            _warn_model(config.llm.model, config.llm.base_url)
            _warn_model(config.llm.code_model, config.llm.base_url)
            self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI
            import httpx
        except ImportError as e:
            print(f"[LLMClient] ⚠ 依赖未安装（{e}），已切换 Mock 模式。")
            print("            请运行: pip install openai httpx")
            self.mock_mode = True
            return

        explicit_proxy = getattr(self.config.llm, "proxy_url", None)

        if explicit_proxy:
            print(f"[LLMClient] ℹ 使用指定代理: {explicit_proxy}")
            ok = self._probe(OpenAI(
                api_key=self.config.llm.api_key,
                base_url=self.config.llm.base_url,
                http_client=httpx.Client(proxy=explicit_proxy, timeout=15),
                max_retries=0,
            ), label="指定代理")
            if ok:
                self._build_clients(OpenAI, httpx, trust_env=False, proxy=explicit_proxy)
            else:
                self.mock_mode = True
            return

        # 自动探测：先直连，再系统代理
        print("[LLMClient] 正在探测接口连通性...")

        probe1 = OpenAI(
            api_key=self.config.llm.api_key,
            base_url=self.config.llm.base_url,
            http_client=httpx.Client(trust_env=False, timeout=15),
            max_retries=0,
        )
        if self._probe(probe1, label="直连", silent=True):
            print("[LLMClient] ✓ 直连成功（已绕过系统代理）")
            self._build_clients(OpenAI, httpx, trust_env=False, proxy=None)
            return

        print("[LLMClient]   直连失败，尝试通过系统代理...")
        probe2 = OpenAI(
            api_key=self.config.llm.api_key,
            base_url=self.config.llm.base_url,
            http_client=httpx.Client(trust_env=True, timeout=20),
            max_retries=0,
        )
        if self._probe(probe2, label="系统代理", silent=True):
            print("[LLMClient] ✓ 系统代理连接成功")
            self._build_clients(OpenAI, httpx, trust_env=True, proxy=None)
            return

        self._report_failure()
        self.mock_mode = True

    def _build_clients(self, OpenAI, httpx, trust_env: bool, proxy: Optional[str]):
        t = self.config.llm.timeout

        def _mk(timeout):
            if proxy:
                return httpx.Client(proxy=proxy, timeout=timeout)
            return httpx.Client(trust_env=trust_env, timeout=timeout)

        self._client = OpenAI(
            api_key=self.config.llm.api_key,
            base_url=self.config.llm.base_url,
            http_client=_mk(t),
            max_retries=1,
        )
        self._client_code = OpenAI(
            api_key=self.config.llm.api_key,
            base_url=self.config.llm.base_url,
            http_client=_mk(t),
            max_retries=1,
        )
        print(f"[LLMClient] ✓ 接口就绪")
        print(f"            接入点   : {self.config.llm.base_url}")
        print(f"            通用模型 : {self.config.llm.model}")
        print(f"            代码模型 : {self.config.llm.code_model}")

    def _probe(self, client, label: str = "", silent: bool = False) -> bool:
        try:
            resp = client.chat.completions.create(
                model=self.config.llm.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5,
            )
            _ = resp.choices[0].message.content
            return True
        except Exception as e:
            if not silent:
                print(f"[LLMClient] ✗ {label}失败: {self._explain(str(e))}")
            return False

    def _report_failure(self):
        print(f"\n[LLMClient] ✗ 直连和系统代理均失败 → 已切换 Mock 模式")
        print(f"            接入点 : {self.config.llm.base_url}")
        print(f"            模型   : {self.config.llm.model}")
        print(f"")
        print(f"            排查步骤：")
        print(f"            1. 确认 LLM_MODEL / CODE_MODEL 使用百炼支持的名称（见 config.example.env）")
        print(f"            2. 确认 LLM_API_KEY 有效（百炼控制台 → API-KEY 管理）")
        print(f"            3. 确认 LLM_BASE_URL 正确")
        print(f"            4. 若需代理：设置 LLM_PROXY_URL=http://127.0.0.1:7890\n")

    @staticmethod
    def _explain(err: str) -> str:
        e = err.lower()
        if "10061" in err or "actively refused" in e or "connection refused" in e:
            return "连接被拒绝 (10061) — 代理未启动或拒绝转发"
        if "10060" in err or "timed out" in e or "timeout" in e:
            return "连接超时 (10060) — 网络不可达或接入点地址有误"
        if "401" in err or "unauthorized" in e:
            return "API Key 无效 (401)"
        if "403" in err or "forbidden" in e:
            return "访问被拒绝 (403) — Key 无权限"
        if "400" in err and "not supported" in e:
            return "模型名无效 (400) — 请检查 LLM_MODEL/CODE_MODEL 是否为百炼支持的名称"
        if "404" in err or "not found" in e:
            return "模型不存在 (404)"
        return err[:120]

    # ── 对话接口 ────────────────────────────────────────────
    def chat(self, messages: list[dict], model: Optional[str] = None,
            temperature: Optional[float] = None, max_tokens: Optional[int] = None,
            use_code_model: bool = False) -> str:

        chosen_model = model or (
            self.config.llm.code_model if use_code_model else self.config.llm.model
        )

        # 代码模型使用单独参数；普通模型使用通用参数
        default_temperature = (
            getattr(self.config.llm, "code_temperature", self.config.llm.temperature)
            if use_code_model else self.config.llm.temperature
        )
        default_max_tokens = (
            getattr(self.config.llm, "code_max_tokens", self.config.llm.max_tokens)
            if use_code_model else self.config.llm.max_tokens
        )

        actual_temperature = temperature if temperature is not None else default_temperature
        actual_max_tokens = max_tokens or default_max_tokens

        print(
            f"[LLM DEBUG] mock_mode={self.mock_mode}, "
            f"use_code_model={use_code_model}, "
            f"model={'MOCK' if self.mock_mode else chosen_model}, "
            f"temperature={actual_temperature}, "
            f"max_tokens={actual_max_tokens}"
        )

        if self.mock_mode:
            print("[LLM DEBUG] returning from mock")
            content = self._mock_response(messages)
            print("[LLM DEBUG] first 80 chars:", repr(content[:80]))
            return content

        client = self._client_code if use_code_model else self._client

        try:
            resp = client.chat.completions.create(
                model=chosen_model,
                messages=messages,
                temperature=actual_temperature,
                max_tokens=actual_max_tokens,
            )

            content = resp.choices[0].message.content or ""
            finish_reason = getattr(resp.choices[0], "finish_reason", None)

            print("[LLM DEBUG] returning from real api")
            print("[LLM DEBUG] finish_reason:", finish_reason)
            print("[LLM DEBUG] first 80 chars:", repr(content[:80]))

            if finish_reason == "length":
                print(
                    "[LLM WARNING] 输出可能因为 max_tokens 上限被截断，"
                    "建议增大 CODE_MAX_TOKENS 或缩小代码生成任务。"
                )

            return content

        except Exception as e:
            err_msg = self._explain(str(e))
            print(f"[LLMClient] ✗ 调用失败 (model={chosen_model}): {err_msg}")

            if "400" in str(e) and "not supported" in str(e).lower():
                print(
                    f"            → 请将 .env 中 {'CODE_MODEL' if use_code_model else 'LLM_MODEL'} "
                    f"改为百炼支持的模型名（如 qwen3-coder-plus / glm-5）"
                )

            print("[LLM DEBUG] returning from mock")
            content = self._mock_response(messages, error=str(e))
            print("[LLM DEBUG] first 80 chars:", repr(content[:80]))
            return content

    def chat_code(self, messages: list[dict], **kwargs) -> str:
        return self.chat(messages, use_code_model=True, **kwargs)

    # ── Mock 模式 ────────────────────────────────────────────
    def _mock_response(self, messages: list[dict], error: str = "") -> str:
        time.sleep(0.05)
        last   = messages[-1].get("content", "") if messages else ""
        system = next((m.get("content", "") for m in messages
                       if m.get("role") == "system"), "")

        if any(k in system for k in ["分解", "规划", "plan", "subtask"]):
            return json.dumps({
                "tasks": [
                    {"id":"T1","name":"文献检索","agent":"LiteratureAgent",
                     "description":"检索相关学术文献","depends_on":[]},
                    {"id":"T2","name":"代码实现","agent":"CodeAgent",
                     "description":"实现核心算法代码","depends_on":["T1"]},
                    {"id":"T3","name":"数据分析","agent":"AnalysisAgent",
                     "description":"分析实验结果并可视化","depends_on":["T2"]},
                    {"id":"T4","name":"报告生成","agent":"EvaluationAgent",
                     "description":"综合输出完整研究报告","depends_on":["T1","T2","T3"]},
                ],
                "reasoning":"先检索文献建立背景知识，再实现核心代码，分析结果后生成完整报告。"
            }, ensure_ascii=False)

        if any(k in system for k in ["thought", "路径", "path"]):
            return json.dumps({"score": round(random.uniform(0.55,0.95),2),
                               "reasoning":"该路径逻辑合理，覆盖了主要研究方向。"},
                              ensure_ascii=False)

        if any(k in system for k in ["代码","Python","python","code","编程","实现"]) or \
           any(k in last.lower() for k in ["python","代码","实现","算法"]):
            return """```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

np.random.seed(42)
X = np.random.randn(100, 1) * 10
y = 2.5 * X.flatten() + np.random.randn(100) * 5

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
print(f"R2 = {r2:.4f}, MSE = {mse:.4f}")

plt.figure(figsize=(8,5))
plt.scatter(X, y, alpha=0.6, label='data')
plt.plot(X, y_pred, 'r-', linewidth=2, label=f'fit (R2={r2:.3f})')
plt.legend(); plt.title('Linear Regression'); plt.tight_layout()
plt.savefig('regression_result.png', dpi=150)
print("chart saved")
```"""

        if any(k in system for k in ["摘要","summary","abstract","综述"]):
            return json.dumps({
                "method":"本文提出了一种基于Transformer的多模态学习框架，融合文本与视觉特征。",
                "conclusion":"在三个标准基准上，所提方法相比最优基线提升了8.3%的准确率。",
                "limitation":"模型计算复杂度较高，边缘设备部署存在挑战。",
                "score":round(random.uniform(0.70,0.95),2)
            }, ensure_ascii=False)

        if any(k in system for k in ["评估","评审","judge","Judge","评分","质量"]):
            score = round(random.uniform(0.62,0.90),2)
            return json.dumps({
                "overall_score":score,
                "accuracy":round(random.uniform(0.65,0.95),2),
                "completeness":round(random.uniform(0.60,0.90),2),
                "format_quality":round(random.uniform(0.70,0.95),2),
                "feedback":"内容基本完整，建议补充更多对比数据。" if score<0.75 else "输出质量良好，逻辑清晰。",
                "needs_reflection":score<0.65,
                "reflection_suggestion":"请补充实验数据。" if score<0.65 else ""
            }, ensure_ascii=False)

        return f"[Mock 模式] 已处理：{last[:80]}...\n提示：配置正确的 LLM_API_KEY 后将调用真实模型。"
