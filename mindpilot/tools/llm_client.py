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
        if self.mock_mode:
            return self._mock_response(messages)

        chosen_model = model or (
            self.config.llm.code_model if use_code_model else self.config.llm.model
        )
        client = self._client_code if use_code_model else self._client

        try:
            resp = client.chat.completions.create(
                model=chosen_model,
                messages=messages,
                temperature=temperature if temperature is not None else self.config.llm.temperature,
                max_tokens=max_tokens or self.config.llm.max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            err_msg = self._explain(str(e))
            print(f"[LLMClient] ✗ 调用失败 (model={chosen_model}): {err_msg}")
            # 400 模型名错误时给出具体建议
            if "400" in str(e) and "not supported" in str(e).lower():
                print(f"            → 请将 .env 中 {'CODE_MODEL' if use_code_model else 'LLM_MODEL'} "
                      f"改为百炼支持的模型名（如 qwen3-coder-plus / glm-5）")
            return self._mock_response(messages, error=str(e))

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
                    {"id":"T2","name":"实验设计","agent":"EvaluationAgent",
                     "description":"基于文献综述设计实验假设、数据集、基线方法、评估指标和可复现流程","depends_on":["T1"]},
                    {"id":"T3","name":"代码实现","agent":"CodeAgent",
                     "description":"根据实验设计实现核心算法代码","depends_on":["T2"]},
                    {"id":"T4","name":"数据分析","agent":"AnalysisAgent",
                     "description":"分析实验结果并可视化","depends_on":["T3"]},
                    {"id":"T5","name":"报告生成","agent":"EvaluationAgent",
                     "description":"综合输出完整研究报告","depends_on":["T1","T2","T3","T4"]},
                ],
                "reasoning":"先检索文献建立背景知识，再设计可复现实验方案，然后实现核心代码，分析结果后生成完整报告。"
            }, ensure_ascii=False)

        if any(k in system for k in ["科研方法评审专家", "评分（0-1）"]):
            return json.dumps({"score": round(random.uniform(0.55,0.95),2),
                               "reasoning":"该路径逻辑合理，覆盖了主要研究方向。"},
                              ensure_ascii=False)

        if any(k in system for k in ["实验设计", "实验方案", "科研实验设计"]):
            return json.dumps({
                "research_hypothesis": "引入轻量化跨模态连接模块能够提升视觉语言模型在图文对齐任务上的准确率，同时降低推理延迟和显存占用。",
                "objectives": [
                    "验证改进连接模块相较于标准投影层的性能增益",
                    "分析不同连接策略在准确率、效率和稳定性上的权衡",
                    "通过消融实验定位门控融合、残差连接等组件的贡献"
                ],
                "dataset": "实验可选用 COCO Captions、VQAv2 与 Flickr30K 等公开数据集。数据预处理包括图像尺寸标准化、文本分词、异常样本过滤，并按照训练集、验证集和测试集划分，保证各模型使用相同数据切分。",
                "baselines": [
                    "Linear Projection: 使用单层线性投影连接视觉编码器与语言模型",
                    "MLP Adapter: 使用两层 MLP 完成视觉特征到语言空间的映射",
                    "Cross-Attention Connector: 通过交叉注意力完成图文特征融合"
                ],
                "metrics": [
                    "Accuracy = 正确样本数 / 总样本数，用于衡量问答或分类任务表现",
                    "Recall@K = Top-K 检索结果中包含正确目标的比例，用于衡量图文检索能力",
                    "Latency = 单样本平均推理时间，用于评估部署效率",
                    "Memory = 推理峰值显存占用，用于衡量资源开销"
                ],
                "variables": {
                    "independent": ["连接模块类型", "视觉特征维度", "训练轮数"],
                    "dependent": ["Accuracy", "Recall@K", "Latency", "Memory"],
                    "controlled": ["数据集划分", "随机种子", "优化器", "batch size"]
                },
                "ablations": [
                    "移除门控融合模块，观察准确率与稳定性的变化",
                    "移除残差连接，分析深层特征传递对性能的影响",
                    "改变投影维度，比较参数量与效果之间的权衡"
                ],
                "procedure": [
                    "统一数据预处理和训练配置，构建可复现实验环境",
                    "分别训练基线模型与改进模型，记录训练日志和验证集指标",
                    "在测试集上评估准确率、Recall@K、延迟和显存占用",
                    "执行消融实验和重复实验，并对主要指标做显著性检验",
                    "结合图表分析不同方法的性能差异和适用场景"
                ],
                "reproducibility": "固定随机种子为 42，每组实验至少重复 3 次；记录 Python、PyTorch、CUDA 版本和硬件配置；使用均值、标准差和配对 t 检验报告结果。",
                "expected_results": "预期改进连接模块在保持较低资源开销的同时提升图文对齐质量，尤其在 VQA 和图文检索任务上优于简单投影层。消融实验预计会显示门控融合对准确率提升贡献最大，残差连接对训练稳定性更关键。",
                "full_description": "本实验围绕视觉语言模型中视觉编码器与语言模型之间的连接机制展开。整体设计遵循控制变量原则，在统一数据集、训练策略和评估协议的前提下，对比线性投影、MLP Adapter、交叉注意力连接器和改进连接模块的性能。实验不仅关注任务准确率，还纳入推理延迟、显存占用和 Recall@K 等效率指标，从效果和资源开销两个角度判断方法优劣。通过消融实验进一步拆解门控融合、残差连接和投影维度等因素的贡献，使结论更具解释性和可复现性。",
                "sections": [
                    {
                        "heading": "3.1 跨模态连接机制的研究假设",
                        "body": "本实验假设轻量化跨模态连接模块能够在视觉编码器和语言模型之间建立更稳定的表征映射，从而提升图文对齐质量。相比单层线性投影，改进模块通过门控融合和残差路径保留视觉细节，并降低噪声特征对语言推理的干扰。实验目标是验证该假设在 VQA、图文检索和推理效率上的综合表现。"
                    },
                    {
                        "heading": "3.2 VQA 与图文检索数据集构建",
                        "body": "数据部分选取 COCO Captions、VQAv2 和 Flickr30K 等公开基准，覆盖图文描述、视觉问答和跨模态检索三类任务。预处理阶段统一图像尺寸、文本分词规则和样本过滤标准，并保持训练集、验证集和测试集划分一致，确保不同连接模块之间的比较具有可重复性。"
                    },
                    {
                        "heading": "3.3 投影层、Adapter 与交叉注意力基线",
                        "body": "对照组包括 Linear Projection、MLP Adapter 和 Cross-Attention Connector。Linear Projection 用于评估最简单连接方式的下限表现，MLP Adapter 反映非线性映射能力，Cross-Attention Connector 则作为更强但开销更高的融合基线。改进模块需要同时与这些方法比较准确率和资源消耗。"
                    },
                    {
                        "heading": "3.4 准确率、Recall@K 与效率指标",
                        "body": "评估指标由任务性能和系统效率两部分构成。Accuracy 衡量视觉问答或分类任务表现，Recall@K 衡量图文检索排序质量，Latency 记录单样本平均推理时间，Memory 记录峰值显存占用。通过这些指标可以同时观察方法有效性和部署成本。"
                    },
                    {
                        "heading": "3.5 门控融合与残差路径的消融设计",
                        "body": "消融实验分别移除门控融合、残差连接并调整投影维度，以定位各组件对性能的贡献。实验中保持数据划分、优化器、batch size 和训练轮数不变，只改变连接模块内部结构，从而保证性能差异可以归因于具体设计。"
                    },
                    {
                        "heading": "3.6 可复现实验流程与预期分析",
                        "body": "实验流程包括统一预处理、训练各基线和改进模型、记录验证集指标、在测试集评估性能与效率，并进行至少三次重复实验。最终用均值、标准差和显著性检验报告结果，预期改进连接模块能在保持较低资源开销的同时提升 VLM 的跨模态对齐能力。"
                    }
                ]
            }, ensure_ascii=False)

        if "科研论文写作专家" in system:
            topic = last.split("\n", 1)[0].replace("：", "：")
            return (
                f"[Mock 模式] {topic}。本节围绕研究问题展开学术化说明，"
                "结合已有文献、实验目标和系统输出，对相关方法、实验设置与结果意义进行概括。"
                "在真实 API 模式下，该部分会由大模型生成更完整的中文学术段落。"
            )

        if any(k in system for k in ["Python 科研工程师", "Python 调试专家", "代码安全专家", "Python 专家"]):
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

        if "LLM-as-Judge" in system or "返回 JSON" in system and any(k in system for k in ["评分", "评审"]):
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
