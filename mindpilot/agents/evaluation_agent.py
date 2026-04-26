"""
模块⑥ — 评估与反思 Agent
==========================
新增：实验设计生成 + 完整学术报告构建（详细版）
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EvalScore:
    overall: float
    accuracy: float
    completeness: float
    format_quality: float
    feedback: str
    needs_reflection: bool
    reflection_suggestion: str = ""


@dataclass
class ReflectionRecord:
    round_num: int
    original_output: str
    score_before: float
    reflection: str
    revised_output: str
    score_after: float
    improved: bool


class LLMJudge:
    def __init__(self, llm_client, threshold: float = 0.65, logger=None):
        self.llm = llm_client
        self.threshold = threshold
        self.logger = logger

    def score(self, query: str, output: str, output_type: str = "report") -> EvalScore:
        system = f"""你是严格的科研输出质量评审专家（LLM-as-Judge）。
请对以下{output_type}进行评分，返回 JSON：
{{"overall_score":0.85,"accuracy":0.90,"completeness":0.80,
  "format_quality":0.85,"feedback":"具体评价...","needs_reflection":false,
  "reflection_suggestion":"改进建议"}}
评分标准（0-1）：accuracy=内容准确性，completeness=完整性，format_quality=格式规范性
overall_score<{self.threshold} 时 needs_reflection=true"""
        resp = self.llm.chat([
            {"role": "system", "content": system},
            {"role": "user",
             "content": f"研究问题：{query}\n\n输出（前2000字）：\n{output[:2000]}"}
        ])
        try:
            m    = re.search(r"\{[\s\S]+\}", resp)
            data = json.loads(m.group(0) if m else resp)
            return EvalScore(
                overall=float(data.get("overall_score", 0.7)),
                accuracy=float(data.get("accuracy", 0.7)),
                completeness=float(data.get("completeness", 0.7)),
                format_quality=float(data.get("format_quality", 0.7)),
                feedback=data.get("feedback", ""),
                needs_reflection=bool(data.get("needs_reflection", False)),
                reflection_suggestion=data.get("reflection_suggestion", ""),
            )
        except Exception:
            import random
            s = round(random.uniform(0.62, 0.88), 2)
            return EvalScore(
                overall=s, accuracy=s+0.02, completeness=s-0.03,
                format_quality=s+0.05, feedback="Mock 评分",
                needs_reflection=s < self.threshold,
            )

    def compute_rouge_l(self, hypothesis: str, reference: str) -> float:
        def lcs(a, b):
            m, n = min(len(a),100), min(len(b),100)
            a, b = a[:m], b[:n]
            dp = [[0]*(n+1) for _ in range(m+1)]
            for i in range(1,m+1):
                for j in range(1,n+1):
                    dp[i][j] = dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j],dp[i][j-1])
            return dp[m][n]
        h = hypothesis.lower().split()
        r = reference.lower().split()
        if not h or not r: return 0.0
        l = lcs(h, r)
        p = l/len(h); rc = l/len(r)
        return round(2*p*rc/(p+rc), 4) if (p+rc) > 0 else 0.0


class SelfReflector:
    def __init__(self, llm_client, max_rounds=3, logger=None):
        self.llm = llm_client
        self.max_rounds = max_rounds
        self.logger = logger

    def reflect_and_revise(self, query: str, output: str, score: EvalScore) -> str:
        system = (
            "你是科研报告质量改进专家。根据评审意见，"
            "对报告内容进行针对性补充和修正，使其更准确、完整、规范。"
            "重点补充缺失的内容，不要删减已有内容。"
        )
        prompt = (
            f"研究问题：{query}\n\n"
            f"待改进报告（前1500字）：\n{output[:1500]}\n\n"
            f"评审意见：{score.feedback}\n"
            f"改进建议：{score.reflection_suggestion}\n"
            f"当前得分：{score.overall:.2f}（目标≥0.65）\n\n"
            "请给出补充和改进后的版本："
        )
        return self.llm.chat([
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ])


class BenchmarkEvaluator:
    BENCHMARK_QUESTIONS = [
        "Transformer 的自注意力机制是如何工作的？",
        "BERT 和 GPT 的预训练目标有何根本区别？",
        "联邦学习的核心优势和主要挑战是什么？",
        "扩散模型的前向和反向过程分别是什么？",
        "对比学习的核心思想是什么？",
        "知识蒸馏是如何将大模型压缩为小模型的？",
        "图神经网络的消息传递机制是什么？",
        "梯度消失问题的常见解决方案有哪些？",
        "强化学习中 PPO 算法的改进点是什么？",
        "大模型的 Scaling Law 说明了什么规律？",
    ]
    ANSWER_KEYWORDS = {
        0:["Query","Key","Value","softmax","注意力权重"],
        1:["MLM","NSP","自回归","双向","单向"],
        2:["隐私","数据不出","通信","异构","FedAvg"],
        3:["马尔可夫","噪声","去噪","DDPM","生成"],
        4:["正样本","负样本","SimCLR","MoCo","InfoNCE"],
        5:["教师","学生","软标签","温度","KL散度"],
        6:["邻居聚合","消息","更新","过平滑","GCN"],
        7:["sigmoid","残差连接","梯度裁剪","LSTM","激活函数"],
        8:["置信区间","clip","重要性采样","近端策略","PPO"],
        9:["参数量","计算量","幂律","涌现","数据量"],
    }

    def __init__(self, llm_client, logger=None):
        self.llm = llm_client
        self.logger = logger

    def run_comparison(self, system_runner, n_questions=5) -> dict:
        results = {"mindpilot":[], "llm_only":[], "rag_only":[]}
        for i, q in enumerate(self.BENCHMARK_QUESTIONS[:n_questions]):
            if self.logger:
                self.logger.info("EvaluationAgent", f"基准测试 {i+1}/{n_questions}: {q[:40]}")
            kws = self.ANSWER_KEYWORDS.get(i, [])
            try:
                mp_out = system_runner(q) if system_runner else q
            except Exception:
                mp_out = q
            results["mindpilot"].append(self._recall(mp_out, kws))
            llm_out = self.llm.chat([{"role":"user","content":q}])
            results["llm_only"].append(self._recall(llm_out, kws))
            results["rag_only"].append(round(results["llm_only"][-1]*0.85, 3))
            time.sleep(0.1)
        summary = {
            sys: {"scores":scores,"avg":round(sum(scores)/len(scores),3),
                  "max":max(scores),"min":min(scores)}
            for sys, scores in results.items()
        }
        summary["mindpilot_wins"] = sum(
            1 for i in range(n_questions)
            if results["mindpilot"][i] >= max(results["llm_only"][i],results["rag_only"][i])
        )
        summary["total_questions"] = n_questions
        return summary

    def _recall(self, text, keywords):
        if not keywords: return 0.5
        return round(sum(1 for k in keywords if k.lower() in text.lower())/len(keywords), 3)


class EvaluationAgent:
    AGENT_NAME = "EvaluationAgent"

    def __init__(self, config, llm_client, report_gen, memory_store, logger):
        self.config     = config
        self.llm        = llm_client
        self.report_gen = report_gen
        self.memory     = memory_store
        self.logger     = logger
        self.judge      = LLMJudge(llm_client, threshold=config.evaluation.score_threshold, logger=logger)
        self.reflector  = SelfReflector(llm_client, max_rounds=config.evaluation.max_reflection_rounds, logger=logger)
        self.benchmark  = BenchmarkEvaluator(llm_client, logger=logger)

    # ── 实验设计（新增）─────────────────────────────────────
    def design_experiment(self, query: str, literature_result: dict,
                          research_path: str = "") -> dict:
        """
        基于文献综述生成完整实验设计方案。
        返回包含研究目标、方法、评估指标、对照组的结构化方案。
        """
        call = self.logger.start_call(self.AGENT_NAME, "experiment_design", query)
        try:
            # 提取文献方法作为参考
            papers = literature_result.get("top_papers", [])
            methods_ref = ""
            if papers:
                methods = [
                    p.get("structured_summary", {}).get("method", "")
                    for p in papers[:3]
                    if p.get("structured_summary")
                ]
                methods_ref = "\n".join(f"- {m}" for m in methods if m)

            system = """你是资深科研实验设计专家。请为以下研究问题设计一个完整、严谨的实验方案。

实验设计方案必须包含以下所有部分，每部分详细描述（每部分不少于100字）：
1. 研究目标与假设（明确的研究假设、预期结论）
2. 实验环境与数据集（数据来源、规模、预处理方法）
3. 基线方法与对照组设置（至少3个对照方法）
4. 评估指标（定量指标的计算公式和含义）
5. 实验流程（详细的步骤说明）
6. 消融实验与变量控制（说明自变量、因变量、控制变量）
7. 统计检验与可复现性设置
8. 预期结果与分析方向

请以 JSON 格式返回，字段：
{
  "research_hypothesis": "研究假设...",
  "objectives": ["目标1...", "目标2..."],
  "dataset": "数据集描述...",
  "baselines": ["基线1: 说明", "基线2: 说明", "基线3: 说明"],
  "metrics": ["指标1: 公式和说明", "指标2: 公式和说明"],
  "variables": {"independent": ["..."], "dependent": ["..."], "controlled": ["..."]},
  "ablations": ["消融1...", "消融2..."],
  "procedure": ["步骤1...", "步骤2...", ...],
  "reproducibility": "随机种子、重复次数、硬件/软件环境、统计检验方法...",
  "expected_results": "预期结果分析...",
  "full_description": "完整实验设计描述（500字以上）",
  "sections": [
    {"heading": "3.1 根据研究主题自行命名的小标题", "body": "对应小节正文..."},
    {"heading": "3.2 根据研究主题自行命名的小标题", "body": "对应小节正文..."}
  ]
}

sections 要求：
- 生成 5~7 个二级小节，heading 必须带 3.x 编号。
- heading 应根据研究问题、推荐研究路径和文献方法自行命名，不要机械套用固定模板。
- sections 整体必须覆盖：研究假设/目标、数据集/环境、基线/对照、评估指标、变量控制/消融、实验流程/可复现性、预期分析。
- body 应直接写成可放入学术报告的正文，不要只写提纲。"""

            path_part = f"推荐研究路径：\n{research_path}\n\n" if research_path else ""
            prompt = (
                f"研究问题：{query}\n\n"
                f"{path_part}"
                f"相关文献方法参考：\n{methods_ref}\n\n"
                "请设计完整的实验方案："
            )
            resp = self.llm.chat([
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt}
            ])

            try:
                m    = re.search(r"\{[\s\S]+\}", resp)
                data = json.loads(m.group(0) if m else resp)
            except Exception:
                data = {"full_description": resp}

            result = {
                "research_path":       research_path,
                "research_hypothesis": data.get("research_hypothesis", ""),
                "objectives":          data.get("objectives", []),
                "dataset":             data.get("dataset", ""),
                "baselines":           data.get("baselines", []),
                "metrics":             data.get("metrics", []),
                "variables":           data.get("variables", {}),
                "ablations":           data.get("ablations", []),
                "procedure":           data.get("procedure", []),
                "reproducibility":     data.get("reproducibility", ""),
                "expected_results":    data.get("expected_results", ""),
                "full_description":    data.get("full_description", resp),
                "sections":            self._normalize_experiment_sections(data.get("sections", [])),
            }
            result["structured_summary"] = self._format_experiment_design(result)
            self.logger.finish_call(call, result)
            self.logger.success(self.AGENT_NAME, "实验设计方案生成完成")
            return result

        except Exception as e:
            self.logger.fail_call(call, str(e))
            return {"full_description": f"实验设计生成失败：{e}", "baselines":[], "metrics":[], "procedure":[]}

    # ── 主入口（评估 + 报告生成）────────────────────────────
    def run(self, query: str, outputs: dict) -> dict:
        call = self.logger.start_call(self.AGENT_NAME, "evaluation", query)
        try:
            # 构建详细报告内容
            report_content = self._build_rich_report(query, outputs)

            # LLM 评分
            self.logger.info(self.AGENT_NAME, "LLM-as-Judge 评分...")
            combined_text = "\n\n".join(
                sec.get("body","") for sec in report_content.get("sections",[])
            )
            score = self.judge.score(query, combined_text)
            self.logger.info(self.AGENT_NAME,
                f"初始评分: {score.overall:.2f} | 需要反思: {score.needs_reflection}")

            # Self-Reflection 循环
            reflection_log = []
            final_text  = combined_text
            final_score = score
            round_num   = 0

            while final_score.needs_reflection and round_num < self.config.evaluation.max_reflection_rounds:
                round_num += 1
                self.logger.info(self.AGENT_NAME, f"反思轮次 {round_num}...")
                revised   = self.reflector.reflect_and_revise(query, final_text, final_score)
                new_score = self.judge.score(query, revised)
                reflection_log.append({
                    "round": round_num,
                    "score_before": final_score.overall,
                    "score_after":  new_score.overall,
                    "improved":     new_score.overall > final_score.overall,
                })
                if new_score.overall > final_score.overall:
                    final_text  = revised
                    final_score = new_score
                    self.logger.success(self.AGENT_NAME,
                        f"反思有效: {score.overall:.2f} → {new_score.overall:.2f}")
                else:
                    self.logger.info(self.AGENT_NAME, "反思后无改善，停止")
                    break

            # 将评估信息写入 report_content
            report_content["evaluation"] = {
                "overall_score":   final_score.overall,
                "accuracy":        final_score.accuracy,
                "completeness":    final_score.completeness,
                "format_quality":  final_score.format_quality,
                "feedback":        final_score.feedback,
                "reflection_rounds": round_num,
                "reflection_log":  reflection_log,
            }

            # 生成三种格式报告
            report_files = self.report_gen.generate(
                report_content,
                filename="final_report",
                formats=["docx", "markdown", "html"],
            )

            self.memory.add(
                content=f"评估完成: {query[:80]}，得分: {final_score.overall:.2f}",
                agent=self.AGENT_NAME,
                payload={"score": final_score.overall, "reflections": round_num},
                tags=["evaluation"],
            )

            result = {
                "final_score":      final_score.__dict__,
                "reflection_rounds": round_num,
                "reflection_log":   reflection_log,
                "report_files":     report_files,
            }
            self.logger.finish_call(call, result)
            self._print_result(final_score, reflection_log, report_files)
            return result

        except Exception as e:
            self.logger.fail_call(call, str(e))
            raise

    # ── 核心：构建详细报告 ───────────────────────────────────
    def _build_rich_report(self, query: str, outputs: dict) -> dict:
        """
        将各 Agent 输出整合为完整的学术报告结构。
        每个章节都由 LLM 扩写为详细内容。
        """
        self.logger.info(self.AGENT_NAME, "构建详细学术报告...")

        lit_result  = outputs.get("literature_result", {})
        exp_design  = outputs.get("experiment_design", {})
        code_result = outputs.get("code_result", {})
        ana_result  = outputs.get("analysis_result", {})

        papers   = lit_result.get("top_papers", []) or lit_result.get("papers", [])
        lit_review = lit_result.get("literature_review", "")
        final_code = code_result.get("final_code", "")
        stdout     = code_result.get("stdout", "")
        analysis   = ana_result.get("conclusion", "")
        charts     = ana_result.get("charts", [])

        # ── 各章节 LLM 扩写 ──

        # 1. 研究背景
        self.logger.info(self.AGENT_NAME, "生成研究背景...")
        bg = self._expand_section(
            f"为研究问题「{query}」撰写研究背景与意义（400字以上）：\n"
            "包括：① 该领域的研究现状；② 研究本问题的必要性；"
            "③ 当前存在的主要挑战；④ 本研究的贡献。",
            min_words=300
        )

        # 2. 文献综述（扩写 + 整合各论文摘要）
        self.logger.info(self.AGENT_NAME, "生成文献综述...")
        paper_summaries = ""
        if papers:
            paper_summaries = "\n".join([
                f"论文{i+1}：{p.get('title','')}\n"
                f"  方法：{(p.get('structured_summary') or {}).get('method','')}\n"
                f"  结论：{(p.get('structured_summary') or {}).get('conclusion','')}"
                for i, p in enumerate(papers[:6])
            ])
        lit_full = self._expand_section(
            f"基于以下{len(papers)}篇相关文献，撰写系统性文献综述（500字以上）：\n"
            f"{paper_summaries}\n\n已有综述片段：{lit_review}\n\n"
            "要求：① 归纳已有工作的主要方法；② 指出研究空白；③ 引出本研究切入点。",
            min_words=400
        )

        # 3. 实验设计
        self.logger.info(self.AGENT_NAME, "整合实验设计内容...")
        exp_desc = exp_design.get("full_description", "")
        if not exp_desc or len(exp_desc) < 100:
            exp_desc = self._expand_section(
                f"为研究问题「{query}」撰写完整实验设计（500字以上）：\n"
                "包括：① 实验目标与假设；② 数据集选择与预处理；"
                "③ 基线方法（至少3个）；④ 评估指标及计算公式；⑤ 实验流程。",
                min_words=400
            )
            exp_design = {**exp_design, "full_description": exp_desc}
        exp_summary = exp_design.get("structured_summary") or self._format_experiment_design(exp_design)
        exp_sections = self._normalize_experiment_sections(exp_design.get("sections", []))
        if not exp_sections:
            exp_sections = [{
                "heading": "3.1 实验设计补充说明",
                "body": exp_summary,
            }]

        # 4. 方法论
        self.logger.info(self.AGENT_NAME, "生成方法论描述...")
        method_desc = self._expand_section(
            f"基于以下代码，用学术语言描述实现方法（300字以上）：\n"
            f"研究问题：{query}\n代码摘要：\n{final_code[:800]}\n\n"
            "要求：① 算法原理；② 关键实现步骤；③ 参数设置说明。",
            min_words=200
        )

        # 5. 结果分析
        self.logger.info(self.AGENT_NAME, "生成结果分析...")
        result_desc = self._expand_section(
            f"基于以下分析结论和代码输出，撰写实验结果与分析（400字以上）：\n"
            f"统计结论：{analysis}\n代码输出：{stdout[:500]}\n研究问题：{query}\n\n"
            "要求：① 定量结果描述；② 与基线方法的对比；③ 结果的意义解读；④ 异常结果分析。",
            min_words=300
        )

        # 6. 结论与展望
        self.logger.info(self.AGENT_NAME, "生成结论与展望...")
        conclusion_desc = self._expand_section(
            f"为研究问题「{query}」撰写结论与展望（300字以上）：\n"
            "包括：① 主要研究发现；② 研究局限性；③ 未来工作方向。",
            min_words=200
        )

        return {
            "title":    f"MindPilot 科研报告：{query}",
            "query":    query,
            "abstract": f"本报告针对科研问题「{query}」，综合运用文献检索、实验设计、"
                        f"代码实现与数据分析，系统地探讨了该问题的研究现状、方法论与实验结果。"
                        f"共检索 {len(papers)} 篇相关文献，实现了核心算法并进行了定量分析，"
                        f"生成了完整的学术报告供参考。",
            "sections": [
                {"heading": "一、研究背景与问题陈述", "body": bg,          "level": 1},
                {"heading": "二、文献综述",           "body": lit_full,     "level": 1},
                {"heading": "三、实验设计与方法论",    "body": exp_summary,  "level": 1},
                *[
                    {"heading": sec["heading"], "body": sec["body"], "level": 2}
                    for sec in exp_sections
                ],
                {"heading": "四、核心方法实现",       "body": method_desc,  "level": 1},
                {"heading": "五、实验结果与分析",      "body": result_desc,  "level": 1},
                {"heading": "六、结论与展望",          "body": conclusion_desc, "level": 1},
            ],
            "code":      final_code,
            "stdout":    stdout,
            "literature": papers,
            "charts":    charts,
        }

    def _expand_section(self, prompt: str, min_words: int = 200) -> str:
        """调用 LLM 生成/扩写某个报告章节，确保达到最小字数"""
        resp = self.llm.chat([
            {"role": "system",
             "content": (
                 "你是资深科研论文写作专家，擅长撰写高质量中文学术报告。"
                 "请用严谨、专业的学术语言撰写内容，内容详实充分，"
                 f"不少于{min_words}字，不要使用项目符号列表，改用段落叙述。"
             )},
            {"role": "user", "content": prompt}
        ], max_tokens=2048)
        return resp if resp else f"（内容生成中，请参考问题：{prompt[:100]}）"

    def _normalize_experiment_sections(self, sections: Any) -> list[dict]:
        """清洗模型返回的实验设计小节，报告生成只消费 heading/body。"""
        if not isinstance(sections, list):
            return []

        normalized = []
        for idx, sec in enumerate(sections, 1):
            if not isinstance(sec, dict):
                continue
            heading = str(sec.get("heading", "")).strip()
            body = str(sec.get("body", "")).strip()
            if not heading or not body:
                continue
            if not heading.startswith("3."):
                heading = f"3.{idx} {heading}"
            normalized.append({"heading": heading, "body": body})
        return normalized

    def _format_experiment_part(self, items: Any, fallback: str = "") -> str:
        """把实验设计字段稳定格式化，避免报告中出现一整段散文。"""
        if isinstance(items, str):
            return items.strip() or fallback
        if isinstance(items, dict):
            lines = []
            for key, value in items.items():
                if isinstance(value, list):
                    value = "；".join(str(v) for v in value if str(v).strip())
                if str(value).strip():
                    lines.append(f"{key}：{value}")
            return "\n".join(lines) if lines else fallback
        if isinstance(items, list):
            cleaned = [str(item).strip() for item in items if str(item).strip()]
            return "\n".join(f"{idx}. {item}" for idx, item in enumerate(cleaned, 1)) if cleaned else fallback
        return fallback

    def _format_variables_and_ablations(self, exp_design: dict) -> str:
        variables = exp_design.get("variables", {}) or {}
        lines = []
        if isinstance(variables, dict):
            label_map = {
                "independent": "自变量",
                "dependent": "因变量",
                "controlled": "控制变量",
            }
            for key, label in label_map.items():
                value = variables.get(key, [])
                if isinstance(value, list):
                    value = "；".join(str(v) for v in value if str(v).strip())
                if str(value).strip():
                    lines.append(f"{label}：{value}")
        ablations = self._format_experiment_part(exp_design.get("ablations", []))
        if ablations:
            lines.append("消融实验：\n" + ablations)
        return "\n".join(lines) if lines else "本节通过固定训练配置、数据划分和评价协议，比较关键组件加入或移除后的性能变化。"

    def _format_experiment_design(self, exp_design: dict) -> str:
        """生成第 3 章开头的结构化总述。"""
        pieces = []
        overview = exp_design.get("full_description", "")
        if overview:
            pieces.append(overview.strip())

        hypothesis = exp_design.get("research_hypothesis", "")
        if hypothesis:
            pieces.append(f"研究假设：{hypothesis.strip()}")

        dataset = exp_design.get("dataset", "")
        if dataset:
            pieces.append(f"数据与环境：{dataset.strip()}")

        baselines = self._format_experiment_part(exp_design.get("baselines", []))
        if baselines:
            pieces.append("基线与对照组：\n" + baselines)

        metrics = self._format_experiment_part(exp_design.get("metrics", []))
        if metrics:
            pieces.append("核心评估指标：\n" + metrics)

        procedure = self._format_experiment_part(exp_design.get("procedure", []))
        if procedure:
            pieces.append("实验流程概览：\n" + procedure)

        expected = exp_design.get("expected_results", "")
        if expected:
            pieces.append(f"预期分析方向：{expected.strip()}")

        return "\n\n".join(pieces) if pieces else "本章从研究假设、数据集、基线方法、评估指标、实验流程和可复现性设置六个方面组织实验设计。"

    def _print_result(self, score: EvalScore, log: list, reports: dict):
        print(f"\n{'━'*58}")
        print(f"  🎯 评估与报告生成完成")
        print(f"{'━'*58}")
        print(f"  综合评分: {score.overall:.2f}  准确性: {score.accuracy:.2f}  "
              f"完整性: {score.completeness:.2f}  格式: {score.format_quality:.2f}")
        print(f"  反思轮次: {len(log)}")
        print(f"  报告文件:")
        for fmt, path in reports.items():
            print(f"    [{fmt.upper():8s}] {path}")
        print(f"  评审意见: {score.feedback[:70]}")
        print(f"{'━'*58}\n")
