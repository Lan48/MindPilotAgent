"""
模块④ — 数据分析与可视化 Agent
===================================
自然语言 → 分析指令转换 + 自动 EDA + 智能图表推荐 + 多格式报告。
"""

import json
import re
import math
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AnalysisResult:
    """数据分析结果"""
    task: str
    summary: dict                        # 基本统计
    statistical_tests: list[dict]        # 统计检验结果
    charts: list[str]                    # 生成的图表路径列表
    eda_report: str                      # EDA 报告文本
    conclusion: str                      # 分析结论
    report_files: dict = field(default_factory=dict)   # 多格式报告路径


class NLToAnalysis:
    """
    自然语言 → 数据分析指令转换器
    示例：「对比两组显著性差异」→ 自动选 t-test 或 Mann-Whitney
    """

    ANALYSIS_PATTERNS = {
        "distribution":      ["分布", "histogram", "正态", "distribution", "spread"],
        "comparison":        ["对比", "比较", "compare", "差异", "difference", "group"],
        "correlation":       ["相关", "correlation", "关系", "relationship", "associate"],
        "significance_test": ["显著", "significant", "p值", "p-value", "检验", "test"],
        "regression":        ["回归", "regression", "预测", "predict", "拟合", "fit"],
        "trend":             ["趋势", "trend", "时序", "time series", "变化"],
        "clustering":        ["聚类", "cluster", "分群", "grouping"],
        "eda":               ["探索", "EDA", "overview", "概览", "描述性"],
    }

    def parse(self, instruction: str) -> dict:
        """解析分析意图"""
        intent = "eda"
        inst_lower = instruction.lower()
        for analysis_type, keywords in self.ANALYSIS_PATTERNS.items():
            if any(k.lower() in inst_lower for k in keywords):
                intent = analysis_type
                break
        return {"intent": intent, "instruction": instruction}

    def select_test(self, intent: str, n_groups: int = 2, normal: bool = True) -> str:
        """选择合适的统计检验"""
        if intent == "significance_test":
            if n_groups == 2:
                return "t-test (独立样本)" if normal else "Mann-Whitney U"
            elif n_groups > 2:
                return "ANOVA" if normal else "Kruskal-Wallis"
        if intent == "correlation":
            return "Pearson" if normal else "Spearman"
        return "descriptive"


class AnalysisAgent:
    """
    模块④ — 数据分析与可视化 Agent
    """

    AGENT_NAME = "AnalysisAgent"

    def __init__(self, config, llm_client, visualizer, report_gen, memory_store, logger):
        self.config = config
        self.llm = llm_client
        self.visualizer = visualizer
        self.report_gen = report_gen
        self.memory = memory_store
        self.logger = logger
        self.nl_parser = NLToAnalysis()

    def run(self, task_description: str, data: Any = None, code_output: str = "") -> dict:
        """
        主入口
        data: 数值数据（list / dict / None），None 时使用模拟数据
        code_output: 代码 Agent 的输出（用于解析结果）
        """
        call = self.logger.start_call(self.AGENT_NAME, "data_analysis", task_description)

        try:
            import numpy as np
            import pandas as pd

            # Step 1: 解析分析意图
            parsed = self.nl_parser.parse(task_description)
            intent = parsed["intent"]
            self.logger.info(self.AGENT_NAME, f"分析意图识别: {intent}")

            # Step 2: 准备数据
            df = self._prepare_data(data, code_output, task_description)

            # Step 3: 自动 EDA
            eda = self._run_eda(df)

            # Step 4: 针对性统计分析
            stats_results = self._run_statistical_analysis(df, intent)

            # Step 5: 推荐图表类型并生成
            data_info = {
                "n_numeric": len(df.select_dtypes(include="number").columns),
                "n_categorical": len(df.select_dtypes(include="object").columns),
                "n_rows": len(df),
            }
            chart_type = self.visualizer.infer_chart_type(task_description, data_info)
            self.logger.info(self.AGENT_NAME, f"推荐图表类型: {chart_type}")

            charts = self._generate_charts(df, intent, chart_type, task_description)

            # Step 6: LLM 生成分析结论
            conclusion = self._generate_conclusion(task_description, eda, stats_results)

            # Step 7: 多格式报告
            report_content = {
                "title": f"数据分析报告：{task_description[:40]}",
                "query": task_description,
                "sections": [
                    {"heading": "描述性统计", "body": self._eda_to_text(eda)},
                    {"heading": "统计检验结果", "body": self._stats_to_text(stats_results)},
                    {"heading": "分析结论", "body": conclusion},
                ],
                "charts": [c.file_path for c in charts if c.file_path],
                "analysis": conclusion,
            }
            report_files = self.report_gen.generate(
                report_content,
                filename="analysis_report",
                formats=self.config.analysis.report_formats[:2]
            )

            # 存入记忆
            self.memory.add(
                content=f"数据分析: {task_description[:80]}，图表: {len(charts)} 张",
                agent=self.AGENT_NAME,
                payload={"intent": intent, "conclusion": conclusion[:200]},
                tags=["analysis"],
            )

            result = AnalysisResult(
                task=task_description,
                summary=eda,
                statistical_tests=stats_results,
                charts=[c.file_path for c in charts],
                eda_report=self._eda_to_text(eda),
                conclusion=conclusion,
                report_files=report_files,
            )

            out = result.__dict__.copy()
            self.logger.finish_call(call, {"intent": intent, "charts": len(charts)})
            self._print_result(result)
            return out

        except Exception as e:
            self.logger.fail_call(call, str(e))
            raise

    def _prepare_data(self, data, code_output: str, task: str):
        """准备 DataFrame"""
        import numpy as np
        import pandas as pd

        if data is not None:
            if isinstance(data, dict):
                return pd.DataFrame(data)
            if isinstance(data, list):
                return pd.DataFrame({"value": data})

        # 从代码输出中尝试解析数值
        if code_output:
            numbers = re.findall(r"[-+]?\d+\.?\d*", code_output)
            if len(numbers) >= 5:
                vals = [float(n) for n in numbers[:50]]
                return pd.DataFrame({
                    "value": vals[:len(vals)//2],
                    "predicted": vals[len(vals)//2:len(vals)//2 + len(vals)//2],
                } if len(vals) >= 10 else {"value": vals})

        # 生成符合任务的模拟数据
        np.random.seed(42)
        n = 80
        task_lower = task.lower()
        if "回归" in task_lower or "regression" in task_lower:
            x = np.linspace(0, 10, n)
            return pd.DataFrame({
                "x": x,
                "y": 2.5 * x + np.random.randn(n) * 2,
                "group": np.random.choice(["A", "B"], n),
            })
        elif "对比" in task_lower or "comparison" in task_lower:
            return pd.DataFrame({
                "group_A": np.random.normal(50, 10, n),
                "group_B": np.random.normal(55, 12, n),
            })
        else:
            return pd.DataFrame({
                "feature1": np.random.randn(n),
                "feature2": np.random.randn(n) * 2 + 1,
                "label": np.random.choice(["Cat1", "Cat2", "Cat3"], n),
                "score": np.random.uniform(0, 100, n),
            })

    def _run_eda(self, df) -> dict:
        """自动 EDA"""
        try:
            import numpy as np
            eda = {"shape": list(df.shape), "columns": list(df.columns)}
            num_df = df.select_dtypes(include="number")
            if not num_df.empty:
                desc = num_df.describe().round(3).to_dict()
                eda["descriptive"] = desc
                eda["missing"] = df.isnull().sum().to_dict()
                corr = num_df.corr().round(3).to_dict() if num_df.shape[1] >= 2 else {}
                eda["correlation"] = corr
            cat_df = df.select_dtypes(include="object")
            if not cat_df.empty:
                eda["categorical_counts"] = {
                    col: df[col].value_counts().to_dict()
                    for col in cat_df.columns[:3]
                }
            return eda
        except Exception:
            return {"shape": [0, 0], "error": "EDA 执行失败"}

    def _run_statistical_analysis(self, df, intent: str) -> list[dict]:
        """执行统计检验"""
        results = []
        try:
            from scipy import stats
            import numpy as np
            num_cols = df.select_dtypes(include="number").columns.tolist()

            if len(num_cols) >= 1:
                col = num_cols[0]
                data = df[col].dropna().values
                stat_n, p_norm = stats.shapiro(data[:50]) if len(data) >= 3 else (0, 1)
                results.append({
                    "test": "Shapiro-Wilk 正态性检验",
                    "column": col,
                    "statistic": round(float(stat_n), 4),
                    "p_value": round(float(p_norm), 4),
                    "conclusion": "正态分布" if p_norm > self.config.analysis.significance_level
                                  else "非正态分布",
                })

            if len(num_cols) >= 2 and intent in ("correlation", "regression"):
                col1, col2 = num_cols[0], num_cols[1]
                r, p_corr = stats.pearsonr(
                    df[col1].dropna().values[:80], df[col2].dropna().values[:80]
                )
                results.append({
                    "test": "Pearson 相关检验",
                    "columns": [col1, col2],
                    "r": round(float(r), 4),
                    "p_value": round(float(p_corr), 4),
                    "conclusion": f"{'显著' if p_corr < self.config.analysis.significance_level else '不显著'}相关 (r={r:.3f})",
                })

            if intent == "comparison" and len(num_cols) >= 2:
                g1 = df[num_cols[0]].dropna().values
                g2 = df[num_cols[1]].dropna().values
                t_stat, p_t = stats.ttest_ind(g1, g2)
                results.append({
                    "test": "独立样本 t 检验",
                    "groups": [num_cols[0], num_cols[1]],
                    "t_statistic": round(float(t_stat), 4),
                    "p_value": round(float(p_t), 4),
                    "conclusion": f"两组{'存在' if p_t < 0.05 else '不存在'}显著差异 (p={p_t:.4f})",
                })
        except ImportError:
            results.append({"test": "统计检验", "note": "scipy 未安装，跳过检验"})
        except Exception as e:
            results.append({"test": "统计检验", "error": str(e)[:100]})
        return results

    def _generate_charts(self, df, intent: str, chart_type: str, title: str) -> list:
        """生成图表"""
        charts = []
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()

        # 主图表
        if num_cols:
            data_for_chart = df[num_cols[0]].tolist()
            chart_title = f"{num_cols[0]} Distribution"
            x_label = num_cols[0]
            y_label = "Frequency"

            if chart_type == "scatter" and len(num_cols) >= 2:
                data_for_chart = {"x": df[num_cols[0]].tolist(),
                                  "y": df[num_cols[1]].tolist()}
                chart_title = f"{num_cols[1]} vs {num_cols[0]} Scatter"
                x_label = num_cols[0]
                y_label = num_cols[1]
            elif chart_type == "barplot":
                if cat_cols:
                    vc = df[cat_cols[0]].value_counts()
                    data_for_chart = dict(zip(vc.index.tolist(), vc.values.tolist()))
                    chart_title = f"{cat_cols[0]} Count"
                    x_label = cat_cols[0]
                    y_label = "Count"
                else:
                    compare_cols = num_cols[:4]
                    mean_values = {
                        col: round(float(df[col].dropna().mean()), 4)
                        for col in compare_cols
                    }
                    data_for_chart = mean_values
                    chart_title = "Group Mean Comparison"
                    x_label = "Metric Group"
                    y_label = "Mean Value"
            elif chart_type == "lineplot":
                chart_title = f"Trend of {num_cols[0]}"
                x_label = "Index"
                y_label = num_cols[0]
            elif chart_type == "histogram":
                chart_title = f"Distribution of {num_cols[0]}"
                x_label = num_cols[0]
                y_label = "Frequency"

            ch = self.visualizer.plot(chart_type, data_for_chart,
                                      title=chart_title,
                                      filename=f"main_{chart_type}",
                                      x_label=x_label,
                                      y_label=y_label)
            charts.append(ch)

            # 比较任务中补充分布图，避免仅看均值导致信息损失
            if intent == "comparison" and len(num_cols) >= 2:
                compare_cols = num_cols[:4]
                box_data = {
                    col: df[col].dropna().tolist()
                    for col in compare_cols
                }
                box = self.visualizer.plot(
                    "boxplot",
                    box_data,
                    title="Group Distribution Comparison",
                    filename="group_distribution_boxplot",
                    x_label="Metric Group",
                    y_label="Value"
                )
                charts.append(box)

        # 补充相关性热力图（若有多列数值）
        if len(num_cols) >= 2 and intent in ("correlation", "eda"):
            try:
                corr_cols = num_cols[:5]
                corr_matrix = df[corr_cols].corr().values.tolist()
                ch2 = self.visualizer.plot("heatmap", corr_matrix,
                                           title="Feature Correlation Matrix",
                                           filename="correlation_heatmap",
                                           x_label="Features",
                                           y_label="Features",
                                           x_tick_labels=corr_cols,
                                           y_tick_labels=corr_cols)
                charts.append(ch2)
            except Exception:
                pass
        return charts

    def _generate_conclusion(self, task: str, eda: dict, tests: list) -> str:
        """LLM 生成分析结论"""
        tests_text = "\n".join([
            f"- {t.get('test','')}: {t.get('conclusion', t.get('note', ''))}"
            for t in tests
        ])
        shape = eda.get("shape", [0, 0])
        system = "你是数据科学家。请根据分析结果写一段100字以内的中文分析结论。"
        prompt = (
            f"分析任务：{task}\n"
            f"数据规模：{shape[0]}行×{shape[1]}列\n"
            f"统计检验：\n{tests_text}"
        )
        resp = self.llm.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ])
        return resp[:400]

    def _eda_to_text(self, eda: dict) -> str:
        shape = eda.get("shape", [0, 0])
        lines = [f"数据集规模：{shape[0]} 行 × {shape[1]} 列"]
        missing = eda.get("missing", {})
        if any(v > 0 for v in missing.values()):
            lines.append(f"缺失值：{missing}")
        desc = eda.get("descriptive", {})
        for col, stats in desc.items():
            mean = stats.get("mean", "N/A")
            std = stats.get("std", "N/A")
            lines.append(f"- {col}: 均值={mean}, 标准差={std}")
        return "\n".join(lines)

    def _stats_to_text(self, tests: list) -> str:
        return "\n".join(
            f"- {t.get('test', '')}: {t.get('conclusion', t.get('note', t.get('error', '')))}"
            for t in tests
        )

    def _print_result(self, result: AnalysisResult):
        print(f"\n{'━'*58}")
        print(f"  📊 数据分析结果")
        print(f"{'━'*58}")
        print(f"  任务: {result.task[:55]}")
        print(f"  数据形状: {result.summary.get('shape', 'N/A')}")
        print(f"  统计检验: {len(result.statistical_tests)} 项")
        print(f"  图表: {len(result.charts)} 张")
        print(f"  报告格式: {list(result.report_files.keys())}")
        print(f"  结论: {result.conclusion[:80]}...")
        print(f"{'━'*58}\n")
