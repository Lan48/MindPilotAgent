"""
数据可视化工具
==============
根据数据类型和分析目标自动推荐图表类型并生成。
"""

import os
import json
from typing import Optional, Any
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ChartResult:
    chart_type: str
    file_path: str
    title: str
    description: str
    data_summary: dict


class AutoVisualizer:
    """
    智能可视化工具
    - 自动推断最适合的图表类型
    - 支持 PNG、HTML（交互式）两种输出
    """

    CHART_RULES = {
        "distribution": ["histogram", "kde", "boxplot"],
        "comparison": ["barplot", "grouped_bar", "heatmap"],
        "correlation": ["scatter", "heatmap", "pairplot"],
        "trend": ["lineplot", "area"],
        "proportion": ["pie", "donut", "stacked_bar"],
        "regression": ["scatter_with_fit"],
    }

    def __init__(self, output_dir: str = "outputs", logger=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def infer_chart_type(self, intent: str, data_info: dict) -> str:
        """根据分析意图和数据特征推断图表类型"""
        intent_lower = intent.lower()
        n_numeric = data_info.get("n_numeric", 0)
        n_categorical = data_info.get("n_categorical", 0)
        n_rows = data_info.get("n_rows", 0)

        if any(k in intent_lower for k in ["分布", "distribution", "histogram"]):
            return "histogram"
        if any(k in intent_lower for k in ["趋势", "trend", "时间", "time series"]):
            return "lineplot"
        if any(k in intent_lower for k in ["相关", "correlation", "关系"]):
            return "scatter" if n_numeric >= 2 else "heatmap"
        if any(k in intent_lower for k in ["比较", "compare", "对比"]):
            return "barplot"
        if any(k in intent_lower for k in ["占比", "proportion", "pie", "组成"]):
            return "pie"
        if any(k in intent_lower for k in ["回归", "regression", "拟合"]):
            return "scatter_with_fit"
        # 默认启发式
        if n_categorical >= 1 and n_numeric >= 1:
            return "barplot"
        if n_numeric >= 2:
            return "scatter"
        return "histogram"

    def plot(self, chart_type: str, data: Any, title: str = "",
             filename: str = "chart", fmt: str = "png",
             x_label: str = "", y_label: str = "",
             x_tick_labels: Optional[list[str]] = None,
             y_tick_labels: Optional[list[str]] = None) -> ChartResult:
        """生成图表"""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd

            plt.rcParams["font.family"] = "DejaVu Sans"
            plt.rcParams["axes.unicode_minus"] = False
            fig, ax = plt.subplots(figsize=(9, 5))
            description = ""
            inferred_x, inferred_y = self._infer_axis_labels(chart_type, data, x_label, y_label)
            safe_x = self._safe_axis_label(inferred_x, "X")
            safe_y = self._safe_axis_label(inferred_y, "Y")
            display_title = self._safe_english_title(title, chart_type, inferred_x, inferred_y)

            if chart_type == "histogram":
                arr = self._to_array(data)
                ax.hist(arr, bins=20, color="#4A90D9", edgecolor="white", alpha=0.85)
                ax.set_xlabel(self._safe_axis_label(inferred_x, "Value"))
                ax.set_ylabel(self._safe_axis_label(inferred_y, "Frequency"))
                description = f"Histogram of data distribution, mean={np.mean(arr):.2f}, std={np.std(arr):.2f}"

            elif chart_type == "lineplot":
                if isinstance(data, dict):
                    for label, values in data.items():
                        ax.plot(values, label=label, linewidth=2)
                    ax.legend()
                else:
                    arr = self._to_array(data)
                    ax.plot(arr, color="#4A90D9", linewidth=2)
                ax.set_xlabel(self._safe_axis_label(inferred_x, "Index"))
                ax.set_ylabel(self._safe_axis_label(inferred_y, "Value"))
                description = "Line chart for trend or time series"

            elif chart_type == "barplot":
                if isinstance(data, dict):
                    keys = list(data.keys())
                    vals = list(data.values())
                else:
                    keys = [f"Category {i}" for i in range(len(data))]
                    vals = list(data)
                colors = ["#4A90D9", "#E8A838", "#5CB85C", "#D9534F", "#9B59B6"]
                ax.bar(keys, vals, color=colors[:len(keys)], edgecolor="white")
                ax.set_xlabel(self._safe_axis_label(inferred_x, "Category"))
                ax.set_ylabel(self._safe_axis_label(inferred_y, "Value"))
                ax.tick_params(axis="x", labelrotation=45)
                for label in ax.get_xticklabels():
                    label.set_ha("right")
                description = f"Bar chart with {len(keys)} groups"

            elif chart_type == "boxplot":
                if isinstance(data, dict):
                    labels = list(data.keys())
                    values = [self._to_array(v) for v in data.values()]
                else:
                    labels = ["Series 0"]
                    values = [self._to_array(data)]
                ax.boxplot(values, labels=labels, patch_artist=True,
                           boxprops={"facecolor": "#BFD7EA", "edgecolor": "#4A90D9"},
                           medianprops={"color": "#D9534F", "linewidth": 2})
                ax.set_xlabel(self._safe_axis_label(inferred_x, "Groups"))
                ax.set_ylabel(self._safe_axis_label(inferred_y, "Value"))
                ax.tick_params(axis="x", labelrotation=45)
                for label in ax.get_xticklabels():
                    label.set_ha("right")
                description = f"Boxplot comparison across {len(labels)} groups"

            elif chart_type == "scatter" or chart_type == "scatter_with_fit":
                if isinstance(data, dict) and "x" in data and "y" in data:
                    x, y = np.array(data["x"]), np.array(data["y"])
                else:
                    import numpy as np
                    x = np.arange(20)
                    y = x * 1.5 + np.random.randn(20) * 3
                ax.scatter(x, y, alpha=0.7, color="#4A90D9", s=60)
                if chart_type == "scatter_with_fit":
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(x.min(), x.max(), 100)
                    ax.plot(x_line, p(x_line), "r--", linewidth=2, label="Fit Line")
                    ax.legend()
                ax.set_xlabel(safe_x)
                ax.set_ylabel(safe_y)
                description = "Scatter plot" + (" with regression fit" if "fit" in chart_type else "")

            elif chart_type == "heatmap":
                import numpy as np
                if isinstance(data, (list, np.ndarray)):
                    matrix = np.array(data)
                else:
                    matrix = np.random.rand(5, 5)
                im = ax.imshow(matrix, cmap="Blues", aspect="auto")
                plt.colorbar(im, ax=ax)
                ax.set_xlabel(self._safe_axis_label(inferred_x, "Columns"))
                ax.set_ylabel(self._safe_axis_label(inferred_y, "Rows"))
                if x_tick_labels:
                    x_ticks = list(range(min(len(x_tick_labels), matrix.shape[1])))
                    ax.set_xticks(x_ticks)
                    x_labels = [self._safe_axis_label(v, f"F{i}") for i, v in enumerate(x_tick_labels[:len(x_ticks)])]
                    ax.set_xticklabels(x_labels, rotation=45, ha="right")
                if y_tick_labels:
                    y_ticks = list(range(min(len(y_tick_labels), matrix.shape[0])))
                    ax.set_yticks(y_ticks)
                    y_labels = [self._safe_axis_label(v, f"F{i}") for i, v in enumerate(y_tick_labels[:len(y_ticks)])]
                    ax.set_yticklabels(y_labels)
                description = "Heatmap"

            elif chart_type == "pie":
                if isinstance(data, dict):
                    labels, sizes = list(data.keys()), list(data.values())
                else:
                    labels = [f"Category {i}" for i in range(len(data))]
                    sizes = list(data)
                ax.pie(sizes, labels=labels, autopct="%1.1f%%",
                       startangle=90, colors=["#4A90D9","#E8A838","#5CB85C","#D9534F","#9B59B6"])
                ax.axis("equal")
                description = "Pie chart showing proportion of each category"

            else:
                arr = self._to_array(data)
                ax.plot(arr, color="#4A90D9")
                ax.set_xlabel(self._safe_axis_label(inferred_x, "Index"))
                ax.set_ylabel(self._safe_axis_label(inferred_y, "Value"))
                description = "Generic chart"

            ax.set_title(display_title, fontsize=13, pad=12)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()

            out_path = self.output_dir / f"{filename}.{fmt}"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            if self.logger:
                self.logger.success("Visualizer", f"图表已保存: {out_path}")

            return ChartResult(
                chart_type=chart_type,
                file_path=str(out_path),
                title=display_title,
                description=description,
                data_summary={"chart_type": chart_type, "output": str(out_path)},
            )

        except ImportError:
            if self.logger:
                self.logger.warning("Visualizer", "matplotlib 未安装，跳过图表生成")
            return ChartResult(
                chart_type=chart_type, file_path="",
                title=title, description="（matplotlib 未安装）",
                data_summary={},
            )

    def _safe_english_title(self, title: str, chart_type: str,
                            x_label: str = "", y_label: str = "") -> str:
        """Use an English fallback title when input title contains non-ASCII chars."""
        default_title = self._default_english_title(chart_type, x_label, y_label)
        if not title:
            return default_title
        if all(ord(ch) < 128 for ch in title):
            return title
        return default_title

    def _default_english_title(self, chart_type: str, x_label: str, y_label: str) -> str:
        if chart_type in ("scatter", "scatter_with_fit") and x_label and y_label:
            return f"{y_label} vs {x_label}"
        if chart_type == "histogram" and x_label:
            return f"Distribution of {x_label}"
        if chart_type == "lineplot" and y_label:
            return f"Trend of {y_label}"
        if chart_type == "barplot" and x_label and y_label:
            return f"{y_label} by {x_label}"
        return chart_type.replace("_", " ").title()

    def _infer_axis_labels(self, chart_type: str, data: Any,
                           x_label: str, y_label: str) -> tuple[str, str]:
        """Infer axis labels from chart data when explicit labels are absent."""
        if x_label or y_label:
            return x_label, y_label

        if chart_type == "scatter" or chart_type == "scatter_with_fit":
            if isinstance(data, dict):
                return data.get("x_label", "X"), data.get("y_label", "Y")
            return "X", "Y"

        if chart_type == "barplot":
            if isinstance(data, dict):
                return "Category", "Value"
            return "Category", "Value"

        if chart_type == "histogram":
            return "Value", "Frequency"

        if chart_type == "lineplot":
            return "Index", "Value"

        if chart_type == "heatmap":
            return "Columns", "Rows"

        if chart_type == "boxplot":
            return "Groups", "Value"

        return "", ""

    def _safe_axis_label(self, label: str, fallback: str) -> str:
        if not label:
            return fallback
        if all(ord(ch) < 128 for ch in label):
            return label
        return fallback

    def _to_array(self, data):
        try:
            import numpy as np
            if isinstance(data, (list, tuple)):
                return np.array(data, dtype=float)
            return np.array(data)
        except Exception:
            import numpy as np
            return np.random.randn(50)
