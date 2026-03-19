"""Rich console utilities for FS_SSC.

提供全局 Console 实例和便捷的美化输出函数。
仅用于 main.py 中的特殊展示（banner、表格、分隔线），
普通日志通过 RichHandler 自动美化。
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# 全局 Console 实例
console = Console()


def print_banner(task_id: str, algo_name: str, model_name: str):
    """打印 pipeline 启动 banner。

    Args:
        task_id: 任务 ID（时间戳）
        algo_name: 算法名称
        model_name: 模型名称
    """
    title = Text("FS_SSC Pipeline", style="bold cyan")
    content = (
        f"Task:  {task_id}\n"
        f"Algo:  {algo_name}\n"
        f"Model: {model_name}"
    )
    console.print(Panel(content, title=title, border_style="cyan", expand=False))


def print_step_header(step_num: int, description: str):
    """打印步骤分隔标题。

    Args:
        step_num: 步骤编号
        description: 步骤描述
    """
    console.rule(f"[bold blue]Step {step_num}[/bold blue]  {description}", style="blue")


def print_report_table(train_metrics: dict, test_metrics: dict,
                       selection_time: float, modeling_time: float):
    """用 rich Table 打印最终报告。

    Args:
        train_metrics: 训练集指标字典
        test_metrics: 测试集指标字典
        selection_time: 特征选择耗时（秒）
        modeling_time: 建模耗时（秒）
    """
    # 指标表格
    table = Table(title="Final Report", show_header=True, header_style="bold magenta")
    table.add_column("Dataset", style="cyan", width=8)
    table.add_column("R2", justify="right")
    table.add_column("RMSE", justify="right")
    table.add_column("MAE", justify="right")
    table.add_column("RPD", justify="right")

    # 根据 R2 值着色
    def _color(val, threshold_good=0.9, threshold_ok=0.8):
        if val >= threshold_good:
            return "green"
        elif val >= threshold_ok:
            return "yellow"
        return "red"

    for label, m in [("Train", train_metrics), ("Test", test_metrics)]:
        r2_color = _color(m['R2'])
        table.add_row(
            label,
            f"[{r2_color}]{m['R2']:.4f}[/{r2_color}]",
            f"{m['RMSE']:.4f}",
            f"{m['MAE']:.4f}",
            f"{m['RPD']:.4f}",
        )

    console.print()
    console.print(table)
    console.print(
        f"  [dim]Time: Selection {selection_time:.2f}s | Modeling {modeling_time:.2f}s[/dim]"
    )
    console.print()
