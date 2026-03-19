"""
结果分析和可视化脚本

此脚本用于分析和可视化批量实验的结果。
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob


def load_all_results():
    """加载所有实验结果"""
    results = []

    for metrics_file in glob.glob("log/*/model_metrics.csv"):
        try:
            df = pd.read_csv(metrics_file)

            # 获取训练集和测试集指标
            train_metrics = df[df['Set'] == 'Train'].iloc[0]
            test_metrics = df[df['Set'] == 'Test'].iloc[0]

            # 从路径提取实验信息
            exp_path = Path(metrics_file).parent.name
            parts = exp_path.split('_')

            algo = parts[0] if len(parts) > 0 else 'Unknown'
            model = parts[1] if len(parts) > 1 else 'Unknown'
            timestamp = '_'.join(parts[2:]) if len(parts) > 2 else ''

            results.append({
                'Experiment': exp_path,
                'Algorithm': algo,
                'Model': model,
                'Timestamp': timestamp,
                'Train_R2': train_metrics['R2'],
                'Train_RMSE': train_metrics['RMSE'],
                'Train_RPD': train_metrics['RPD'],
                'Test_R2': test_metrics['R2'],
                'Test_RMSE': test_metrics['RMSE'],
                'Test_RPD': test_metrics['RPD'],
                'Overfit': train_metrics['R2'] - test_metrics['R2']
            })
        except Exception as e:
            print(f"警告: 无法读取 {metrics_file}: {e}")

    if not results:
        print("未找到实验结果")
        return None

    return pd.DataFrame(results)


def plot_algorithm_comparison(df):
    """绘制算法对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 按算法分组的 R² 对比
    algo_r2 = df.groupby('Algorithm')['Test_R2'].agg(['mean', 'std', 'count'])
    algo_r2 = algo_r2.sort_values('mean', ascending=False)

    ax = axes[0, 0]
    x = np.arange(len(algo_r2))
    ax.bar(x, algo_r2['mean'], yerr=algo_r2['std'], capsize=5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(algo_r2.index, rotation=45)
    ax.set_ylabel('Test R²')
    ax.set_title('Algorithm Performance (R²)')
    ax.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for i, v in enumerate(algo_r2['mean']):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

    # 2. 按模型分组的 R² 对比
    model_r2 = df.groupby('Model')['Test_R2'].agg(['mean', 'std', 'count'])
    model_r2 = model_r2.sort_values('mean', ascending=False)

    ax = axes[0, 1]
    x = np.arange(len(model_r2))
    ax.bar(x, model_r2['mean'], yerr=model_r2['std'], capsize=5, alpha=0.7, color='orange')
    ax.set_xticks(x)
    ax.set_xticklabels(model_r2.index, rotation=45)
    ax.set_ylabel('Test R²')
    ax.set_title('Model Performance (R²)')
    ax.grid(axis='y', alpha=0.3)

    for i, v in enumerate(model_r2['mean']):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

    # 3. RMSE 对比
    algo_rmse = df.groupby('Algorithm')['Test_RMSE'].agg(['mean', 'std'])
    algo_rmse = algo_rmse.sort_values('mean')

    ax = axes[1, 0]
    x = np.arange(len(algo_rmse))
    ax.bar(x, algo_rmse['mean'], yerr=algo_rmse['std'], capsize=5, alpha=0.7, color='green')
    ax.set_xticks(x)
    ax.set_xticklabels(algo_rmse.index, rotation=45)
    ax.set_ylabel('Test RMSE')
    ax.set_title('Algorithm Performance (RMSE, lower is better)')
    ax.grid(axis='y', alpha=0.3)

    for i, v in enumerate(algo_rmse['mean']):
        ax.text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom')

    # 4. 过拟合分析
    overfit = df.groupby('Algorithm')['Overfit'].mean().sort_values(ascending=False)

    ax = axes[1, 1]
    x = np.arange(len(overfit))
    colors = ['red' if v > 0.3 else 'orange' if v > 0.15 else 'green' for v in overfit]
    ax.bar(x, overfit, alpha=0.7, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(overfit.index, rotation=45)
    ax.set_ylabel('Train R² - Test R²')
    ax.set_title('Overfitting Analysis (lower is better)')
    ax.axhline(y=0.15, color='orange', linestyle='--', alpha=0.5, label='Moderate')
    ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='High')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for i, v in enumerate(overfit):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('analysis_algorithm_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ 算法对比图已保存: analysis_algorithm_comparison.png")
    plt.close()


def plot_heatmap(df):
    """绘制算法-模型组合热力图"""
    # 创建透视表
    pivot = df.pivot_table(values='Test_R2', index='Algorithm', columns='Model', aggfunc='mean')

    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制热力图
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # 设置刻度
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)

    # 添加数值标签
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.values[i, j]
            if not np.isnan(value):
                text = ax.text(j, i, f'{value:.3f}',
                             ha="center", va="center", color="black", fontsize=10)

    ax.set_title('Algorithm-Model Combination Performance (Test R²)', fontsize=14, pad=20)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Algorithm', fontsize=12)

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Test R²', rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig('analysis_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ 热力图已保存: analysis_heatmap.png")
    plt.close()


def plot_scatter_analysis(df):
    """绘制散点分析图"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Train R² vs Test R² (检测过拟合)
    ax = axes[0]

    for algo in df['Algorithm'].unique():
        data = df[df['Algorithm'] == algo]
        ax.scatter(data['Train_R2'], data['Test_R2'], label=algo, alpha=0.7, s=100)

    # 添加理想线 (y=x)
    lim = [0, 1]
    ax.plot(lim, lim, 'k--', alpha=0.5, label='Ideal (no overfit)')

    ax.set_xlabel('Train R²', fontsize=12)
    ax.set_ylabel('Test R²', fontsize=12)
    ax.set_title('Overfitting Analysis', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # 2. Test R² vs Test RMSE
    ax = axes[1]

    for model in df['Model'].unique():
        data = df[df['Model'] == model]
        ax.scatter(data['Test_R2'], data['Test_RMSE'], label=model, alpha=0.7, s=100)

    ax.set_xlabel('Test R²', fontsize=12)
    ax.set_ylabel('Test RMSE', fontsize=12)
    ax.set_title('R² vs RMSE Trade-off', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('analysis_scatter.png', dpi=300, bbox_inches='tight')
    print("✓ 散点分析图已保存: analysis_scatter.png")
    plt.close()


def generate_report(df):
    """生成文本报告"""
    report = []
    report.append("="*60)
    report.append("FS_SSC 实验结果分析报告")
    report.append("="*60)
    report.append("")

    # 基本统计
    report.append(f"总实验数: {len(df)}")
    report.append(f"算法数: {df['Algorithm'].nunique()}")
    report.append(f"模型数: {df['Model'].nunique()}")
    report.append("")

    # 最佳结果
    report.append("-"*60)
    report.append("最佳结果 (按 Test R² 排序)")
    report.append("-"*60)

    top5 = df.nlargest(5, 'Test_R2')[['Algorithm', 'Model', 'Test_R2', 'Test_RMSE', 'Test_RPD']]
    report.append(top5.to_string(index=False))
    report.append("")

    # 算法排名
    report.append("-"*60)
    report.append("算法平均性能排名")
    report.append("-"*60)

    algo_stats = df.groupby('Algorithm').agg({
        'Test_R2': ['mean', 'std', 'max'],
        'Test_RMSE': ['mean', 'std', 'min']
    }).round(4)
    algo_stats = algo_stats.sort_values(('Test_R2', 'mean'), ascending=False)
    report.append(algo_stats.to_string())
    report.append("")

    # 模型排名
    report.append("-"*60)
    report.append("模型平均性能排名")
    report.append("-"*60)

    model_stats = df.groupby('Model').agg({
        'Test_R2': ['mean', 'std', 'max'],
        'Test_RMSE': ['mean', 'std', 'min']
    }).round(4)
    model_stats = model_stats.sort_values(('Test_R2', 'mean'), ascending=False)
    report.append(model_stats.to_string())
    report.append("")

    # 过拟合分析
    report.append("-"*60)
    report.append("过拟合分析 (Train R² - Test R²)")
    report.append("-"*60)

    overfit_stats = df.groupby('Algorithm')['Overfit'].agg(['mean', 'std', 'max']).round(4)
    overfit_stats = overfit_stats.sort_values('mean')
    report.append(overfit_stats.to_string())
    report.append("")

    # 稳定性分析
    report.append("-"*60)
    report.append("稳定性分析 (Test R² 标准差，越小越稳定)")
    report.append("-"*60)

    stability = df.groupby('Algorithm')['Test_R2'].std().sort_values()
    report.append(stability.to_string())
    report.append("")

    # 推荐组合
    report.append("-"*60)
    report.append("推荐组合")
    report.append("-"*60)

    best = df.loc[df['Test_R2'].idxmax()]
    report.append(f"最高精度: {best['Algorithm']} + {best['Model']}")
    report.append(f"  Test R² = {best['Test_R2']:.4f}")
    report.append(f"  Test RMSE = {best['Test_RMSE']:.4f}")
    report.append("")

    # 找到过拟合最小的
    best_stable = df.loc[df['Overfit'].idxmin()]
    report.append(f"最稳定 (最小过拟合): {best_stable['Algorithm']} + {best_stable['Model']}")
    report.append(f"  Overfit = {best_stable['Overfit']:.4f}")
    report.append(f"  Test R² = {best_stable['Test_R2']:.4f}")
    report.append("")

    report.append("="*60)

    # 保存报告
    report_text = '\n'.join(report)
    with open('analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(report_text)
    print("\n✓ 报告已保存: analysis_report.txt")


def main():
    """主函数"""
    print("="*60)
    print("FS_SSC 结果分析工具")
    print("="*60)
    print()

    # 加载结果
    print("加载实验结果...")
    df = load_all_results()

    if df is None or len(df) == 0:
        print("错误: 未找到实验结果")
        print("请先运行实验: python main.py")
        return

    print(f"✓ 加载了 {len(df)} 个实验结果")
    print()

    # 保存汇总数据
    df.to_csv('analysis_summary.csv', index=False)
    print("✓ 汇总数据已保存: analysis_summary.csv")
    print()

    # 生成可视化
    print("生成可视化...")
    plot_algorithm_comparison(df)
    plot_heatmap(df)
    plot_scatter_analysis(df)
    print()

    # 生成报告
    print("生成分析报告...")
    generate_report(df)
    print()

    print("="*60)
    print("分析完成！")
    print("="*60)
    print("\n生成的文件:")
    print("  - analysis_summary.csv (汇总数据)")
    print("  - analysis_algorithm_comparison.png (算法对比)")
    print("  - analysis_heatmap.png (热力图)")
    print("  - analysis_scatter.png (散点分析)")
    print("  - analysis_report.txt (文本报告)")


if __name__ == "__main__":
    main()
