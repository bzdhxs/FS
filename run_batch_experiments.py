"""
批量实验运行脚本

此脚本演示如何批量运行多个算法/模型组合进行对比实验。
"""

import os
import sys
import yaml
import subprocess
from datetime import datetime
from pathlib import Path

# 实验配置
EXPERIMENTS = [
    # 元启发式算法 + 不同模型
    {"algo": "HHO", "model": "PLS", "boxcox": False},
    {"algo": "HHO", "model": "RF", "boxcox": False},
    {"algo": "HHO", "model": "SVM", "boxcox": False},

    {"algo": "GA", "model": "PLS", "boxcox": False},
    {"algo": "GA", "model": "RF", "boxcox": False},
    {"algo": "GA", "model": "SVM", "boxcox": False},

    {"algo": "GWO", "model": "PLS", "boxcox": False},
    {"algo": "GWO", "model": "RF", "boxcox": False},

    {"algo": "MPA", "model": "PLS", "boxcox": False},
    {"algo": "MPA", "model": "RF", "boxcox": False},

    # 统计算法
    {"algo": "CARS", "model": "PLS", "boxcox": False},
    {"algo": "CARS", "model": "RF", "boxcox": False},

    {"algo": "SPA", "model": "PLS", "boxcox": False},
    {"algo": "SPA", "model": "RF", "boxcox": False},

    # PCA 提取
    {"algo": "PCA", "model": "PLS", "boxcox": False},
    {"algo": "PCA", "model": "RF", "boxcox": False},

    # Box-Cox 变换对比
    {"algo": "HHO", "model": "RF", "boxcox": True},
    {"algo": "GA", "model": "SVM", "boxcox": True},
]


def create_config(algo, model, boxcox, output_file):
    """创建实验配置文件"""
    config = {
        'algo_name': algo,
        'model_name': model,
        'resource_dir': 'resource',
        'data_file': 'dataSet.csv',
        'target_col': 'TS',
        'band_start': 14,
        'band_end': 164,
        'use_boxcox': boxcox,
        'test_size': 0.3,
        'show_plots': False,
        'base_log_dir': 'log',
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)

    return output_file


def run_experiment(config_file, exp_name):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"运行实验: {exp_name}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            ['python', 'main.py', '--config', config_file],
            capture_output=True,
            text=True,
            timeout=600  # 10分钟超时
        )

        if result.returncode == 0:
            print(f"✓ {exp_name} 完成")
            return True
        else:
            print(f"✗ {exp_name} 失败")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ {exp_name} 超时")
        return False
    except Exception as e:
        print(f"✗ {exp_name} 异常: {e}")
        return False


def collect_results():
    """收集所有实验结果"""
    import pandas as pd
    import glob

    results = []

    for metrics_file in glob.glob("log/*/model_metrics.csv"):
        try:
            df = pd.read_csv(metrics_file)
            test_metrics = df[df['Set'] == 'Test'].iloc[0]

            # 从路径提取实验名称
            exp_name = Path(metrics_file).parent.name
            algo_model = exp_name.split('_')[:2]

            results.append({
                'Experiment': exp_name,
                'Algorithm': algo_model[0] if len(algo_model) > 0 else 'Unknown',
                'Model': algo_model[1] if len(algo_model) > 1 else 'Unknown',
                'R2': test_metrics['R2'],
                'RMSE': test_metrics['RMSE'],
                'RPD': test_metrics['RPD']
            })
        except Exception as e:
            print(f"警告: 无法读取 {metrics_file}: {e}")

    if results:
        summary = pd.DataFrame(results)
        summary = summary.sort_values('R2', ascending=False)

        # 保存汇总结果
        output_file = f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        summary.to_csv(output_file, index=False)

        print(f"\n{'='*60}")
        print("实验结果汇总")
        print(f"{'='*60}")
        print(summary.to_string(index=False))
        print(f"\n结果已保存至: {output_file}")

        # 打印最佳组合
        best = summary.iloc[0]
        print(f"\n最佳组合: {best['Algorithm']} + {best['Model']}")
        print(f"  R² = {best['R2']:.4f}")
        print(f"  RMSE = {best['RMSE']:.4f}")
        print(f"  RPD = {best['RPD']:.4f}")
    else:
        print("未找到实验结果")


def main():
    """主函数"""
    print("="*60)
    print("FS_SSC 批量实验运行器")
    print("="*60)
    print(f"总实验数: {len(EXPERIMENTS)}")
    print()

    # 创建临时配置目录
    temp_dir = Path("temp_configs")
    temp_dir.mkdir(exist_ok=True)

    success_count = 0
    failed_experiments = []

    for i, exp in enumerate(EXPERIMENTS, 1):
        exp_name = f"{exp['algo']}_{exp['model']}"
        if exp['boxcox']:
            exp_name += "_BoxCox"

        print(f"\n[{i}/{len(EXPERIMENTS)}] {exp_name}")

        # 创建配置文件
        config_file = temp_dir / f"{exp_name}.yaml"
        create_config(exp['algo'], exp['model'], exp['boxcox'], config_file)

        # 运行实验
        if run_experiment(str(config_file), exp_name):
            success_count += 1
        else:
            failed_experiments.append(exp_name)

    # 打印总结
    print(f"\n{'='*60}")
    print("批量实验完成")
    print(f"{'='*60}")
    print(f"成功: {success_count}/{len(EXPERIMENTS)}")
    print(f"失败: {len(failed_experiments)}/{len(EXPERIMENTS)}")

    if failed_experiments:
        print("\n失败的实验:")
        for exp in failed_experiments:
            print(f"  - {exp}")

    # 收集结果
    print(f"\n{'='*60}")
    print("收集实验结果...")
    print(f"{'='*60}")
    collect_results()

    # 清理临时文件
    print(f"\n清理临时配置文件...")
    for config_file in temp_dir.glob("*.yaml"):
        config_file.unlink()
    temp_dir.rmdir()
    print("完成！")


if __name__ == "__main__":
    main()
