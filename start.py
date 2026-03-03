#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FS_SSC 一键启动脚本

此脚本提供交互式菜单，方便快速使用各种功能。
"""

import os
import sys
import subprocess

def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """打印标题"""
    print("=" * 70)
    print("FS_SSC 2.0 - 一键启动菜单")
    print("=" * 70)
    print()

def print_menu():
    """打印菜单"""
    print("请选择操作:")
    print()
    print("【验证和演示】")
    print("  1. 运行系统验证 (verify_refactoring.py)")
    print("  2. 查看快速入门演示 (quick_start_demo.py)")
    print()
    print("【基本运行】")
    print("  3. 运行默认配置 (HHO + RF)")
    print("  4. 运行 PCA + PLS 组合")
    print("  5. 运行 GA + SVM 组合")
    print()
    print("【批量实验】")
    print("  6. 运行批量实验 (所有算法/模型组合)")
    print("  7. 分析实验结果")
    print()
    print("【查看结果】")
    print("  8. 列出所有运行结果")
    print("  9. 查看最新运行的日志")
    print("  10. 查看最新运行的指标")
    print()
    print("【文档和帮助】")
    print("  11. 打开 README.md")
    print("  12. 打开快速参考指南")
    print("  13. 查看可用算法和模型")
    print()
    print("  0. 退出")
    print()

def run_command(cmd, description):
    """运行命令"""
    print(f"\n{'='*70}")
    print(f"执行: {description}")
    print(f"{'='*70}\n")

    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"\n✓ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} 失败: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n\n✗ {description} 已中断")
        return False

def get_latest_log_dir():
    """获取最新的日志目录"""
    log_dir = "log"
    if not os.path.exists(log_dir):
        return None

    dirs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    if not dirs:
        return None

    # 按时间戳排序
    dirs.sort(reverse=True)
    return os.path.join(log_dir, dirs[0])

def main():
    """主函数"""
    while True:
        clear_screen()
        print_header()
        print_menu()

        choice = input("请输入选项 (0-13): ").strip()

        if choice == '0':
            print("\n再见！")
            break

        elif choice == '1':
            run_command("python verify_refactoring.py", "系统验证")
            input("\n按 Enter 继续...")

        elif choice == '2':
            run_command("python quick_start_demo.py", "快速入门演示")
            input("\n按 Enter 继续...")

        elif choice == '3':
            run_command("python main.py", "运行默认配置 (HHO + RF)")
            input("\n按 Enter 继续...")

        elif choice == '4':
            # 创建临时配置
            import yaml
            config = {
                'algo_name': 'PCA',
                'model_name': 'PLS',
                'resource_dir': 'resource',
                'data_file': 'dataSet.csv',
                'target_col': 'TS',
                'band_start': 14,
                'band_end': 164,
                'use_boxcox': False,
                'test_size': 0.3,
                'show_plots': False,
                'base_log_dir': 'log',
            }
            with open('_temp_pca_pls.yaml', 'w') as f:
                yaml.dump(config, f)

            run_command("python main.py --config _temp_pca_pls.yaml", "运行 PCA + PLS")

            # 清理临时文件
            if os.path.exists('_temp_pca_pls.yaml'):
                os.remove('_temp_pca_pls.yaml')

            input("\n按 Enter 继续...")

        elif choice == '5':
            # 创建临时配置
            import yaml
            config = {
                'algo_name': 'GA',
                'model_name': 'SVM',
                'resource_dir': 'resource',
                'data_file': 'dataSet.csv',
                'target_col': 'TS',
                'band_start': 14,
                'band_end': 164,
                'use_boxcox': False,
                'test_size': 0.3,
                'show_plots': False,
                'base_log_dir': 'log',
            }
            with open('_temp_ga_svm.yaml', 'w') as f:
                yaml.dump(config, f)

            run_command("python main.py --config _temp_ga_svm.yaml", "运行 GA + SVM")

            # 清理临时文件
            if os.path.exists('_temp_ga_svm.yaml'):
                os.remove('_temp_ga_svm.yaml')

            input("\n按 Enter 继续...")

        elif choice == '6':
            print("\n警告: 批量实验可能需要较长时间 (30分钟以上)")
            confirm = input("确认运行? (y/n): ").strip().lower()
            if confirm == 'y':
                run_command("python run_batch_experiments.py", "批量实验")
            input("\n按 Enter 继续...")

        elif choice == '7':
            run_command("python analyze_results.py", "分析实验结果")
            input("\n按 Enter 继续...")

        elif choice == '8':
            print(f"\n{'='*70}")
            print("所有运行结果:")
            print(f"{'='*70}\n")

            if os.path.exists("log"):
                dirs = [d for d in os.listdir("log") if os.path.isdir(os.path.join("log", d))]
                if dirs:
                    dirs.sort(reverse=True)
                    for i, d in enumerate(dirs[:10], 1):
                        print(f"  {i:2d}. {d}")
                    if len(dirs) > 10:
                        print(f"\n  ... 还有 {len(dirs) - 10} 个结果")
                else:
                    print("  (无运行结果)")
            else:
                print("  (log 目录不存在)")

            input("\n按 Enter 继续...")

        elif choice == '9':
            latest = get_latest_log_dir()
            if latest:
                log_file = os.path.join(latest, "run.log")
                if os.path.exists(log_file):
                    print(f"\n{'='*70}")
                    print(f"最新运行日志: {latest}")
                    print(f"{'='*70}\n")

                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        # 显示最后 30 行
                        for line in lines[-30:]:
                            print(line.rstrip())
                else:
                    print(f"\n日志文件不存在: {log_file}")
            else:
                print("\n未找到运行结果")

            input("\n按 Enter 继续...")

        elif choice == '10':
            latest = get_latest_log_dir()
            if latest:
                metrics_file = os.path.join(latest, "model_metrics.csv")
                if os.path.exists(metrics_file):
                    print(f"\n{'='*70}")
                    print(f"最新运行指标: {latest}")
                    print(f"{'='*70}\n")

                    with open(metrics_file, 'r', encoding='utf-8') as f:
                        print(f.read())
                else:
                    print(f"\n指标文件不存在: {metrics_file}")
            else:
                print("\n未找到运行结果")

            input("\n按 Enter 继续...")

        elif choice == '11':
            if os.path.exists("README.md"):
                if os.name == 'nt':  # Windows
                    os.system("start README.md")
                else:  # Linux/Mac
                    os.system("xdg-open README.md 2>/dev/null || open README.md")
                print("\n✓ 已打开 README.md")
            else:
                print("\n✗ README.md 不存在")
            input("\n按 Enter 继续...")

        elif choice == '12':
            if os.path.exists("QUICK_REFERENCE.md"):
                if os.name == 'nt':  # Windows
                    os.system("start QUICK_REFERENCE.md")
                else:  # Linux/Mac
                    os.system("xdg-open QUICK_REFERENCE.md 2>/dev/null || open QUICK_REFERENCE.md")
                print("\n✓ 已打开 QUICK_REFERENCE.md")
            else:
                print("\n✗ QUICK_REFERENCE.md 不存在")
            input("\n按 Enter 继续...")

        elif choice == '13':
            print(f"\n{'='*70}")
            print("可用算法和模型")
            print(f"{'='*70}\n")

            try:
                sys.path.insert(0, '.')
                import feature_selection
                import model
                from core.registry import list_algorithms, list_models, get_algorithm

                algos = list_algorithms()
                models = list_models()

                print(f"算法 ({len(algos)} 个):")
                for i, algo in enumerate(sorted(algos), 1):
                    cls = get_algorithm(algo)
                    print(f"  {i}. {algo:<8} (mode={cls.mode})")

                print(f"\n模型 ({len(models)} 个):")
                for i, model_name in enumerate(sorted(models), 1):
                    print(f"  {i}. {model_name}")

            except Exception as e:
                print(f"✗ 获取信息失败: {e}")

            input("\n按 Enter 继续...")

        else:
            print("\n✗ 无效选项，请重新输入")
            input("\n按 Enter 继续...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已退出")
    except Exception as e:
        print(f"\n\n程序异常: {e}")
        import traceback
        traceback.print_exc()
