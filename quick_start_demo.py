#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FS_SSC 快速入门演示

此脚本演示如何使用重构后的 FS_SSC 系统。
运行此脚本可以快速了解系统的主要功能。
"""

import sys
import os

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def demo_1_list_available():
    """演示 1: 查看可用的算法和模型"""
    print("=" * 70)
    print("演示 1: 查看可用的算法和模型")
    print("=" * 70)

    import feature_selection
    import model
    from core.registry import list_algorithms, list_models

    algos = list_algorithms()
    models = list_models()

    print(f"\n可用算法 ({len(algos)} 个):")
    for i, algo in enumerate(sorted(algos), 1):
        print(f"  {i}. {algo}")

    print(f"\n可用模型 ({len(models)} 个):")
    for i, model_name in enumerate(sorted(models), 1):
        print(f"  {i}. {model_name}")

    print("\n✓ 所有算法和模型已自动注册")


def demo_2_check_algorithm_mode():
    """演示 2: 检查算法模式"""
    print("\n" + "=" * 70)
    print("演示 2: 检查算法模式")
    print("=" * 70)

    from core.registry import get_algorithm

    print("\n特征选择算法 (mode='selection'):")
    for algo in ['HHO', 'GA', 'GWO', 'MPA', 'CARS', 'SPA']:
        cls = get_algorithm(algo)
        print(f"  {algo}: mode='{cls.mode}'")

    print("\n特征提取算法 (mode='extraction'):")
    pca = get_algorithm('PCA')
    print(f"  PCA: mode='{pca.mode}'")

    print("\n✓ 系统会根据 mode 自动选择处理方式")


def demo_3_load_config():
    """演示 3: 加载配置文件"""
    print("\n" + "=" * 70)
    print("演示 3: 加载配置文件")
    print("=" * 70)

    from core.config import AppConfig

    if not os.path.exists('config.yaml'):
        print("\n✗ config.yaml 不存在")
        return

    cfg = AppConfig.from_yaml('config.yaml')

    print("\n当前配置:")
    print(f"  算法: {cfg.algo_name}")
    print(f"  模型: {cfg.model_name}")
    print(f"  目标列: {cfg.target_col}")
    print(f"  波段范围: {cfg.band_range[0]} - {cfg.band_range[1]}")
    print(f"  Box-Cox: {cfg.use_boxcox}")
    print(f"  测试集比例: {cfg.test_size}")
    print(f"  数据文件: {cfg.original_data_path}")

    print("\n✓ 配置加载成功")


def demo_4_check_constants():
    """演示 4: 查看全局常量"""
    print("\n" + "=" * 70)
    print("演示 4: 查看全局常量")
    print("=" * 70)

    from core.constants import (
        BINARY_THRESHOLD,
        FITNESS_PENALTY_DEFAULT,
        MAX_PLS_COMPONENTS,
        INTERNAL_VAL_SIZE,
        DEFAULT_RANDOM_STATE,
        BOXCOX_OFFSET,
        PLOT_DPI,
        PLOT_FONT
    )

    print("\n元启发式算法常量:")
    print(f"  二值化阈值: {BINARY_THRESHOLD}")
    print(f"  适应度惩罚值: {FITNESS_PENALTY_DEFAULT}")
    print(f"  PLS 最大主成分: {MAX_PLS_COMPONENTS}")
    print(f"  内部验证集比例: {INTERNAL_VAL_SIZE}")

    print("\n数据处理常量:")
    print(f"  随机种子: {DEFAULT_RANDOM_STATE}")
    print(f"  Box-Cox 偏移: {BOXCOX_OFFSET}")

    print("\n可视化常量:")
    print(f"  图表 DPI: {PLOT_DPI}")
    print(f"  字体: {PLOT_FONT}")

    print("\n✓ 所有常量集中管理，易于维护")


def demo_5_algorithm_parameters():
    """演示 5: 查看算法默认参数"""
    print("\n" + "=" * 70)
    print("演示 5: 查看算法默认参数")
    print("=" * 70)

    from core.registry import get_algorithm

    print("\n元启发式算法默认参数:")
    for algo in ['HHO', 'GA', 'GWO', 'MPA']:
        cls = get_algorithm(algo)
        print(f"\n  {algo}:")
        print(f"    epoch: {cls.default_epoch}")
        print(f"    pop_size: {cls.default_pop_size}")
        print(f"    penalty: {cls.default_penalty}")
        if hasattr(cls, 'default_pc'):
            print(f"    pc (交叉概率): {cls.default_pc}")
            print(f"    pm (变异概率): {cls.default_pm}")

    print("\n✓ 参数在算法文件中定义，可直接修改")


def demo_6_model_parameters():
    """演示 6: 查看模型默认参数"""
    print("\n" + "=" * 70)
    print("演示 6: 查看模型默认参数")
    print("=" * 70)

    from core.registry import get_model

    print("\n回归模型默认参数:")
    for model_name in ['PLS', 'RF', 'SVM']:
        cls = get_model(model_name)
        print(f"\n  {model_name}:")
        if hasattr(cls, 'default_n_trials'):
            print(f"    n_trials: {cls.default_n_trials}")
        if hasattr(cls, 'default_cv_folds'):
            print(f"    cv_folds: {cls.default_cv_folds}")

    print("\n✓ 模型参数可通过 config.yaml 覆盖")


def demo_7_create_custom_config():
    """演示 7: 创建自定义配置"""
    print("\n" + "=" * 70)
    print("演示 7: 创建自定义配置")
    print("=" * 70)

    import yaml

    # 创建一个示例配置
    custom_config = {
        'algo_name': 'GA',
        'model_name': 'SVM',
        'resource_dir': 'resource',
        'data_file': 'dataSet.csv',
        'target_col': 'TS',
        'band_start': 14,
        'band_end': 164,
        'use_boxcox': True,
        'test_size': 0.3,
        'show_plots': False,
        'base_log_dir': 'log',
        'algo_params': {
            'epoch': 300,
            'pop_size': 100,
            'penalty': 0.4
        },
        'model_params': {
            'n_trials': 500
        }
    }

    # 保存为示例文件
    example_file = 'config_example.yaml'
    with open(example_file, 'w', encoding='utf-8') as f:
        yaml.dump(custom_config, f, allow_unicode=True, default_flow_style=False)

    print(f"\n示例配置已创建: {example_file}")
    print("\n配置内容:")
    print(f"  算法: {custom_config['algo_name']}")
    print(f"  模型: {custom_config['model_name']}")
    print(f"  Box-Cox: {custom_config['use_boxcox']}")
    print(f"  算法参数覆盖: {custom_config['algo_params']}")
    print(f"  模型参数覆盖: {custom_config['model_params']}")

    print(f"\n使用方法:")
    print(f"  python main.py --config {example_file}")

    print("\n✓ 可以创建多个配置文件进行不同实验")


def demo_8_error_handling():
    """演示 8: 错误处理"""
    print("\n" + "=" * 70)
    print("演示 8: 错误处理")
    print("=" * 70)

    from core.registry import get_algorithm, get_model

    print("\n测试不存在的算法:")
    try:
        get_algorithm('NONEXISTENT')
    except KeyError as e:
        print(f"  ✓ 捕获到错误: {e}")

    print("\n测试不存在的模型:")
    try:
        get_model('NONEXISTENT')
    except KeyError as e:
        print(f"  ✓ 捕获到错误: {e}")

    print("\n✓ 错误消息清晰，列出可用选项")


def demo_9_file_structure():
    """演示 9: 项目文件结构"""
    print("\n" + "=" * 70)
    print("演示 9: 项目文件结构")
    print("=" * 70)

    print("\n核心模块:")
    for module in ['core', 'feature_selection', 'model', 'utils', 'visualizer']:
        if os.path.isdir(module):
            files = [f for f in os.listdir(module) if f.endswith('.py')]
            print(f"  {module}/ ({len(files)} 个文件)")

    print("\n配置和文档:")
    for file in ['config.yaml', 'requirements.txt', 'README.md', 'QUICK_REFERENCE.md']:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"  {file} ({size:,} bytes)")

    print("\n工具脚本:")
    for script in ['verify_refactoring.py', 'run_batch_experiments.py', 'analyze_results.py']:
        if os.path.exists(script):
            print(f"  {script}")

    print("\n✓ 清晰的模块化结构")


def demo_10_next_steps():
    """演示 10: 下一步操作"""
    print("\n" + "=" * 70)
    print("演示 10: 下一步操作")
    print("=" * 70)

    print("\n立即可做:")
    print("  1. 运行验证: python verify_refactoring.py")
    print("  2. 运行默认配置: python main.py")
    print("  3. 查看结果: ls log/HHO_RF_*/")
    print("  4. 阅读文档: README.md")

    print("\n实验对比:")
    print("  1. 创建多个配置文件 (config_hho.yaml, config_ga.yaml, ...)")
    print("  2. 批量运行: python run_batch_experiments.py")
    print("  3. 分析结果: python analyze_results.py")

    print("\n添加新算法:")
    print("  1. 创建文件: feature_selection/woa.py")
    print("  2. 继承基类: BaseMealpySelector")
    print("  3. 添加装饰器: @register_algorithm('WOA')")
    print("  4. 实现方法: create_optimizer()")
    print("  5. 修改配置: algo_name: 'WOA'")
    print("  6. 运行: python main.py")

    print("\n参数调整:")
    print("  方法1 (永久): 编辑 feature_selection/hho.py")
    print("  方法2 (临时): 在 config.yaml 添加 algo_params")
    print("  方法3 (全局): 编辑 core/constants.py")

    print("\n✓ 系统已就绪，可以开始使用！")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("FS_SSC 快速入门演示")
    print("=" * 70)
    print("\n此演示将展示重构后系统的主要功能。")
    print("每个演示都是独立的，可以单独运行。")

    demos = [
        ("查看可用的算法和模型", demo_1_list_available),
        ("检查算法模式", demo_2_check_algorithm_mode),
        ("加载配置文件", demo_3_load_config),
        ("查看全局常量", demo_4_check_constants),
        ("查看算法默认参数", demo_5_algorithm_parameters),
        ("查看模型默认参数", demo_6_model_parameters),
        ("创建自定义配置", demo_7_create_custom_config),
        ("错误处理", demo_8_error_handling),
        ("项目文件结构", demo_9_file_structure),
        ("下一步操作", demo_10_next_steps),
    ]

    try:
        for i, (name, demo_func) in enumerate(demos, 1):
            demo_func()

            if i < len(demos):
                input(f"\n按 Enter 继续下一个演示 ({i+1}/{len(demos)})...")

    except KeyboardInterrupt:
        print("\n\n演示已中断。")
        return
    except Exception as e:
        print(f"\n\n演示出错: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
    print("\n感谢您的观看。现在您可以:")
    print("  1. 运行 python verify_refactoring.py 验证系统")
    print("  2. 运行 python main.py 开始使用")
    print("  3. 查看 README.md 了解详细用法")
    print("\n祝您使用愉快！🎉")


if __name__ == "__main__":
    main()
