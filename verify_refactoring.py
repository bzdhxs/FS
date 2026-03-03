"""
FS_SSC 重构验证脚本

运行此脚本以验证重构后的系统是否正常工作。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试所有核心模块导入"""
    print("=" * 60)
    print("测试 1: 核心模块导入")
    print("=" * 60)

    try:
        from core.config import AppConfig
        from core.constants import BINARY_THRESHOLD, DEFAULT_RANDOM_STATE
        from core.registry import register_algorithm, get_algorithm, discover_plugins
        from core.logging_setup import setup_logger
        print("✓ core/ 模块导入成功")
        return True
    except Exception as e:
        print(f"✗ core/ 模块导入失败: {e}")
        return False


def test_plugin_discovery():
    """测试插件自动发现"""
    print("\n" + "=" * 60)
    print("测试 2: 插件自动发现")
    print("=" * 60)

    try:
        import feature_selection
        import model
        from core.registry import list_algorithms, list_models

        algos = list_algorithms()
        models = list_models()

        expected_algos = {'CARS', 'GA', 'GWO', 'HHO', 'MPA', 'PCA', 'SPA'}
        expected_models = {'PLS', 'RF', 'SVM'}

        if set(algos) == expected_algos:
            print(f"✓ 算法注册成功 ({len(algos)}个): {sorted(algos)}")
        else:
            print(f"✗ 算法注册不完整")
            print(f"  期望: {sorted(expected_algos)}")
            print(f"  实际: {sorted(algos)}")
            return False

        if set(models) == expected_models:
            print(f"✓ 模型注册成功 ({len(models)}个): {sorted(models)}")
        else:
            print(f"✗ 模型注册不完整")
            print(f"  期望: {sorted(expected_models)}")
            print(f"  实际: {sorted(models)}")
            return False

        return True
    except Exception as e:
        print(f"✗ 插件发现失败: {e}")
        return False


def test_algorithm_modes():
    """测试算法模式检测"""
    print("\n" + "=" * 60)
    print("测试 3: 算法模式检测")
    print("=" * 60)

    try:
        from core.registry import get_algorithm

        # 测试选择模式
        selection_algos = ['HHO', 'GA', 'GWO', 'MPA', 'CARS', 'SPA']
        for name in selection_algos:
            cls = get_algorithm(name)
            if cls.mode != 'selection':
                print(f"✗ {name} 模式错误: 期望 'selection', 实际 '{cls.mode}'")
                return False
        print(f"✓ 选择模式算法 ({len(selection_algos)}个): {selection_algos}")

        # 测试提取模式
        pca = get_algorithm('PCA')
        if pca.mode != 'extraction':
            print(f"✗ PCA 模式错误: 期望 'extraction', 实际 '{pca.mode}'")
            return False
        print(f"✓ 提取模式算法 (1个): ['PCA']")

        return True
    except Exception as e:
        print(f"✗ 模式检测失败: {e}")
        return False


def test_config_loading():
    """测试配置加载"""
    print("\n" + "=" * 60)
    print("测试 4: 配置加载")
    print("=" * 60)

    try:
        from core.config import AppConfig

        if not os.path.exists('config.yaml'):
            print("✗ config.yaml 不存在")
            return False

        cfg = AppConfig.from_yaml('config.yaml')

        print(f"✓ 配置加载成功")
        print(f"  - 算法: {cfg.algo_name}")
        print(f"  - 模型: {cfg.model_name}")
        print(f"  - 目标列: {cfg.target_col}")
        print(f"  - 波段范围: {cfg.band_range}")
        print(f"  - Box-Cox: {cfg.use_boxcox}")
        print(f"  - 测试集比例: {cfg.test_size}")

        return True
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        return False


def test_error_handling():
    """测试错误处理"""
    print("\n" + "=" * 60)
    print("测试 5: 错误处理")
    print("=" * 60)

    try:
        from core.registry import get_algorithm, get_model

        # 测试不存在的算法
        try:
            get_algorithm('NONEXISTENT')
            print("✗ 应该抛出 KeyError")
            return False
        except KeyError as e:
            if 'NONEXISTENT' in str(e) and 'Available algorithms' in str(e):
                print("✓ 算法错误处理正确")
            else:
                print(f"✗ 错误消息不完整: {e}")
                return False

        # 测试不存在的模型
        try:
            get_model('NONEXISTENT')
            print("✗ 应该抛出 KeyError")
            return False
        except KeyError as e:
            if 'NONEXISTENT' in str(e) and 'Available models' in str(e):
                print("✓ 模型错误处理正确")
            else:
                print(f"✗ 错误消息不完整: {e}")
                return False

        return True
    except Exception as e:
        print(f"✗ 错误处理测试失败: {e}")
        return False


def test_base_classes():
    """测试基类功能"""
    print("\n" + "=" * 60)
    print("测试 6: 基类功能")
    print("=" * 60)

    try:
        from feature_selection.base import BaseFeatureSelector, BaseMealpySelector
        from model.base import BaseModel, calc_metrics
        import numpy as np

        # 测试 calc_metrics
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        metrics = calc_metrics(y_true, y_pred, "Test")

        if 'R2' in metrics and 'RMSE' in metrics and 'RPD' in metrics:
            print(f"✓ calc_metrics 工作正常")
            print(f"  - R²: {metrics['R2']:.4f}")
            print(f"  - RMSE: {metrics['RMSE']:.4f}")
            print(f"  - RPD: {metrics['RPD']:.4f}")
        else:
            print("✗ calc_metrics 返回值不完整")
            return False

        # 测试基类存在
        print(f"✓ BaseFeatureSelector 存在")
        print(f"✓ BaseMealpySelector 存在")
        print(f"✓ BaseModel 存在")

        return True
    except Exception as e:
        print(f"✗ 基类测试失败: {e}")
        return False


def test_constants():
    """测试常量定义"""
    print("\n" + "=" * 60)
    print("测试 7: 常量定义")
    print("=" * 60)

    try:
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

        constants = {
            'BINARY_THRESHOLD': BINARY_THRESHOLD,
            'FITNESS_PENALTY_DEFAULT': FITNESS_PENALTY_DEFAULT,
            'MAX_PLS_COMPONENTS': MAX_PLS_COMPONENTS,
            'INTERNAL_VAL_SIZE': INTERNAL_VAL_SIZE,
            'DEFAULT_RANDOM_STATE': DEFAULT_RANDOM_STATE,
            'BOXCOX_OFFSET': BOXCOX_OFFSET,
            'PLOT_DPI': PLOT_DPI,
            'PLOT_FONT': PLOT_FONT
        }

        print("✓ 所有常量已定义:")
        for name, value in constants.items():
            print(f"  - {name}: {value}")

        return True
    except Exception as e:
        print(f"✗ 常量测试失败: {e}")
        return False


def test_file_structure():
    """测试文件结构"""
    print("\n" + "=" * 60)
    print("测试 8: 文件结构")
    print("=" * 60)

    required_files = [
        'main.py',
        'config.yaml',
        'requirements.txt',
        'README.md',
        'core/__init__.py',
        'core/config.py',
        'core/constants.py',
        'core/registry.py',
        'core/logging_setup.py',
        'feature_selection/__init__.py',
        'feature_selection/base.py',
        'feature_selection/hho.py',
        'feature_selection/ga.py',
        'feature_selection/gwo.py',
        'feature_selection/mpa.py',
        'feature_selection/cars.py',
        'feature_selection/spa.py',
        'feature_selection/pca.py',
        'model/__init__.py',
        'model/base.py',
        'model/plsr.py',
        'model/rf.py',
        'model/svm.py',
    ]

    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)

    if missing:
        print(f"✗ 缺少文件 ({len(missing)}个):")
        for file in missing:
            print(f"  - {file}")
        return False
    else:
        print(f"✓ 所有必需文件存在 ({len(required_files)}个)")
        return True


def test_old_files_removed():
    """测试旧文件是否已删除"""
    print("\n" + "=" * 60)
    print("测试 9: 旧文件清理")
    print("=" * 60)

    old_files = [
        'feature_selection/original_HHO.py',
        'feature_selection/original_GA.py',
        'feature_selection/original_GWO.py',
        'feature_selection/original_MPA.py',
        'feature_selection/original_CARS.py',
        'feature_selection/original_SPA.py',
        'feature_selection/original_PCA.py',
        'soil_salt_content_feature_selection_appliction.py'
    ]

    remaining = [f for f in old_files if os.path.exists(f)]

    if remaining:
        print(f"✗ 仍有旧文件存在 ({len(remaining)}个):")
        for file in remaining:
            print(f"  - {file}")
        return False
    else:
        print(f"✓ 所有旧文件已删除 ({len(old_files)}个)")
        return True


def test_data_file():
    """测试数据文件"""
    print("\n" + "=" * 60)
    print("测试 10: 数据文件")
    print("=" * 60)

    try:
        import pandas as pd

        data_path = 'resource/dataSet.csv'
        if not os.path.exists(data_path):
            print(f"✗ 数据文件不存在: {data_path}")
            return False

        df = pd.read_csv(data_path)
        print(f"✓ 数据文件加载成功")
        print(f"  - 样本数: {len(df)}")
        print(f"  - 列数: {len(df.columns)}")

        # 检查目标列
        if 'TS' not in df.columns:
            print("✗ 缺少目标列 'TS'")
            return False
        print(f"  - 目标列 'TS' 存在")

        # 检查波段列
        band_cols = [c for c in df.columns if c.startswith('b')]
        print(f"  - 波段列数: {len(band_cols)}")

        return True
    except Exception as e:
        print(f"✗ 数据文件测试失败: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("FS_SSC 重构验证测试")
    print("=" * 60)
    print()

    tests = [
        ("核心模块导入", test_imports),
        ("插件自动发现", test_plugin_discovery),
        ("算法模式检测", test_algorithm_modes),
        ("配置加载", test_config_loading),
        ("错误处理", test_error_handling),
        ("基类功能", test_base_classes),
        ("常量定义", test_constants),
        ("文件结构", test_file_structure),
        ("旧文件清理", test_old_files_removed),
        ("数据文件", test_data_file),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ 测试 '{name}' 异常: {e}")
            results.append((name, False))

    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {name}")

    print("\n" + "=" * 60)
    print(f"总计: {passed}/{total} 测试通过")

    if passed == total:
        print("🎉 所有测试通过！重构成功完成。")
        print("=" * 60)
        return True
    else:
        print(f"⚠️  {total - passed} 个测试失败，请检查上述错误。")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
