import os
import sys
import numpy as np
import pandas as pd

# 将当前目录加入路径，确保能导入 visualizer 模块
sys.path.append(os.getcwd())

from visualizer import model_visualizer


def main():
    print("🧪 正在生成测试数据...")

    # ==========================================
    # 1. 伪造数据 (模拟 TS 在 1-15 范围)
    # ==========================================
    np.random.seed(42)  # 固定种子，保证每次图一样

    # 模拟真实值 (1 到 15)
    y_train = np.linspace(1, 15, 50)  # 50个训练样本
    y_test = np.linspace(2, 14, 20)  # 20个测试样本

    # 模拟预测值 (真实值 + 高斯噪声)
    # 训练集噪声小一点 (模拟拟合得好)
    p_train = y_train + np.random.normal(0, 0.5, size=len(y_train))

    # 测试集噪声大一点 (模拟泛化误差)
    p_test = y_test + np.random.normal(0, 1.2, size=len(y_test))

    # ==========================================
    # 2. 构造评估指标字典
    # ==========================================
    # 这里我们随便写几个数字，仅用于测试显示格式
    metrics_train = {"R2": 0.95, "RMSE": 0.45}
    metrics_test = {"R2": 0.82, "RMSE": 1.15}

    # ==========================================
    # 3. 定义输出路径
    # ==========================================
    output_file = "test_prediction_plot.png"

    # 如果存在旧图先删除
    if os.path.exists(output_file):
        os.remove(output_file)

    # ==========================================
    # 4. 调用绘图函数
    # ==========================================
    print("🎨 正在调用 model_visualizer...")

    model_visualizer.plot_regression_results(
        y_train, p_train,
        y_test, p_test,
        metrics_train, metrics_test,
        save_path=output_file
    )

    print(f"✅ 测试成功！请查看根目录下的图片: {output_file}")


if __name__ == "__main__":
    main()