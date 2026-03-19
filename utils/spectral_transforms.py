"""光谱变换工具模块。

提供 4 种高光谱变换方法，用于多光谱融合候选池构建。
所有变换均保持输入 shape 不变：(n_samples, n_bands) -> (n_samples, n_bands)
"""

import numpy as np
from scipy.signal import savgol_filter


def apply_first_derivative(X: np.ndarray, delta: float = 1) -> np.ndarray:
    """S-G 一阶导数变换。

    Args:
        X:     输入光谱矩阵，shape (n_samples, n_bands)
        delta: 采样间隔（nm）。默认 1，假设等距采样。
               若已知波段间隔约为 N nm，可传入 delta=N。
               论文中说明：假设波段等距采样，delta=1。

    Returns:
        一阶导数光谱，shape 与输入相同
    """
    return savgol_filter(X, window_length=9, polyorder=2, deriv=1, delta=delta, axis=1)


def apply_snv(X: np.ndarray) -> np.ndarray:
    """标准正态变换（Standard Normal Variate）。

    按行标准化：每个样本减去自身均值，除以自身标准差。

    Args:
        X: 输入光谱矩阵，shape (n_samples, n_bands)

    Returns:
        SNV 变换后的光谱，shape 与输入相同
    """
    mean = X.mean(axis=1, keepdims=True)
    std  = X.std(axis=1, keepdims=True)
    return (X - mean) / (std + 1e-10)


def apply_log_reciprocal(X: np.ndarray) -> np.ndarray:
    """对数倒数变换 log10(1/R)。

    Args:
        X: 输入光谱矩阵，shape (n_samples, n_bands)，值应为反射率（>0）

    Returns:
        log10(1/R) 变换后的光谱，shape 与输入相同
    """
    return np.log10(1.0 / (X + 1e-10))


def apply_all_transforms(X: np.ndarray) -> dict:
    """对输入光谱应用全部 4 种变换。

    Args:
        X: 输入光谱矩阵，shape (n_samples, n_bands)

    Returns:
        dict，key 为变换名称，value 为变换后矩阵（shape 与输入相同）：
        {
            "R":   原始反射率,
            "FD":  S-G 一阶导数,
            "SNV": 标准正态变换,
            "LOG": log10(1/R),
        }
    """
    return {
        "R":   X,
        "FD":  apply_first_derivative(X),
        "SNV": apply_snv(X),
        "LOG": apply_log_reciprocal(X),
    }
