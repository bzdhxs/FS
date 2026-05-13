"""
光谱变换脚本：对 train.csv 和 test.csv 分别进行 FOD 0.4阶、FOD 1.6阶、CR变换，
然后将三种变换后的波段拼接，保存为新的训练集和测试集。

变换方案：
  - FOD 0.4阶：低阶微分，保留整体形状，增强微弱吸收特征
  - FOD 1.6阶：内容一最优阶数，消除基线漂移，增强盐分敏感特征
  - CR（连续统去除）：标准化反射率形状，突出吸收谷深度

输出：
  - train_fused.csv：训练集
  - test_fused.csv：测试集

波段命名规则：
  - FOD0.4变换后的波段：fod04_b14, fod04_b15, ..., fod04_b163
  - FOD1.6变换后的波段：fod16_b14, fod16_b15, ..., fod16_b163
  - CR变换后的波段：    cr_b22, cr_b23, ..., cr_b155（CR后有效波段可能减少）
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def fractional_order_derivative(spectrum, order):
    """
    计算光谱的分数阶微分（Grünwald-Letnikov 定义）。

    Parameters
    ----------
    spectrum : np.ndarray, shape (n_samples, n_bands)
        光谱反射率矩阵
    order : float
        微分阶数（如 0.4, 1.6）

    Returns
    -------
    np.ndarray, shape (n_samples, n_bands)
        分数阶微分后的光谱
    """
    n_bands = spectrum.shape[1]
    # 计算 Grünwald-Letnikov 系数
    coeffs = np.zeros(n_bands)
    coeffs[0] = 1.0
    for k in range(1, n_bands):
        coeffs[k] = coeffs[k-1] * (1 - (order + 1) / k)

    # 对每个样本计算微分
    result = np.zeros_like(spectrum)
    for i in range(spectrum.shape[0]):
        for j in range(n_bands):
            val = 0.0
            for k in range(j + 1):
                val += coeffs[k] * spectrum[i, j - k]
            result[i, j] = val

    return result


def _compute_continuum(spectrum):
    """
    计算光谱的上包络线（连续统）。
    使用凸包上边界方法。
    """
    n = len(spectrum)
    maxima_indices = [0]
    for j in range(1, n - 1):
        if spectrum[j] >= spectrum[j-1] and spectrum[j] >= spectrum[j+1]:
            maxima_indices.append(j)
    maxima_indices.append(n - 1)
    maxima_indices = sorted(set(maxima_indices))

    continuum = np.zeros(n)
    for k in range(len(maxima_indices) - 1):
        i_start = maxima_indices[k]
        i_end = maxima_indices[k + 1]
        for j in range(i_start, i_end + 1):
            t = (j - i_start) / (i_end - i_start) if i_end != i_start else 0
            continuum[j] = spectrum[i_start] + t * (spectrum[i_end] - spectrum[i_start])

    return continuum


def continuum_removal(spectrum):
    """
    连续统去除（Continuum Removal）。

    Parameters
    ----------
    spectrum : np.ndarray, shape (n_samples, n_bands)
        光谱反射率矩阵（需要是正值）

    Returns
    -------
    np.ndarray, shape (n_samples, n_bands)
        CR变换后的光谱（值在0-1之间）
    """
    n_samples, n_bands = spectrum.shape
    result = np.zeros_like(spectrum)

    for i in range(n_samples):
        sp = spectrum[i]
        continuum = _compute_continuum(sp)
        with np.errstate(divide='ignore', invalid='ignore'):
            cr = np.where(continuum > 0, sp / continuum, 0.0)
        result[i] = cr

    return result


def transform_single(df, band_cols, cr_valid_mask=None):
    """
    对单个DataFrame进行三种光谱变换并拼接。

    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    band_cols : list of str
        有效波段列名
    cr_valid_mask : np.ndarray or None
        CR有效波段掩码（训练集计算，测试集复用）

    Returns
    -------
    tuple: (X_fused, fused_cols, cr_valid_mask)
    """
    X_raw = df[band_cols].values

    # FOD 0.4阶
    X_fod04 = fractional_order_derivative(X_raw, order=0.4)
    fod04_cols = [f'fod04_{col}' for col in band_cols]

    # FOD 1.6阶
    X_fod16 = fractional_order_derivative(X_raw, order=1.6)
    fod16_cols = [f'fod16_{col}' for col in band_cols]

    # CR变换
    X_smooth = np.zeros_like(X_raw)
    for i in range(X_raw.shape[0]):
        X_smooth[i] = savgol_filter(X_raw[i], window_length=9, polyorder=2, deriv=0)
    X_smooth = np.clip(X_smooth, 1e-6, None)
    X_cr = continuum_removal(X_smooth)

    # CR有效波段掩码：训练集计算，测试集复用
    if cr_valid_mask is None:
        cr_valid_mask = (np.var(X_cr, axis=0) > 1e-6) & np.all(X_cr < 2.0, axis=0)

    X_cr_valid = X_cr[:, cr_valid_mask]
    cr_cols = [f'cr_{band_cols[j]}' for j in range(len(band_cols)) if cr_valid_mask[j]]

    # 拼接
    X_fused = np.concatenate([X_fod04, X_fod16, X_cr_valid], axis=1)
    fused_cols = fod04_cols + fod16_cols + cr_cols

    return X_fused, fused_cols, cr_valid_mask


if __name__ == '__main__':
    resource_dir = r'D:\_code\FS\resource'
    target_col = 'TS'

    # 读取数据
    train_df = pd.read_csv(f'{resource_dir}\\train.csv')
    test_df = pd.read_csv(f'{resource_dir}\\test.csv')

    # 确定有效波段（去除全零列）
    all_band_cols = [f'b{i}' for i in range(14, 164)]  # b14-b163
    X_all = train_df[all_band_cols].values
    nonzero_mask = np.any(X_all > 0, axis=0)
    valid_band_cols = [all_band_cols[j] for j in range(len(all_band_cols)) if nonzero_mask[j]]
    print(f"Valid bands: {len(valid_band_cols)} (from {valid_band_cols[0]} to {valid_band_cols[-1]})")

    # 训练集变换
    print("\nProcessing train...")
    X_train_fused, fused_cols, cr_valid_mask = transform_single(train_df, valid_band_cols, cr_valid_mask=None)
    print(f"  FOD0.4: {len([c for c in fused_cols if c.startswith('fod04_')])} features")
    print(f"  FOD1.6: {len([c for c in fused_cols if c.startswith('fod16_')])} features")
    print(f"  CR:     {len([c for c in fused_cols if c.startswith('cr_')])} features")
    print(f"  Total fused: {len(fused_cols)} features")

    # 测试集变换（复用训练集的CR掩码）
    print("\nProcessing test...")
    X_test_fused, fused_cols_test, _ = transform_single(test_df, valid_band_cols, cr_valid_mask=cr_valid_mask)
    assert fused_cols == fused_cols_test, "Column mismatch between train and test!"

    # 构建输出DataFrame
    meta_cols = ['id', 'Lon', 'Lat', target_col]

    train_out = pd.concat([train_df[meta_cols].reset_index(drop=True),
                           pd.DataFrame(X_train_fused, columns=fused_cols)], axis=1)
    test_out = pd.concat([test_df[meta_cols].reset_index(drop=True),
                          pd.DataFrame(X_test_fused, columns=fused_cols)], axis=1)

    # 保存
    train_out.to_csv(f'{resource_dir}\\train_fused.csv', index=False)
    test_out.to_csv(f'{resource_dir}\\test_fused.csv', index=False)

    # 验证
    print(f"\nTrain output: {train_out.shape}")
    print(f"Test output:  {test_out.shape}")

    # 各变换与TS的相关性
    y_train = train_df[target_col].values
    for prefix in ['fod04_', 'fod16_', 'cr_']:
        cols = [c for c in fused_cols if c.startswith(prefix)]
        X = train_out[cols].values
        corrs = [np.abs(np.corrcoef(X[:, j], y_train)[0, 1]) for j in range(X.shape[1])]
        corrs = [c for c in corrs if not np.isnan(c)]
        if corrs:
            print(f"  {prefix} max|corr|: {max(corrs):.4f}, mean|corr|: {np.mean(corrs):.4f}")

    print(f"\nDone! Saved to:")
    print(f"  {resource_dir}\\train_fused.csv")
    print(f"  {resource_dir}\\test_fused.csv")
