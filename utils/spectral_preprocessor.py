# utils/spectral_preprocessor.py

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


class SpectralPreprocessor:
    """
    高光谱数据预处理工具类。

    对指定波长范围内的波段执行 Savitzky-Golay 一阶导数处理：
    - 平滑去噪（window_length=9, polyorder=2）
    - 一阶导数（deriv=1），增强吸收峰特征
    - 范围外的波段保持原值不变

    Parameters
    ----------
    window_length : int
        S-G 滤波窗口长度（必须为奇数），默认 9
    polyorder : int
        S-G 多项式阶数，默认 2
    wl_start : int
        数据集第一个波段对应的波长（nm），默认 350
    wl_step : int
        相邻波段的波长步长（nm），默认 4
    filter_wl_min : int
        需要处理的波长下限（nm），默认 400
    filter_wl_max : int
        需要处理的波长上限（nm），默认 1000
    """

    def __init__(self,
                 window_length=9,
                 polyorder=2,
                 wl_start=350,
                 wl_step=4,
                 filter_wl_min=400,
                 filter_wl_max=1000):
        self.window_length = window_length
        self.polyorder = polyorder
        self.wl_start = wl_start
        self.wl_step = wl_step
        self.filter_wl_min = filter_wl_min
        self.filter_wl_max = filter_wl_max

    def _band_to_wavelength(self, band_name):
        """将波段列名（如 'b14'）转换为对应波长（nm）。"""
        idx = int(band_name[1:])  # 去掉前缀 'b'，取数字
        return self.wl_start + (idx - 1) * self.wl_step

    def get_filter_bands(self, band_cols):
        """
        从所有波段列中筛选出 filter_wl_min ~ filter_wl_max 范围内的列名。

        Parameters
        ----------
        band_cols : list of str
            所有波段列名，如 ['b1', 'b2', ..., 'b164']

        Returns
        -------
        list of str
            需要处理的波段列名
        """
        return [
            col for col in band_cols
            if self.filter_wl_min <= self._band_to_wavelength(col) <= self.filter_wl_max
        ]

    def apply(self, df, band_cols):
        """
        对 df 中的波段列执行 S-G 一阶导数处理。

        仅处理 filter_wl_min ~ filter_wl_max 范围内的波段，
        范围外的波段列保持原值，其他非波段列（id, Lon, Lat, TS 等）不变。

        Parameters
        ----------
        df : pd.DataFrame
            包含波段列的原始数据
        band_cols : list of str
            所有波段列名

        Returns
        -------
        pd.DataFrame
            处理后的数据（副本，不修改原始 df）
        """
        df_out = df.copy()

        # 筛选需要处理的波段
        target_cols = self.get_filter_bands(band_cols)

        # 对每个样本（行）做 S-G 一阶导数
        # savgol_filter 的 delta 参数保证导数单位为 反射率/nm
        spectra = df_out[target_cols].values  # shape: (n_samples, n_bands)
        spectra_fd = savgol_filter(
            spectra,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=1,
            delta=self.wl_step,
            axis=1,  # 沿波段方向（列方向）处理
        )

        df_out[target_cols] = spectra_fd
        return df_out
