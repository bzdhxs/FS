"""
光谱预处理脚本

对原始高光谱数据执行 S-G 一阶导数处理：
- 仅处理 400-1000nm 范围内的波段（b13~b163）
- 其他列（id, Lon, Lat, TS, EC 等）保持不变
- 生成处理后的 CSV 和对比可视化图

用法：
  python script/preprocess_spectra.py
"""

import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.spectral_preprocessor import SpectralPreprocessor

# =============================================================================
# 配置
# =============================================================================

INPUT_FILE  = PROJECT_ROOT / "resource" / "dataSet.csv"
OUTPUT_FILE = PROJECT_ROOT / "resource" / "dataSet_sg_fd.csv"
PLOT_FILE   = PROJECT_ROOT / "resource" / "spectral_comparison.png"

# 波段配置（与 constants.py 保持一致）
WL_START = 350
WL_STEP  = 4
BAND_START = 1
BAND_END   = 164  # 含

# S-G 参数
WINDOW_LENGTH  = 9
POLYORDER      = 2
FILTER_WL_MIN  = 400
FILTER_WL_MAX  = 1000

# 对比图随机抽取样本数
N_PLOT_SAMPLES = 5

# matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 主流程
# =============================================================================

def main():
    print(f"\n{'='*60}")
    print(f"  高光谱 S-G 一阶导数预处理")
    print(f"  窗口长度={WINDOW_LENGTH}  多项式阶数={POLYORDER}")
    print(f"  处理范围：{FILTER_WL_MIN}-{FILTER_WL_MAX}nm")
    print(f"{'='*60}\n")

    # 读取原始数据
    print(f"读取原始数据：{INPUT_FILE.name}")
    df = pd.read_csv(INPUT_FILE)
    print(f"  样本数：{len(df)}  列数：{len(df.columns)}")

    # 识别波段列
    band_cols = [f'b{i}' for i in range(BAND_START, BAND_END + 1)]
    band_cols = [c for c in band_cols if c in df.columns]

    # 初始化预处理器
    preprocessor = SpectralPreprocessor(
        window_length=WINDOW_LENGTH,
        polyorder=POLYORDER,
        wl_start=WL_START,
        wl_step=WL_STEP,
        filter_wl_min=FILTER_WL_MIN,
        filter_wl_max=FILTER_WL_MAX,
    )

    # 获取实际处理的波段
    target_cols = preprocessor.get_filter_bands(band_cols)
    print(f"  处理波段数：{len(target_cols)}（{target_cols[0]} ~ {target_cols[-1]}）")

    # 执行预处理
    print("执行 S-G 一阶导数处理...")
    df_processed = preprocessor.apply(df, band_cols)

    # 保存处理后的数据
    df_processed.to_csv(OUTPUT_FILE, index=False)
    print(f"  已保存：{OUTPUT_FILE.name}")

    # 验证输出
    assert len(df_processed) == len(df), "行数不一致"
    assert list(df_processed.columns) == list(df.columns), "列结构不一致"
    print(f"  验证通过：行数={len(df_processed)}，列数={len(df_processed.columns)}")

    # 绘制对比图
    print("绘制对比图...")
    _plot_comparison(df, df_processed, target_cols, band_cols)
    print(f"  已保存：{PLOT_FILE.name}")

    print(f"\n完成！处理后数据：{OUTPUT_FILE}\n")


def _plot_comparison(df_raw, df_proc, target_cols, band_cols):
    """绘制原始光谱 vs S-G 一阶导数光谱对比图（随机抽 N_PLOT_SAMPLES 个样本）。"""
    random.seed(42)
    sample_idx = random.sample(range(len(df_raw)), min(N_PLOT_SAMPLES, len(df_raw)))

    # 计算所有波段对应波长
    wavelengths = np.array([WL_START + (int(c[1:]) - 1) * WL_STEP for c in band_cols])
    target_wl   = np.array([WL_START + (int(c[1:]) - 1) * WL_STEP for c in target_cols])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']

    # 左图：原始光谱
    ax = axes[0]
    for i, idx in enumerate(sample_idx):
        y = df_raw.iloc[idx][band_cols].values.astype(float)
        ax.plot(wavelengths, y, color=colors[i], linewidth=1.2,
                label=f'样本 {idx}', alpha=0.85)
    ax.axvspan(FILTER_WL_MIN, FILTER_WL_MAX, alpha=0.08, color='gray', label='处理范围')
    ax.set_xlabel('波长 (nm)', fontsize=11)
    ax.set_ylabel('反射率', fontsize=11)
    ax.set_title('原始光谱', fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.35)

    # 右图：S-G 一阶导数
    ax = axes[1]
    for i, idx in enumerate(sample_idx):
        y = df_proc.iloc[idx][target_cols].values.astype(float)
        ax.plot(target_wl, y, color=colors[i], linewidth=1.2,
                label=f'样本 {idx}', alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('波长 (nm)', fontsize=11)
    ax.set_ylabel('一阶导数（反射率/nm）', fontsize=11)
    ax.set_title(f'S-G 一阶导数（window={WINDOW_LENGTH}, order={POLYORDER}）', fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.35)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
