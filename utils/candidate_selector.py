"""候选特征粗筛与结构构建模块。

提供：
- CandidateFeature 结构体：携带 transform/band_name/wavelength 等元信息
- 相关性粗筛：仅在外层训练集上计算，测试集完全隔离
- 多光谱融合候选池构建
- 按真实波长构建 spectral windows 和 neighbors
- 结构检查函数（阶段 A 固定产物）

方法学说明：
    候选池基于外层训练集的 Pearson 相关系数构建，用于将搜索空间从 150 维压缩至
    120 维，以降低小样本下的优化复杂度。内层 5-fold fitness 评估在同一候选池上
    进行。未在每个内层 fold 内重复粗筛（近似处理，后续可扩展为 nested candidate
    generation）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from core.constants import WAVELENGTH_START, WAVELENGTH_STEP


@dataclass
class CandidateFeature:
    """多光谱融合候选特征的元信息。"""

    fusion_col: int    # 在 X_fusion 中的列索引（0-based）
    transform:  str    # 光谱变换类型："R" / "FD" / "SNV" / "LOG"
    band_idx:   int    # 原始波段在 feat_cols 中的位置（0-based）
    band_name:  str    # 列名，如 "b14"
    wavelength: float  # 真实波长（nm）= WAVELENGTH_START + band_number * WAVELENGTH_STEP


def _band_name_to_wavelength(band_name: str) -> float:
    """将波段列名（如 "b14"）转换为真实波长（nm）。

    使用 constants.py 中的 WAVELENGTH_START 和 WAVELENGTH_STEP。
    b1 对应 WAVELENGTH_START，b2 对应 WAVELENGTH_START + WAVELENGTH_STEP，以此类推。
    """
    band_number = int(band_name[1:])  # "b14" -> 14
    return WAVELENGTH_START + (band_number - 1) * WAVELENGTH_STEP


def select_top_correlated(X_train: np.ndarray, y_train: np.ndarray, n_top: int = 30) -> np.ndarray:
    """基于 Pearson 相关系数选 Top N 特征。

    仅在外层训练集上计算，测试集完全隔离。

    Args:
        X_train: 训练集特征矩阵，shape (n_train, n_features)
        y_train: 训练集目标向量，shape (n_train,)
        n_top:   保留的特征数量

    Returns:
        top_indices: 长度为 n_top 的索引数组（按相关性升序排列，最后 n_top 个最高）
    """
    n_features = X_train.shape[1]
    corr_scores = np.zeros(n_features)
    for j in range(n_features):
        # 处理常数列（std=0）
        if X_train[:, j].std() < 1e-10:
            corr_scores[j] = 0.0
        else:
            corr_scores[j] = abs(np.corrcoef(X_train[:, j], y_train)[0, 1])
    return np.argsort(corr_scores)[-n_top:]


def build_fusion_candidates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    transforms: Dict[str, np.ndarray],
    band_names: List[str],
    n_top: int = 30,
) -> Tuple[List[CandidateFeature], np.ndarray]:
    """对 4 种变换各选 Top n_top，构建多光谱融合候选池。

    Args:
        X_train:    原始训练集特征矩阵，shape (n_train, n_bands)
        y_train:    训练集目标向量，shape (n_train,)
        transforms: apply_all_transforms 的返回值，dict of {name: X_transformed}
        band_names: 原始波段列名列表，如 ["b14", "b15", ..., "b163"]
        n_top:      每种变换保留的特征数量

    Returns:
        candidates: List[CandidateFeature]，长度 = len(transforms) * n_top
        X_fusion:   np.ndarray，shape (n_train, len(transforms)*n_top)
    """
    candidates: List[CandidateFeature] = []
    X_fusion_list: List[np.ndarray] = []
    fusion_col = 0

    for transform_name, X_transformed in transforms.items():
        top_indices = select_top_correlated(X_transformed, y_train, n_top)
        for band_idx in top_indices:
            bname = band_names[band_idx]
            candidates.append(CandidateFeature(
                fusion_col=fusion_col,
                transform=transform_name,
                band_idx=int(band_idx),
                band_name=bname,
                wavelength=_band_name_to_wavelength(bname),
            ))
            fusion_col += 1
        X_fusion_list.append(X_transformed[:, top_indices])

    X_fusion = np.hstack(X_fusion_list)
    return candidates, X_fusion


def build_spectral_windows(
    candidates: List[CandidateFeature],
    window_nm: float = 20.0,
) -> List[List[int]]:
    """按真实波长构建 spectral windows。

    每个 window 包含波长落在同一 window_nm 区间内的所有候选特征（跨光谱表示）。
    不按 Fusion 列索引顺序切块，而是按真实波长分组。

    Args:
        candidates: CandidateFeature 列表
        window_nm:  窗口宽度（nm），默认 20nm

    Returns:
        windows: List[List[int]]，每个 window 是 fusion_col 索引列表（非空）
    """
    if not candidates:
        return []

    wavelengths = np.array([cf.wavelength for cf in candidates])
    wl_min = wavelengths.min()
    wl_max = wavelengths.max()

    windows: List[List[int]] = []
    wl = wl_min
    while wl <= wl_max:
        # 包含 [wl, wl + window_nm) 区间内的所有候选特征
        mask = (wavelengths >= wl) & (wavelengths < wl + window_nm)
        cols = [candidates[i].fusion_col for i in np.where(mask)[0]]
        if cols:
            windows.append(cols)
        wl += window_nm

    return windows


def build_spectral_neighbors(
    candidates: List[CandidateFeature],
    neighbor_nm: float = 10.0,
) -> Dict[int, List[int]]:
    """为每个候选特征构建真实光谱邻居列表。

    邻居定义：波长差 <= neighbor_nm 的其他候选特征（含不同光谱表示）。

    Args:
        candidates:  CandidateFeature 列表
        neighbor_nm: 邻居波长阈值（nm），默认 10nm

    Returns:
        neighbors: Dict[fusion_col -> List[neighbor fusion_col]]
    """
    wavelengths = np.array([cf.wavelength for cf in candidates])
    neighbors: Dict[int, List[int]] = {}

    for cf in candidates:
        wl = cf.wavelength
        mask = np.abs(wavelengths - wl) <= neighbor_nm
        mask[cf.fusion_col] = False  # 排除自身
        neighbors[cf.fusion_col] = [
            candidates[j].fusion_col for j in np.where(mask)[0]
        ]

    return neighbors


def print_candidate_summary(
    candidates: List[CandidateFeature],
    windows: List[List[int]],
    neighbors: Dict[int, List[int]],
    save_path: str,
) -> None:
    """打印结构检查摘要并保存 candidate_summary.csv。

    阶段 A 固定产物，必须在正式实验前执行一次。

    Args:
        candidates: CandidateFeature 列表
        windows:    build_spectral_windows 的返回值
        neighbors:  build_spectral_neighbors 的返回值
        save_path:  candidate_summary.csv 的保存路径（必填）
    """
    # ── 构建 window_id 映射 ──────────────────────────────
    col_to_window: Dict[int, int] = {}
    for w_id, window in enumerate(windows):
        for col in window:
            col_to_window[col] = w_id

    # ── 保存 CSV ─────────────────────────────────────────
    rows = []
    for cf in candidates:
        rows.append({
            "fusion_col":    cf.fusion_col,
            "transform":     cf.transform,
            "band_name":     cf.band_name,
            "wavelength":    cf.wavelength,
            "neighbor_count": len(neighbors.get(cf.fusion_col, [])),
            "window_id":     col_to_window.get(cf.fusion_col, -1),
        })
    df = pd.DataFrame(rows)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)

    # ── 终端摘要 ─────────────────────────────────────────
    n_candidates = len(candidates)
    n_windows    = len(windows)
    neighbor_counts = [len(neighbors.get(cf.fusion_col, [])) for cf in candidates]

    print(f"\n{'='*50}")
    print(f"候选特征总数: {n_candidates}")
    print(f"Spectral windows 数量: {n_windows}")
    print(f"Neighbors 数量分布: min={min(neighbor_counts)}, "
          f"mean={np.mean(neighbor_counts):.1f}, max={max(neighbor_counts)}")
    print(f"\nWindows 详情（按波长排序）:")
    for w_id, window in enumerate(windows):
        wls = [candidates[col].wavelength for col in window]
        transforms_in_window = list({candidates[col].transform for col in window})
        print(f"  Window {w_id:2d}: {min(wls):.0f}–{max(wls):.0f} nm, "
              f"{len(window)} 个特征, 变换类型: {sorted(transforms_in_window)}")
    print(f"\ncandidate_summary.csv 已保存至: {save_path}")
    print(f"{'='*50}\n")
