import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

logger = logging.getLogger(__name__)


def regression_stratified_split(X, y, test_size=0.3, n_bins=5, random_state=42):
    """
    针对回归问题的分层抽样工具函数 (Stratified Split for Regression)。
    特别针对小样本 (N < 100) 进行了优化，使用等频分箱 (qcut) 保证稳定性。

    参数:
    ----------
    X : pd.DataFrame 或 np.ndarray
        特征矩阵
    y : pd.Series 或 np.ndarray
        目标变量 (连续值)
    test_size : float
        测试集比例 (默认 0.3)
    n_bins : int
        分层数量。对于 N=68 的数据，强烈建议设为 5。
    random_state : int
        随机种子

    返回:
    ----------
    X_train, X_test, y_train, y_test
    (保留输入的数据类型，如果是 DataFrame 则返回 DataFrame)
    """

    # 1. 确保 y 是 Pandas Series 格式，方便使用 qcut
    if isinstance(y, np.ndarray):
        y_series = pd.Series(y)
    else:
        y_series = y.copy()

    # 2. 生成分层标签 (Binning)
    # 优先使用 qcut (等频分箱)，保证每个桶里的样本数大致相同
    # duplicates='drop' 用于处理大量重复值导致分箱边界重合的情况
    try:
        bins = pd.qcut(y_series, q=n_bins, labels=False, duplicates='drop')
    except ValueError:
        logger.warning(f"[DataSplit] Cannot perform {n_bins}-bin quantile split, reducing bins...")
        # 降级策略：如果 5 层分不了，就试着分 3 层
        bins = pd.qcut(y_series, q=max(2, n_bins // 2), labels=False, duplicates='drop')

    # 3. 安全检查：每个桶里的样本是否足够？
    # 理论上，如果 test_size=0.3，每个桶至少要有 2 个样本才能保证 train/test 都有人
    min_samples = bins.value_counts().min()

    if min_samples < 2:
        logger.warning(f"[DataSplit] Insufficient samples in a bin ({min_samples}). Falling back to random split.")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    # 4. 执行分层抽样
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    # StratifiedShuffleSplit 返回的是索引 (indices)
    for train_idx, test_idx in split.split(X, bins):
        # 5. 根据输入类型切分数据
        if isinstance(X, pd.DataFrame):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
        else:
            X_train = X[train_idx]
            X_test = X[test_idx]

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
        else:
            y_train = y[train_idx]
            y_test = y[test_idx]

    return X_train, X_test, y_train, y_test