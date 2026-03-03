import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# 1. 你的原始数据
ts_data = [
    1.675, 1.65, 1.65, 2.125, 9, 6.45, 3.8, 12.875, 1.475, 2, 2.125, 4.475,
    3.875, 4.95, 10.2, 1.775, 1.625, 2.4, 1.7, 2.175, 8.125, 8.275, 1.9, 1.675,
    1.575, 4.35, 2.4, 7.575, 2.15, 1.45, 2.3, 8.875, 4.65, 6.05, 8.875, 5.2,
    4.625, 3.9, 1.325, 2.175, 2.175, 4.525, 3, 4.225, 2.1, 3.975, 1.625, 4.825,
    9.925, 8.2, 8.55, 3.475, 3.275, 3.325, 3.1, 5.475, 8.725, 6.45, 5.775, 11.2,
    2.15, 2.25, 6.825, 7.5, 4.125, 8.225, 8.375, 11.8
]

df = pd.DataFrame(ts_data, columns=['TS'])

# 2. 【核心步骤】创建分层标签 (Binning)
# n_bins=5 是针对 68 个样本的黄金数字
# duplicates='drop' 是为了防止大量重复数值导致分箱失败
df['bin_label'] = pd.qcut(df['TS'], q=5, labels=False, duplicates='drop')

print("--- 分桶情况统计 (确保每个桶样本数 > 5) ---")
print(df['bin_label'].value_counts().sort_index())
# 你应该会看到每个桶都有 13 或 14 个样本

# 3. 执行分层抽样
split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

for train_idx, test_idx in split.split(df, df['bin_label']):
    train_set = df.loc[train_idx]
    test_set = df.loc[test_idx]

# 4. 验证结果
print("\n--- 最终划分验证 ---")
print(f"训练集数量: {len(train_set)}")
print(f"测试集数量: {len(test_set)}")

print(f"\n训练集极值: Min={train_set['TS'].min():.3f}, Max={train_set['TS'].max():.3f}")
print(f"测试集极值: Min={test_set['TS'].min():.3f}, Max={test_set['TS'].max():.3f}")

# 检查高值样本的分配情况 (TS > 10)
high_val_train = train_set[train_set['TS'] > 10]
high_val_test = test_set[test_set['TS'] > 10]
print(f"\n>10 的高盐样本分配: 训练集有 {len(high_val_train)} 个, 测试集有 {len(high_val_test)} 个")