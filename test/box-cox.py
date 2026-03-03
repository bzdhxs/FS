import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import pandas as pd

# 数据录入
data = [
    1.675, 1.65, 1.65, 2.125, 9, 6.45, 3.8, 12.875, 1.475, 2, 2.125, 4.475,
    3.875, 4.95, 10.2, 1.775, 1.625, 2.4, 1.7, 2.175, 8.125, 8.275, 1.9,
    1.675, 1.575, 4.35, 2.4, 7.575, 2.15, 1.45, 2.3, 8.875, 4.65, 6.05,
    8.875, 5.2, 4.625, 3.9, 1.325, 2.175, 2.175, 4.525, 3, 4.225, 2.1,
    3.975, 1.625, 4.825, 9.925, 8.2, 8.55, 3.475, 3.275, 3.325, 3.1, 5.475,
    8.725, 6.45, 5.775, 11.2, 2.15, 2.25, 6.825, 7.5, 4.125, 8.225, 8.375, 11.8
]
df = pd.DataFrame(data, columns=['TS'])

# 计算统计量
skew_orig = df['TS'].skew()

# 1. Log 变换
df['Log_TS'] = np.log(df['TS'])
skew_log = df['Log_TS'].skew()

# 2. Box-Cox 变换
df['BoxCox_TS'], best_lambda = stats.boxcox(df['TS'])
skew_boxcox = pd.Series(df['BoxCox_TS']).skew()

# 绘图
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 原始数据
sns.histplot(df['TS'], kde=True, ax=axes[0], color='skyblue')
axes[0].set_title(f'Original Data\nSkewness: {skew_orig:.2f} (Right Skewed)')

# Log 变换
sns.histplot(df['Log_TS'], kde=True, ax=axes[1], color='lightgreen')
axes[1].set_title(f'Log Transform\nSkewness: {skew_log:.2f}')

# Box-Cox 变换
sns.histplot(df['BoxCox_TS'], kde=True, ax=axes[2], color='salmon')
axes[2].set_title(f'Box-Cox Transform\nLambda: {best_lambda:.2f}, Skewness: {skew_boxcox:.2f}')

plt.tight_layout()
plt.show()