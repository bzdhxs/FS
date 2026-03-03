# FS_SSC 快速参考指南

## 🚀 常用命令

### 基本运行
```bash
# 使用默认配置 (HHO + RF)
python main.py

# 使用自定义配置
python main.py --config experiments/exp1.yaml

# 查看可用算法和模型
python -c "import sys; sys.path.insert(0, '.'); import feature_selection, model; from core.registry import list_algorithms, list_models; print('算法:', list_algorithms()); print('模型:', list_models())"
```

### 快速实验配置

创建不同的配置文件进行实验：

```bash
# experiments/hho_rf.yaml
algo_name: "HHO"
model_name: "RF"
use_boxcox: false

# experiments/ga_svm.yaml
algo_name: "GA"
model_name: "SVM"
use_boxcox: true

# experiments/pca_pls.yaml
algo_name: "PCA"
model_name: "PLS"
use_boxcox: false
```

然后运行：
```bash
python main.py --config experiments/hho_rf.yaml
python main.py --config experiments/ga_svm.yaml
python main.py --config experiments/pca_pls.yaml
```

---

## 📋 算法参数速查表

### 元启发式算法

| 算法 | 文件 | 默认参数 | 调整建议 |
|------|------|----------|----------|
| **HHO** | `feature_selection/hho.py` | epoch=200, pop_size=250, penalty=0.2 | 增加 epoch 提高精度 |
| **GA** | `feature_selection/ga.py` | epoch=200, pop_size=50, pc=0.85, pm=0.05, penalty=0.6 | 调整 pc/pm 控制交叉/变异 |
| **GWO** | `feature_selection/gwo.py` | epoch=200, pop_size=100, penalty=0.2 | 增加 pop_size 提高多样性 |
| **MPA** | `feature_selection/mpa.py` | epoch=250, pop_size=120, penalty=0.4 | penalty 控制特征数量 |

### 统计算法

| 算法 | 文件 | 默认参数 | 调整建议 |
|------|------|----------|----------|
| **CARS** | `feature_selection/cars.py` | n_iter=200, k_fold=5, n_components=5 | 增加 n_iter 提高稳定性 |
| **SPA** | `feature_selection/spa.py` | m_min=2, m_max=30 | 调整 m_max 控制最大特征数 |
| **PCA** | `feature_selection/pca.py` | n_components=3 | 根据方差贡献率调整 |

### 回归模型

| 模型 | 文件 | 默认参数 | 调整建议 |
|------|------|----------|----------|
| **PLS** | `model/plsr.py` | n_iter=100, cv_folds=5 | 增加 n_iter 扩大搜索空间 |
| **RF** | `model/rf.py` | n_trials=200, cv_folds=5 | 增加 n_trials 提高优化质量 |
| **SVM** | `model/svm.py` | n_trials=300, cv_folds=5 | SVM 对参数敏感，多试几次 |

---

## 🎯 参数调整方法

### 方法 1: 永久修改（推荐用于确定的改进）

直接编辑算法文件：

```python
# feature_selection/hho.py
@register_algorithm("HHO")
class HHOSelector(BaseMealpySelector):
    default_epoch = 500        # 从 200 改为 500
    default_pop_size = 300     # 从 250 改为 300
    default_penalty = 0.15     # 从 0.2 改为 0.15
```

### 方法 2: 临时覆盖（推荐用于实验）

在 `config.yaml` 中添加：

```yaml
algo_name: "HHO"
model_name: "RF"

# 覆盖算法参数
algo_params:
  epoch: 500
  pop_size: 300
  penalty: 0.15

# 覆盖模型参数
model_params:
  n_trials: 500
  cv_folds: 10
```

### 方法 3: 修改全局常量

编辑 `core/constants.py`：

```python
# 影响所有元启发式算法
BINARY_THRESHOLD = 0.5        # 二值化阈值
FITNESS_PENALTY_DEFAULT = 99999.0  # 无特征惩罚值
MAX_PLS_COMPONENTS = 5        # PLS 最大主成分数
INTERNAL_VAL_SIZE = 0.3       # 内部验证集比例
DEFAULT_RANDOM_STATE = 42     # 随机种子
```

---

## 🔧 常见调参场景

### 场景 1: 选中的特征太多

**问题**: HHO 选了 50+ 个特征，想减少到 10-20 个

**解决方案**: 增加 penalty 参数

```yaml
algo_params:
  penalty: 0.5  # 从默认 0.2 增加到 0.5
```

### 场景 2: 选中的特征太少

**问题**: GA 只选了 3 个特征，想增加到 15-25 个

**解决方案**: 减少 penalty 参数

```yaml
algo_params:
  penalty: 0.3  # 从默认 0.6 减少到 0.3
```

### 场景 3: 算法收敛不稳定

**问题**: 每次运行结果差异很大

**解决方案**: 增加迭代次数和种群大小

```yaml
algo_params:
  epoch: 500      # 增加迭代次数
  pop_size: 300   # 增加种群大小
```

### 场景 4: 模型过拟合

**问题**: 训练集 R²=0.95，测试集 R²=0.30

**解决方案**:
1. 启用 Box-Cox 变换
2. 增加交叉验证折数
3. 减少特征数量

```yaml
use_boxcox: true

model_params:
  cv_folds: 10  # 从 5 增加到 10

algo_params:
  penalty: 0.4  # 增加惩罚，减少特征
```

### 场景 5: 计算时间太长

**问题**: HHO 运行超过 10 分钟

**解决方案**: 减少迭代次数或种群大小

```yaml
algo_params:
  epoch: 100      # 从 200 减少到 100
  pop_size: 150   # 从 250 减少到 150
```

或者换用更快的算法：

```yaml
algo_name: "SPA"  # SPA 通常比元启发式算法快
```

---

## 📊 输出文件说明

每次运行在 `log/` 下创建时间戳目录：

```
log/HHO_RF_20260228_143052/
├── train_data.csv                 # 训练集（分层抽样后）
├── test_data.csv                  # 测试集（分层抽样后）
├── selected_features-HHO.csv      # 选中的特征（仅特征选择模式）
├── train_data_pca.csv             # PCA 训练集（仅 PCA 模式）
├── test_data_pca.csv              # PCA 测试集（仅 PCA 模式）
├── prediction_results.csv         # 预测结果（Measured vs Predicted）
├── model_metrics.csv              # 模型指标（R², RMSE, RPD）
├── plot_pred_vs_meas.png         # 预测vs实测散点图
├── HHO_selection_plot.png        # 特征选择可视化（仅特征选择模式）
├── lambda_value.txt               # Box-Cox lambda 值（如启用）
└── run.log                        # 完整运行日志
```

### 关键文件解读

**model_metrics.csv**:
```csv
Set,R2,RMSE,RPD,Best_Params
Train,0.8825,1.0067,2.92,"{'n_estimators': 17, 'max_depth': 3, ...}"
Test,0.2536,2.7151,1.16,"{'n_estimators': 17, 'max_depth': 3, ...}"
```

**prediction_results.csv**:
```csv
Measured,Predicted
5.23,5.45
7.89,7.12
...
```

---

## 🐛 故障排除

### 问题 1: ModuleNotFoundError

**错误**: `ModuleNotFoundError: No module named 'mealpy'`

**解决**:
```bash
pip install -r requirements.txt
```

### 问题 2: 算法未找到

**错误**: `KeyError: "Algorithm 'WOA' not found"`

**原因**: 算法文件未创建或未注册

**检查**:
```python
python -c "import sys; sys.path.insert(0, '.'); import feature_selection; from core.registry import list_algorithms; print(list_algorithms())"
```

### 问题 3: 配置文件未找到

**错误**: `FileNotFoundError: config.yaml`

**解决**: 确保在项目根目录运行，或使用绝对路径：
```bash
python main.py --config C:\_code\FS_SSC\config.yaml
```

### 问题 4: 数据文件未找到

**错误**: `FileNotFoundError: resource\dataSet.csv`

**检查 config.yaml**:
```yaml
resource_dir: "resource"
data_file: "dataSet.csv"
```

### 问题 5: 内存不足

**错误**: `MemoryError` 或程序卡死

**解决**: 减少参数
```yaml
algo_params:
  epoch: 50       # 减少迭代
  pop_size: 50    # 减少种群

model_params:
  n_trials: 50    # 减少搜索次数
```

---

## 💡 最佳实践

### 1. 实验管理

创建 `experiments/` 目录存放配置：

```
experiments/
├── baseline_hho_rf.yaml
├── tuned_hho_rf.yaml
├── ga_svm_boxcox.yaml
└── pca_pls.yaml
```

### 2. 参数搜索

使用脚本批量运行：

```bash
# run_experiments.sh
for config in experiments/*.yaml; do
    echo "Running $config"
    python main.py --config "$config"
done
```

### 3. 结果对比

使用 Python 脚本汇总结果：

```python
import pandas as pd
import glob

results = []
for metrics_file in glob.glob("log/*/model_metrics.csv"):
    df = pd.read_csv(metrics_file)
    test_metrics = df[df['Set'] == 'Test'].iloc[0]
    results.append({
        'Experiment': metrics_file.split('\\')[1],
        'R2': test_metrics['R2'],
        'RMSE': test_metrics['RMSE']
    })

summary = pd.DataFrame(results).sort_values('R2', ascending=False)
print(summary)
```

### 4. 日志分析

查看特定实验的日志：

```bash
# Windows
type log\HHO_RF_20260228_143052\run.log | findstr "INFO"

# Linux/Mac
cat log/HHO_RF_20260228_143052/run.log | grep "INFO"
```

### 5. 可视化对比

使用 matplotlib 对比多个实验：

```python
import matplotlib.pyplot as plt
import pandas as pd

experiments = [
    'log/HHO_RF_20260228_143052/prediction_results.csv',
    'log/GA_SVM_20260228_150000/prediction_results.csv',
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, exp_file in enumerate(experiments):
    df = pd.read_csv(exp_file)
    axes[i].scatter(df['Measured'], df['Predicted'])
    axes[i].plot([df['Measured'].min(), df['Measured'].max()],
                 [df['Measured'].min(), df['Measured'].max()], 'r--')
    axes[i].set_title(exp_file.split('/')[1])
    axes[i].set_xlabel('Measured')
    axes[i].set_ylabel('Predicted')
plt.tight_layout()
plt.savefig('comparison.png')
```

---

## 🎓 进阶技巧

### 自定义适应度函数

编辑 `feature_selection/base.py` 中的 `BaseMealpySelector.run_selection()`:

```python
def fitness_function(solution):
    sel_idx = np.where(solution > BINARY_THRESHOLD)[0]
    if len(sel_idx) == 0:
        return FITNESS_PENALTY_DEFAULT

    try:
        # 使用 RF 代替 PLS
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train[:, sel_idx], y_train)
        y_pred = model.predict(X_val[:, sel_idx])

        r2 = r2_score(y_val, y_pred)
        ratio = len(sel_idx) / X.shape[1]

        # 自定义适应度
        fitness = (1 - r2) + (self.penalty * ratio)
        return fitness
    except Exception:
        return FITNESS_PENALTY_DEFAULT
```

### 添加新的评估指标

编辑 `model/base.py` 中的 `calc_metrics()`:

```python
def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> Dict[str, Any]:
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rpd = (np.std(y_true) / rmse) if rmse > 1e-6 else 0

    # 添加新指标
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        "Set": label,
        "R2": r2,
        "RMSE": rmse,
        "RPD": rpd,
        "MAE": mae,      # 新增
        "MAPE": mape     # 新增
    }
```

---

## 📞 获取帮助

1. **查看日志**: `log/*/run.log` 包含详细的执行信息
2. **检查 README**: `README.md` 有完整的使用指南
3. **查看源码**: 所有代码都有详细的 docstrings
4. **测试导入**: 使用 Python 交互式环境测试

```python
import sys
sys.path.insert(0, '.')
import feature_selection
import model
from core.registry import list_algorithms, list_models, get_algorithm

# 查看可用算法
print(list_algorithms())

# 获取算法类
HHO = get_algorithm('HHO')
print(HHO.__doc__)
print(HHO.default_epoch)
```

---

**最后更新**: 2026-02-28
**版本**: 2.0.0
