# 更新日志

## 2026-03-03 - 重大重构

### 删除功能
- **移除 Box-Cox 变换**: 经测试发现该变换对模型性能提升不明显,已完全移除
  - 删除 `core/constants.py` 中的 `BOXCOX_OFFSET`
  - 删除 `core/config.py` 中的 `use_boxcox` 配置项
  - 简化 `utils/data_processor.py`,移除所有 Box-Cox 相关方法
  - 简化 `model/base.py`,移除自动反变换逻辑
  - 更新 `config.yaml`,移除 `use_boxcox` 配置

### 新增功能

#### 1. 优化的日志结构
**新的目录命名格式**: `YYYYMMDD_HHMMSS_算法_模型/`
- 时间戳前置,便于按时间排序
- 示例: `20260303_171234_HHO_RF/`

**结构化输出目录**:
```
log/20260303_171234_HHO_RF/
├── config.json                    # 实验配置(JSON格式)
├── summary.json                   # 核心指标汇总(JSON格式)
├── logs/                          # 日志文件夹
│   └── main.log                   # 主流程日志
├── data/                          # 数据文件夹
│   ├── train.csv                  # 训练集
│   ├── test.csv                   # 测试集
│   └── selected_features_HHO.csv  # 选择的特征
├── plots/                         # 可视化文件夹
│   ├── feature_selection_HHO.png  # 特征选择可视化
│   └── prediction_scatter.png     # 预测散点图
└── results/                       # 结果文件夹
    ├── predictions.csv            # 预测结果
    └── metrics.csv                # 评估指标
```

#### 2. JSON 配置和结果导出
**config.json** - 完整实验配置:
```json
{
  "experiment": {
    "id": "20260303_171234_HHO_RF",
    "timestamp": "2026-03-03 17:12:34",
    "algorithm": "HHO",
    "model": "RF"
  },
  "algorithm_params": {...},
  "model_params": {...},
  "data_config": {...}
}
```

**summary.json** - 核心指标汇总:
```json
{
  "experiment_id": "20260303_171234_HHO_RF",
  "feature_selection": {
    "n_selected": 8,
    "selection_time_sec": 76.23
  },
  "model_training": {
    "training_time_sec": 30.45,
    "best_hyperparams": {...}
  },
  "performance": {
    "train": {"R2": 0.8825, "RMSE": 1.0067, "RPD": 2.89},
    "test": {"R2": 0.2536, "RMSE": 2.7151, "RPD": 1.16}
  }
}
```

#### 3. 性能计时
- 自动记录特征选择耗时
- 自动记录模型训练耗时
- 在日志和 summary.json 中输出

### 优势
1. **便于批量分析**: JSON 格式便于编写脚本批量提取和对比实验结果
2. **时间序列友好**: 新的命名格式自然按时间排序
3. **代码简化**: 移除 Box-Cox 相关代码,减少维护成本
4. **结构清晰**: 分层目录结构,文件分类明确

### 兼容性说明
- 旧的实验结果目录(如 `HHO_RF_20260228_145609/`)仍然保留,不受影响
- 新运行的实验将使用新的目录结构和命名格式
- 配置文件 `config.yaml` 保持向后兼容

### 批量分析示例
```python
import json
import pandas as pd
from pathlib import Path

# 批量读取所有实验结果
results = []
for exp_dir in Path("log").glob("202*_*_*_*/"):  # 新格式
    summary_file = exp_dir / "summary.json"
    if summary_file.exists():
        summary = json.load(open(summary_file))
        results.append({
            "实验ID": summary["experiment_id"],
            "算法": summary["experiment_id"].split("_")[2],
            "模型": summary["experiment_id"].split("_")[3],
            "特征数": summary["feature_selection"]["n_selected"],
            "测试R2": summary["performance"]["test"]["R2"],
            "测试RMSE": summary["performance"]["test"]["RMSE"]
        })

df = pd.DataFrame(results)
print(df.sort_values("测试R2", ascending=False))
```
