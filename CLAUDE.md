# FS_SSC 项目说明 (Claude 上下文)

## 项目概述

**FS_SSC** (Feature Selection for Soil Salt Content) 是一个插件式框架，用于高光谱遥感土壤盐分含量的特征选择与预测。

- 目标：从 150+ 高光谱波段中选出最优特征子集，结合回归模型预测土壤盐分
- 数据集：68 个小样本，169 列（含 164 个高光谱波段 b1-b164）
- 目标变量：`TS`（土壤盐分含量）

---

## 项目结构

```
FS_SSC/
├── core/                    # 框架基础设施
│   ├── config.py            # YAML 配置管理 (AppConfig dataclass)
│   ├── registry.py          # 插件注册 (@register_algorithm / @register_model)
│   ├── logging_setup.py     # 统一日志 (RichHandler + 文件日志)
│   ├── constants.py         # 全局常量
│   └── console.py           # Rich 美化输出
├── feature_selection/       # 特征选择算法（插件式）
│   ├── base.py              # 基类 BaseFeatureSelector / BaseMealpySelector
│   ├── hho.py               # Harris Hawks Optimization
│   ├── ga.py                # Genetic Algorithm
│   ├── gwo.py               # Grey Wolf Optimizer
│   ├── mpa.py               # Marine Predators Algorithm
│   ├── cars.py              # CARS（基于 PLS 的竞争性采样）
│   ├── spa.py               # SPA（逐步投影算法）
│   ├── pca.py               # PCA（特征提取，非选择）
│   ├── sghho.py             # SG-HHO（改进算法，光谱组结构）
│   └── __init__.py          # 自动发现注册
├── model/                   # 回归模型（插件式）
│   ├── base.py              # 基类 BaseModel + calc_metrics
│   ├── plsr.py              # PLS Regression
│   ├── rf.py                # Random Forest + Optuna
│   ├── svm.py               # SVM + Optuna
│   └── __init__.py          # 自动发现注册
├── utils/
│   ├── data_processor.py    # 数据读取与分层抽样
│   └── data_split.py        # 回归分层抽样（小样本优化）
├── visualizer/
│   ├── feature_selection_visualizer.py
│   └── model_visualizer.py
├── improve/                 # 改进算法原型
│   ├── SGHHO.py             # SG-HHO 完整实现（457行）
│   └── CLHHO.py
├── main.py                  # 主入口（5步管道）
├── config.yaml              # 运行配置
├── requirements.txt
└── resource/
    └── dataSet.csv          # 原始数据（68样本，169列）
```

---

## 插件架构

添加新算法只需创建一个文件，无需修改主程序：

```python
# feature_selection/woa.py
from core.registry import register_algorithm
from feature_selection.base import BaseMealpySelector
from mealpy.swarm_based.WOA import OriginalWOA

@register_algorithm("WOA")
class WOASelector(BaseMealpySelector):
    default_epoch = 200
    default_pop_size = 100
    default_penalty = 0.3

    def create_optimizer(self):
        return OriginalWOA(epoch=self.epoch, pop_size=self.pop_size)
```

模型同理，使用 `@register_model("NAME")` 装饰器。

---

## 支持的算法与模型

| 类型 | 可用选项 |
|------|---------|
| 特征选择 | HHO, GA, GWO, MPA, CARS, SPA, SGHHO |
| 特征提取 | PCA |
| 回归模型 | PLS, RF, SVM |

---

## 执行管道（5步）

1. **数据预处理** — 分层抽样，生成 train/test CSV
2. **特征选择/提取** — 运行指定算法，保存选中特征
3. **可视化** — 绘制光谱曲线 + 选中波段
4. **建模** — 超参数优化（Optuna / RandomizedSearchCV）+ 交叉验证
5. **结果导出** — config.json, summary.json, predictions.csv, metrics.csv

---

## 配置说明

关键字段（`config.yaml`）：

```yaml
algo_name: "PCA"       # 特征选择算法
model_name: "PLS"      # 回归模型
target_col: "TS"       # 目标变量
band_start: 14         # 起始波段索引
band_end: 164          # 结束波段索引
test_size: 0.3
show_plots: false
base_log_dir: "log"
algo_params:
  epoch: 200
  pop_size: 50
model_params:
  n_trials: 500
  cv_folds: 5
```

参数优先级：全局常量 < 类默认值 < config.yaml < 命令行

---

## 输出结构

每次运行生成时间戳目录：

```
log/20260309_XXXXXX_HHO_RF/
├── config.json
├── summary.json
├── logs/main.log
├── data/train.csv, test.csv, selected_features_HHO.csv
├── plots/feature_selection_HHO.png, prediction_scatter.png
└── results/predictions.csv, metrics.csv
```

---

## 常用命令

```bash
# 运行默认配置
python main.py

# 使用自定义配置
python main.py --config my_config.yaml
```

---

## 技术栈

- Python 3.14（虚拟环境）
- pandas, numpy, scikit-learn, scipy
- mealpy（元启发式算法）
- optuna（超参数优化）
- matplotlib, rich, pyyaml
