# FS_SSC 使用检查清单

## 🎯 开始使用前的准备

### 1. 环境检查 ✅

```bash
# 检查 Python 版本 (需要 3.7+)
python --version

# 检查当前目录
pwd  # 应该在 C:\_code\FS_SSC

# 查看项目文件
ls -la
```

### 2. 安装依赖 ✅

```bash
# 安装所有依赖
pip install -r requirements.txt

# 验证关键包
python -c "import pandas, numpy, sklearn, mealpy, optuna, yaml; print('✓ 所有依赖已安装')"
```

### 3. 验证系统 ✅

```bash
# 运行完整验证
python verify_refactoring.py

# 预期输出: 10/10 测试通过
```

---

## 🚀 第一次运行

### 方法 1: 使用默认配置

```bash
# 运行默认配置 (HHO + RF)
python main.py

# 查看输出
ls log/HHO_RF_*/

# 预期文件:
# - train_data.csv
# - test_data.csv
# - selected_features-HHO.csv
# - prediction_results.csv
# - model_metrics.csv
# - plot_pred_vs_meas.png
# - HHO_selection_plot.png
# - run.log
```

### 方法 2: 快速入门演示

```bash
# 运行交互式演示
python quick_start_demo.py

# 按 Enter 逐步查看各个功能
```

---

## 📝 常见使用场景

### 场景 1: 切换算法和模型

```bash
# 编辑 config.yaml
algo_name: "GA"      # 改为 GA
model_name: "SVM"    # 改为 SVM

# 运行
python main.py
```

### 场景 2: 调整参数

```yaml
# 在 config.yaml 中添加
algo_params:
  epoch: 500
  pop_size: 300
  penalty: 0.25

model_params:
  n_trials: 500
```

### 场景 3: 启用 Box-Cox 变换

```yaml
# 在 config.yaml 中修改
use_boxcox: true
```

### 场景 4: 批量实验

```bash
# 运行批量实验脚本
python run_batch_experiments.py

# 等待所有实验完成
# 查看汇总结果
cat experiment_summary_*.csv
```

### 场景 5: 分析结果

```bash
# 运行结果分析
python analyze_results.py

# 查看生成的文件:
# - analysis_summary.csv
# - analysis_algorithm_comparison.png
# - analysis_heatmap.png
# - analysis_scatter.png
# - analysis_report.txt
```

---

## 🔧 添加新算法示例

### 步骤 1: 创建算法文件

```python
# feature_selection/woa.py
from mealpy.swarm_based.WOA import OriginalWOA
from core.registry import register_algorithm
from feature_selection.base import BaseMealpySelector

@register_algorithm("WOA")
class WOASelector(BaseMealpySelector):
    default_epoch = 200
    default_pop_size = 100
    default_penalty = 0.3

    def create_optimizer(self):
        return OriginalWOA(epoch=self.epoch, pop_size=self.pop_size)
```

### 步骤 2: 修改配置

```yaml
# config.yaml
algo_name: "WOA"
```

### 步骤 3: 运行

```bash
python main.py
```

### 步骤 4: 验证

```bash
# 检查算法是否注册
python -c "import feature_selection; from core.registry import list_algorithms; print(list_algorithms())"

# 应该看到 'WOA' 在列表中
```

---

## 📊 查看结果

### 查看日志

```bash
# 查看最新运行的日志
ls -lt log/ | head -5

# 查看具体日志
cat log/HHO_RF_20260228_*/run.log
```

### 查看指标

```bash
# 查看模型指标
cat log/HHO_RF_20260228_*/model_metrics.csv

# 预期格式:
# Set,R2,RMSE,RPD,Best_Params
# Train,0.8825,1.0067,2.92,"{'n_estimators': 17, ...}"
# Test,0.2536,2.7151,1.16,"{'n_estimators': 17, ...}"
```

### 查看预测结果

```bash
# 查看预测值
cat log/HHO_RF_20260228_*/prediction_results.csv

# 预期格式:
# Measured,Predicted
# 5.23,5.45
# 7.89,7.12
# ...
```

### 查看图表

```bash
# Windows
start log/HHO_RF_20260228_*/plot_pred_vs_meas.png
start log/HHO_RF_20260228_*/HHO_selection_plot.png

# Linux/Mac
xdg-open log/HHO_RF_20260228_*/plot_pred_vs_meas.png
```

---

## 🐛 故障排除

### 问题 1: ModuleNotFoundError

```bash
# 错误: ModuleNotFoundError: No module named 'xxx'
# 解决:
pip install -r requirements.txt
```

### 问题 2: 算法未找到

```bash
# 错误: KeyError: "Algorithm 'XXX' not found"
# 检查可用算法:
python -c "import feature_selection; from core.registry import list_algorithms; print(list_algorithms())"

# 确保算法名称正确 (区分大小写)
```

### 问题 3: 配置文件错误

```bash
# 错误: FileNotFoundError: config.yaml
# 解决: 确保在项目根目录运行
cd C:\_code\FS_SSC
python main.py
```

### 问题 4: 数据文件未找到

```bash
# 错误: FileNotFoundError: resource/dataSet.csv
# 检查数据文件:
ls resource/dataSet.csv

# 检查配置:
cat config.yaml | grep -A2 "resource"
```

### 问题 5: 内存不足

```bash
# 错误: MemoryError
# 解决: 减少参数
# 在 config.yaml 中:
algo_params:
  epoch: 50       # 减少迭代次数
  pop_size: 50    # 减少种群大小

model_params:
  n_trials: 50    # 减少搜索次数
```

---

## 📚 文档快速索引

### 基础使用
- **README.md** - 第一次使用必读
- **QUICK_REFERENCE.md** - 常用命令和参数

### 深入了解
- **REFACTORING_COMPLETE.md** - 重构详细说明
- **PROJECT_STATUS.md** - 项目状态和指标

### 开发参考
- **代码内 docstrings** - API 文档
- **feature_selection/base.py** - 算法基类
- **model/base.py** - 模型基类

---

## ✅ 使用检查清单

### 首次使用
- [ ] 安装依赖: `pip install -r requirements.txt`
- [ ] 运行验证: `python verify_refactoring.py`
- [ ] 查看演示: `python quick_start_demo.py`
- [ ] 运行默认配置: `python main.py`
- [ ] 查看结果: `ls log/HHO_RF_*/`

### 日常使用
- [ ] 编辑 `config.yaml` 选择算法/模型
- [ ] 运行: `python main.py`
- [ ] 查看日志: `cat log/*/run.log`
- [ ] 查看指标: `cat log/*/model_metrics.csv`
- [ ] 查看图表: 打开 PNG 文件

### 批量实验
- [ ] 运行批量脚本: `python run_batch_experiments.py`
- [ ] 等待完成
- [ ] 运行分析: `python analyze_results.py`
- [ ] 查看报告: `cat analysis_report.txt`

### 添加新算法
- [ ] 创建算法文件: `feature_selection/xxx.py`
- [ ] 继承基类: `BaseMealpySelector` 或 `BaseFeatureSelector`
- [ ] 添加装饰器: `@register_algorithm("XXX")`
- [ ] 实现方法: `create_optimizer()` 或 `run_selection()`
- [ ] 修改配置: `algo_name: "XXX"`
- [ ] 测试运行: `python main.py`

### 参数调优
- [ ] 确定调优目标 (精度/速度/特征数)
- [ ] 选择调优方法 (永久/临时/全局)
- [ ] 修改参数
- [ ] 运行实验
- [ ] 对比结果
- [ ] 记录最佳参数

---

## 🎓 最佳实践

### 1. 实验管理
```bash
# 创建实验目录
mkdir experiments

# 为每个实验创建配置
cp config.yaml experiments/exp1_hho_rf.yaml
cp config.yaml experiments/exp2_ga_svm.yaml

# 运行实验
python main.py --config experiments/exp1_hho_rf.yaml
python main.py --config experiments/exp2_ga_svm.yaml
```

### 2. 结果备份
```bash
# 备份重要结果
cp -r log/HHO_RF_20260228_143052 results_backup/best_run/

# 或压缩保存
tar -czf results_20260228.tar.gz log/
```

### 3. 版本控制
```bash
# 提交配置文件
git add config.yaml experiments/*.yaml
git commit -m "Add experiment configurations"

# 不要提交 log/ 目录 (已在 .gitignore)
```

### 4. 性能监控
```bash
# 记录运行时间
time python main.py

# 监控内存使用
# Windows: 任务管理器
# Linux: htop 或 top
```

---

## 🚀 进阶技巧

### 1. 自定义适应度函数
编辑 `feature_selection/base.py` 中的 `fitness_function`

### 2. 添加新的评估指标
编辑 `model/base.py` 中的 `calc_metrics`

### 3. 修改可视化样式
编辑 `core/constants.py` 中的可视化常量

### 4. 实现并行处理
在算法中使用 `n_jobs=-1` 参数

---

## 📞 获取帮助

### 在线资源
1. 查看文档: `README.md`, `QUICK_REFERENCE.md`
2. 运行演示: `python quick_start_demo.py`
3. 查看日志: `log/*/run.log`

### 调试技巧
1. 增加日志级别: 编辑 `core/logging_setup.py`
2. 使用 Python 调试器: `python -m pdb main.py`
3. 打印中间结果: 在代码中添加 `logger.info()`

### 常见问题
- 算法不收敛: 增加 `epoch` 或调整 `penalty`
- 过拟合: 启用 `use_boxcox` 或减少特征数
- 运行太慢: 减少 `epoch`, `pop_size`, `n_trials`
- 内存不足: 减少参数或使用更小的数据集

---

## 🎉 开始使用

现在您已经准备好开始使用 FS_SSC 2.0 了！

**推荐的第一步**:
1. ✅ 运行 `python verify_refactoring.py` 确保系统正常
2. ✅ 运行 `python quick_start_demo.py` 了解功能
3. ✅ 运行 `python main.py` 测试默认配置
4. ✅ 查看 `README.md` 了解详细用法

**祝您使用愉快！** 🚀

---

**最后更新**: 2026-02-28
**版本**: 2.0.0
