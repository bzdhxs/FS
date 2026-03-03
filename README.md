# FS_SSC: Feature Selection for Soil Salt Content

A plugin-based framework for hyperspectral remote sensing soil salt content feature selection and prediction.

## Features

- **Plugin Architecture**: Add new algorithms/models without modifying main code
- **7 Feature Selection Algorithms**: HHO, GA, GWO, MPA, CARS, SPA, PCA
- **3 Regression Models**: PLS, RF, SVM with hyperparameter optimization
- **Automatic Discovery**: Algorithms and models auto-register via decorators
- **YAML Configuration**: External config with command-line overrides
- **Small Sample Optimized**: Stratified sampling, Box-Cox transform support

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
python main.py                    # Use default config.yaml
python main.py --config exp.yaml  # Use custom config
```

### Configuration

Edit `config.yaml`:

```yaml
algo_name: "HHO"          # Feature selection algorithm
model_name: "RF"          # Regression model
target_col: "TS"          # Target variable
band_start: 14            # Starting band index
band_end: 164             # Ending band index
use_boxcox: false         # Enable Box-Cox transformation
test_size: 0.3            # Test set proportion
```

## Supported Algorithms

### Feature Selection

| Algorithm | Type | Default Parameters |
|-----------|------|-------------------|
| HHO | Metaheuristic | epoch=200, pop_size=250, penalty=0.2 |
| GA | Metaheuristic | epoch=200, pop_size=50, pc=0.85, pm=0.05 |
| GWO | Metaheuristic | epoch=200, pop_size=100, penalty=0.2 |
| MPA | Metaheuristic | epoch=250, pop_size=120, penalty=0.4 |
| CARS | Iterative | n_iter=200, k_fold=5 |
| SPA | Iterative | m_min=2, m_max=30 |
| PCA | Extraction | n_components=3 |

### Regression Models

| Model | Optimizer | Default Parameters |
|-------|-----------|-------------------|
| PLS | RandomizedSearchCV | n_iter=100, cv_folds=5 |
| RF | Optuna | n_trials=200, cv_folds=5 |
| SVM | Optuna | n_trials=300, cv_folds=5 |

## Parameter Tuning

### Permanent Changes
Edit algorithm/model files directly:

```python
# feature_selection/hho.py
class HHOSelector(BaseMealpySelector):
    default_epoch = 500        # Change here
    default_pop_size = 300
    default_penalty = 0.25
```

### Temporary Overrides
Use `config.yaml`:

```yaml
algo_params:
  epoch: 500
  pop_size: 300
  penalty: 0.25

model_params:
  n_trials: 500
```

## Adding New Algorithms

### Metaheuristic Algorithm

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

Then update `config.yaml`:
```yaml
algo_name: "WOA"
```

### Custom Algorithm

```python
# feature_selection/my_algo.py
from core.registry import register_algorithm
from feature_selection.base import BaseFeatureSelector, SelectionResult

@register_algorithm("MY_ALGO")
class MyAlgoSelector(BaseFeatureSelector):
    def run_selection(self, input_path, output_path, **kwargs):
        df, X_raw, y = self.load_data(input_path)

        # Your algorithm logic here
        selected_indices = [0, 5, 10, 15]
        selected_features = [self.feat_cols[i] for i in selected_indices]

        self.save_selection_result(df, y, selected_features, output_path)

        return SelectionResult(
            selected_features=selected_features,
            selected_indices=selected_indices
        )
```

## Adding New Models

```python
# model/xgb.py
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score
import optuna
from core.registry import register_model
from model.base import BaseModel

@register_model("XGB")
class XGBModel(BaseModel):
    default_n_trials = 200

    def train_and_predict(self, X_train, y_train, X_test):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            }
            model = XGBRegressor(**params, random_state=42)
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            return -cross_val_score(model, X_train, y_train, cv=cv,
                                    scoring='neg_root_mean_squared_error').mean()

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.default_n_trials)

        final_model = XGBRegressor(**study.best_params, random_state=42)
        final_model.fit(X_train, y_train)

        return (final_model.predict(X_train),
                final_model.predict(X_test),
                study.best_params)
```

## Output Structure

Each run creates a timestamped directory:

```
log/HHO_RF_20260228_143052/
├── train_data.csv                    # Training set
├── test_data.csv                     # Test set
├── selected_features-HHO.csv         # Selected features
├── prediction_results.csv            # Predictions
├── model_metrics.csv                 # Metrics (R², RMSE, RPD)
├── plot_pred_vs_meas.png            # Scatter plot
├── HHO_selection_plot.png           # Feature visualization
└── run.log                           # Execution log
```

## Project Structure

```
FS_SSC/
├── config.yaml              # Configuration file
├── main.py                  # Main entry point
├── requirements.txt         # Dependencies
├── core/                    # Framework core
│   ├── config.py            # Config management
│   ├── constants.py         # Global constants
│   ├── registry.py          # Plugin system
│   └── logging_setup.py     # Logging config
├── feature_selection/       # Feature selection algorithms
│   ├── base.py              # Base classes
│   ├── hho.py, ga.py, ...   # Specific algorithms
│   └── __init__.py          # Auto-discovery
├── model/                   # Regression models
│   ├── base.py              # Base class
│   ├── plsr.py, rf.py, ...  # Specific models
│   └── __init__.py          # Auto-discovery
├── utils/                   # Utilities
│   ├── data_processor.py    # Data preprocessing
│   └── data_split.py        # Stratified sampling
├── visualizer/              # Visualization
│   ├── feature_selection_visualizer.py
│   └── model_visualizer.py
├── resource/                # Data
│   └── dataSet.csv
└── log/                     # Output results
```

## Key Design Principles

1. **No Code Changes for New Algorithms**: Use `@register_algorithm` decorator
2. **Parameter Hierarchy**: config.yaml > algorithm defaults > global constants
3. **Unified Fitness Function**: All metaheuristic algorithms use `(1-R²) + penalty*ratio`
4. **Data Integrity**: Stratified sampling, train-only fitting, auto-inverse transform
5. **Logging Over Print**: All output via logging module

## Data Requirements

CSV format with:
- Target column (e.g., `TS` for soil salt content)
- Spectral band columns (e.g., `b14`, `b15`, ..., `b163`)
- Optional: `Sample_ID`, `Lon`, `Lat`

Example:
```csv
Sample_ID,b14,b15,...,b163,TS
1,2345,2456,...,3456,5.23
2,2123,2234,...,3234,7.89
```

## Common Tasks

### Run Different Algorithm/Model Combinations

```bash
# HHO + RF (default)
python main.py

# GA + SVM
# Edit config.yaml: algo_name: "GA", model_name: "SVM"
python main.py

# PCA + PLS
# Edit config.yaml: algo_name: "PCA", model_name: "PLS"
python main.py
```

### Enable Box-Cox Transformation

```yaml
use_boxcox: true
```

The lambda value is saved and predictions are automatically inverse-transformed.

### Adjust Algorithm Parameters

```yaml
algo_params:
  epoch: 300
  pop_size: 200
  penalty: 0.3
```

### Adjust Model Parameters

```yaml
model_params:
  n_trials: 500
  cv_folds: 10
```

## Troubleshooting

### Import Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Algorithm Not Found
Check available algorithms:
```python
from core.registry import list_algorithms
print(list_algorithms())
```

### Data File Not Found
Verify paths in `config.yaml`:
```yaml
resource_dir: "resource"
data_file: "dataSet.csv"
```

## Citation

If you use this framework, please cite:

```
[Your citation information here]
```

## License

[Your license information here]

## Contact

[Your contact information here]
