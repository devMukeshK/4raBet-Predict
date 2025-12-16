# New Modular Architecture

## Overview

The system has been restructured into a modular architecture with:
- **Separate model modules** for training/evaluation
- **Integration layer** that manages multiple models
- **Unified interface** for predictions
- **Automatic training schedules** based on model type

## File Structure

```
├── transformer_model.py      # Transformer model class and training/evaluation methods
├── model_integration.py       # Integration layer managing multiple models
├── transformer_app.py         # Flask app using integration layer
├── models/                    # Saved models and metadata
│   ├── transformer_global_classification.pth
│   ├── transformer_current_regression.pth
│   ├── model1_metadata.json
│   └── model2_metadata.json
└── data/                      # Data files
    ├── aviator_payouts_global.csv
    └── aviator_payouts_YYYYMMDD.csv
```

## Two Model System

### Model 1: Global Classification Model
- **Task**: Binary classification (predict if multiplier > 20)
- **Data Source**: `data/aviator_payouts_global.csv` (all-time data)
- **Training Schedule**: Once per day
- **Purpose**: Capture patterns for high multipliers (>20x) from larger historical dataset
- **Output**: Probability that next multiplier will be > 20

### Model 2: Current Regression Model
- **Task**: Regression (predict exact multiplier value)
- **Data Source**: `data/aviator_payouts_YYYYMMDD.csv` (current day data)
- **Training Schedule**: Every 30 minutes
- **Purpose**: Accurate multiplier prediction using recent data
- **Output**: Predicted multiplier value with confidence range

## Training Metadata

Both models save training metadata in JSON files:
- `models/model1_metadata.json`
- `models/model2_metadata.json`

Each metadata file contains:
```json
{
  "last_training_time": "2024-12-13T10:30:00",
  "training_duration_seconds": 120.5,
  "training_samples": 5000,
  "metrics": {
    "accuracy": 0.85,  // For classification
    "mae": 0.5,        // For regression
    ...
  },
  "model_path": "models/transformer_global_classification.pth"
}
```

## Usage

### Initialize Integration Layer

```python
from model_integration import ModelIntegration, ModelConfig

# Create integration instance
integration = ModelIntegration()

# Load existing models
integration.load_model1()  # Global classification
integration.load_model2()  # Current regression
```

### Train Models

```python
# Train Model 1 (daily, on global data)
if integration.should_retrain_model1():
    result = integration.train_model1()
    print(f"Model 1 trained: {result['metrics']}")

# Train Model 2 (every 30 min, on current data)
if integration.should_retrain_model2():
    result = integration.train_model2()
    print(f"Model 2 trained: {result['metrics']}")
```

### Make Predictions

```python
# Load data
import pandas as pd
data = pd.read_csv("data/aviator_payouts_20241213.csv")
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Predict
prediction = integration.predict(data)

# Unified output format:
# {
#   'predicted_value': 2.5,        # From Model 2
#   'min_range': 2.0,
#   'max_range': 3.0,
#   'confidence': 85.0,
#   'risk_level': 'LOW',
#   'prob_above_20': 0.15,        # From Model 1
#   'input_sequence': [...],
#   'recent_actuals': [...],
#   ...
# }
```

### Check Model Status

```python
status = integration.get_model_status()
print(status)
# {
#   'model1': {'loaded': True, 'should_retrain': False, 'metadata': {...}},
#   'model2': {'loaded': True, 'should_retrain': True, 'metadata': {...}}
# }
```

## Integration with Flask App

The `transformer_app.py` should be updated to:
1. Use `ModelIntegration` instead of direct model access
2. Check training schedules automatically
3. Use unified prediction interface
4. Display both model metrics in UI

## Benefits

1. **Modularity**: Models can be swapped/added easily
2. **Unified Interface**: All models return same format
3. **Automatic Scheduling**: Training happens automatically
4. **Metadata Tracking**: Training times and metrics saved
5. **Scalability**: Easy to add more models in future

## Future Models

To add a new model:
1. Create model class in separate file (e.g., `lstm_model.py`)
2. Implement same interface as `transformer_model.py`
3. Add to `ModelIntegration` class
4. Update Flask app to use new model

All models should return the same prediction format for consistency.

