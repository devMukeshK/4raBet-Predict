# Bayesian Sequence Model - Implementation Guide

## Overview

The system now includes **three prediction approaches**:

1. **H2O Binary Classification**: ML-based probability prediction (requires training)
2. **Bayesian Sequence Model**: Statistical sequence-based prediction (always available, no training)
3. **Both run in parallel** for comparison

## Bayesian Model Details

### How It Works

The Bayesian model uses **Beta-Bernoulli updating** to estimate:
```
P(next multiplier > threshold | recent sequence)
```

**Key Features:**
- ✅ No training required (always available)
- ✅ Adapts to recent patterns
- ✅ Provides uncertainty estimates (confidence)
- ✅ Avoids overfitting
- ✅ Statistically defensible

### Parameters

- **threshold**: Success threshold (default: 2.0x)
- **window**: Recent samples to consider (default: 20)
- **alpha_prior, beta_prior**: Beta prior parameters (default: 1.5, 1.5)
- **decay**: Exponential decay for older observations (default: 0.95)

### Betting Logic

- **BET**: Probability > 0.6 AND Confidence ≥ 0.5
- **SMALL_BET**: Probability > 0.55 AND Confidence ≥ 0.5
- **NO_BET**: Otherwise

## Files Modified

### 1. `h2o_automl_model.py`
Added:
- `bayesian_seq_predict()` - Core Bayesian prediction function
- `bayesian_action()` - Convert probability to betting action
- `multi_threshold_bayes()` - Multi-threshold predictions
- `predict_bayesian()` - Full prediction pipeline

### 2. `model_integration.py`
Added:
- `predict_bayesian()` - Integration method for Bayesian predictions

### 3. `transformer_app.py`
Added:
- `predict_next_bayesian()` - Get Bayesian prediction
- `simulate_bayesian_strategy()` - Betting simulation for Bayesian
- `run_betting_simulations()` - Run both simulations
- Updated CSV monitor loop to handle both approaches
- Separate betting stats for Bayesian model

### 4. `templates/transformer_index.html`
Updated:
- Side-by-side display of H2O and Bayesian predictions
- Separate betting stats for both approaches
- Simulation results comparison
- Clean, modern UI

## API Endpoints

### Get Bayesian Prediction
```bash
GET /api/predict_bayesian?threshold=2.0
```

### Run Simulations
```bash
POST /api/simulate
```

### Get Simulation Results
```bash
GET /api/simulation_results
```

## Betting Logs

- **H2O**: `models/h2o_bet_history.csv`
- **Bayesian**: `models/bayesian_bet_history.csv`
- **Simulations**: `models/simulation_results.json`

## Usage Example

```python
from h2o_automl_model import predict_bayesian
import pandas as pd

# Load data
df = pd.read_csv('data/aviator_payouts_global.csv')

# Get Bayesian prediction
prediction = predict_bayesian(
    data=df,
    threshold=2.0,
    window=20
)

print(f"Probability: {prediction['probability_gt_threshold']:.2%}")
print(f"Action: {prediction['betting_action']}")
print(f"Reason: {prediction['reason']}")
```

## Model Comparison

| Feature | H2O Binary | Bayesian Sequence |
|---------|-----------|-------------------|
| Training Required | ✅ Yes | ❌ No |
| Adapts to Patterns | ✅ Yes | ✅ Yes |
| Uncertainty Estimate | Partial | ✅ Full (confidence) |
| Speed | Fast (after training) | ⚡ Instant |
| Data Requirements | 100+ samples | 20+ samples |
| Overfitting Risk | Medium | Low |

## Key Advantages of Bayesian Model

1. **Always Available**: No training needed
2. **Honest Uncertainty**: Provides confidence estimates
3. **Adaptive**: Updates beliefs based on recent sequence
4. **Stable**: Doesn't overreact to noise
5. **Interpretable**: Clear reasoning for decisions

## Simulation Results

Both models run separate simulations:
- **H2O**: Uses prob ≥ 0.7, target 8.0x
- **Bayesian**: Uses action-based logic, target 2.0x

Results are saved to `models/simulation_results.json` for comparison.

---

**Note**: The Bayesian model is designed to be conservative and honest about uncertainty. It won't make false predictions but will adapt to recent patterns in the data.

