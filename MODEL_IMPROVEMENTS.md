# Model Improvements to Fix Naive Baseline Problem

## Problem Identified

The model was predicting values very close to the last observed value (naive baseline):
- If last value = 1.5x → predicts ~1.5-2x
- If last value = 70x → predicts ~70-72x

This happens because:
1. **High autocorrelation** in time series makes predicting close to last value give good R²
2. **Features include last_1, last_2, last_3** which dominate predictions
3. **Model learns to copy** rather than learn patterns

## Solutions Implemented

### 1. **Delta Prediction (Primary Fix)**
- **Changed target**: Instead of predicting absolute multiplier value, predict **change/delta** from last value
- **Reconstruction**: `prediction = last_value + predicted_delta`
- **Why it works**: Forces model to learn patterns of change, not just copy values
- **Example**: 
  - Old: last=1.5 → predict 1.5-2.0 (naive)
  - New: last=1.5 → predict delta=-0.3 → final=1.2 (learned pattern)

### 2. **Reduced Recent Value Weight**
- **Feature engineering**: Reduced weight of `last_1`, `last_2`, `last_3` by multiplying by 0.3
- **Why**: Prevents these features from dominating predictions
- **Impact**: Model relies more on patterns, trends, and statistical features

### 3. **Better Loss Function**
- **Changed**: MSE → HuberLoss (delta=1.0)
- **Why**: Less sensitive to outliers, encourages learning patterns
- **Benefit**: More robust training

### 4. **Improved Regularization**
- **Learning rate**: Reduced from 0.001 → 0.0005
- **Weight decay**: Increased from 0.01 → 0.05
- **Why**: Prevents overfitting to recent patterns
- **Impact**: Model generalizes better

### 5. **Feature Emphasis**
- **Reduced**: Direct recent values (last_1, last_2, last_3) weight
- **Emphasized**: 
  - Trends (trend_5, trend_10)
  - Momentum (momentum_1, momentum_2, acceleration)
  - Patterns (high_ratio, volatility, transitions)
  - Relative features (z-scores, percentiles, distances from MA)

## Expected Results

After retraining with these changes:

1. **Predictions should vary more** from last value
2. **Model learns actual patterns** (momentum, trends, volatility)
3. **Better generalization** to different multiplier ranges
4. **More meaningful predictions** that capture market dynamics

## Why R² Was High Before

- **Time series autocorrelation**: Predicting close to last value naturally gives high R²
- **Not a real metric**: High R² doesn't mean the model learned patterns
- **Better metrics**: Focus on:
  - **Direction accuracy**: Does it predict increases/decreases correctly?
  - **Range prediction**: Does it predict realistic ranges?
  - **Pattern recognition**: Does it capture momentum/trends?

## Next Steps

1. **Retrain Model 2** with new delta prediction approach
2. **Monitor predictions** - they should now vary more from last value
3. **Check metrics** - R² might be lower but predictions should be more meaningful
4. **Validate** - Compare predictions vs actuals to see if patterns are learned

## Technical Details

### Delta Prediction Flow

```
Training:
  Target = actual_value - last_value  (delta)
  Model learns: f(features) → delta

Prediction:
  predicted_delta = model(features)
  final_prediction = last_value + predicted_delta
```

### Feature Weight Reduction

```python
# Before:
last_1, last_2, last_3  # Full weight

# After:
last_1 * 0.3, last_2 * 0.3, last_3 * 0.3  # Reduced weight
```

This forces the model to use other features (trends, patterns, statistics) instead of just copying recent values.

