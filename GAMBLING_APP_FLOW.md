# 🎰 Gambling Application - Complete Flow & Profit/Loss Logic

## 📋 Overview

This document explains how the Aviator gambling application works, from data collection to profit/loss calculation.

---

## 🔄 Complete Application Flow

### **Step 1: Data Collection** (`main.py`)
- **Purpose**: Continuously monitor Aviator game for new payout multipliers
- **Process**:
  1. Logs into 4RaBet website
  2. Navigates to Aviator game
  3. Monitors payout elements in real-time
  4. Saves each new multiplier with timestamp to CSV
  5. Example: `2025-12-10 23:45:17, 2.5x`

### **Step 2: Model Training** (`train_model()`)
- **Purpose**: Train ML model to predict next multiplier
- **Process**:
  1. Loads all historical data from CSV
  2. Extracts 70+ time-series features:
     - Rolling statistics (mean, std, min, max)
     - Trends and momentum
     - Pattern recognition
     - Global vs local comparisons
  3. Trains ensemble model (XGBoost, LightGBM, RandomForest, etc.)
  4. Model learns patterns from historical data

### **Step 3: Prediction** (`predict_next()`)
- **Purpose**: Predict next multiplier range
- **Process**:
  1. Uses trained model with latest data
  2. Generates prediction range: `"2.3x - 3.8x"`
  3. Stores in `current_prediction` for betting decision
  4. Example: `pred_min = 2.3x`, `pred_max = 3.8x`

### **Step 4: Betting Decision** (`simulate_bet()`)
- **Purpose**: Decide whether to place bet based on prediction
- **Rules**:
  - **RULE 1**: If `pred_min < 2.0` → **NO BET** (skip this round)
  - **RULE 2**: If `pred_min > 2.0` → **PLACE BET** (₹100)

### **Step 5: Actual Result Processing** (`check_and_process_bet()`)
- **Purpose**: Compare actual result with prediction and calculate profit/loss
- **Process**:
  1. New multiplier appears in CSV (e.g., `3.2x`)
  2. Compares `actual_multiplier` vs `pred_min`
  3. Applies profit/loss rules (RULE 3 & 4)

### **Step 6: Profit/Loss Calculation** (`simulate_bet_min_range()`)
- **Purpose**: Calculate profit or loss based on actual result
- **Rules**:
  - **RULE 3**: If `actual >= pred_min` → **WIN**
    - Profit = `pred_min × 100`
    - Example: `pred_min = 2.5x` → Profit = ₹250
  - **RULE 4**: If `actual < pred_min` → **LOSS**
    - Loss = ₹100 (bet amount)

### **Step 7: Wallet Update**
- **Purpose**: Update wallet balance after each bet
- **Calculation**:
  - **WIN**: `balance - ₹100 (bet) + (pred_min × ₹100) (return)`
    - Example: ₹50,000 - ₹100 + ₹250 = ₹50,150
  - **LOSS**: `balance - ₹100 (bet)`
    - Example: ₹50,000 - ₹100 = ₹49,900

---

## 💰 Profit/Loss Examples

### Example 1: WIN Scenario
```
Initial Balance: ₹50,000
Prediction: 2.5x - 3.8x (pred_min = 2.5x)
Bet Amount: ₹100
Actual Result: 3.2x

Decision: 3.2x >= 2.5x → WIN (RULE 3)
Profit Calculation: 2.5 × 100 = ₹250
Wallet Update: ₹50,000 - ₹100 + ₹250 = ₹50,150
Net Change: +₹150
```

### Example 2: LOSS Scenario
```
Initial Balance: ₹50,000
Prediction: 2.5x - 3.8x (pred_min = 2.5x)
Bet Amount: ₹100
Actual Result: 1.8x

Decision: 1.8x < 2.5x → LOSS (RULE 4)
Loss Calculation: ₹100
Wallet Update: ₹50,000 - ₹100 = ₹49,900
Net Change: -₹100
```

### Example 3: NO BET Scenario
```
Prediction: 1.8x - 2.1x (pred_min = 1.8x)
Decision: 1.8x < 2.0 → NO BET (RULE 1)
Result: No bet placed, no profit/loss, wallet unchanged
```

---

## 📊 Key Variables

| Variable | Description | Initial Value |
|----------|-------------|---------------|
| `current_balance` | Main wallet balance | ₹50,000 |
| `base_bet_amount` | Bet amount per round | ₹100 |
| `max_balance` | Maximum balance cap | ₹50,000 |
| `betting_history` | List of all bets | `[]` |
| `min_range_bets` | Bets based on min range | `[]` |

---

## 🔍 Profit/Loss Tracking

### Total Profit/Loss Calculation
```python
# Sum of all profit_loss values from betting_history
total_profit_loss = sum(bet['profit_loss'] for bet in betting_history)

# Wallet P/L (current vs initial)
wallet_profit_loss = current_balance - max_balance  # ₹50,000
```

### Win Rate Calculation
```python
total_wins = sum(1 for bet in betting_history if bet['is_win'])
total_bets = len(betting_history)
win_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0
```

---

## ⚙️ Technical Implementation

### Betting Flow Diagram
```
CSV Update (New Multiplier)
    ↓
check_and_process_bet()
    ↓
simulate_bet(prediction, actual_multiplier)
    ↓
Check: pred_min > 2.0?
    ├─ NO → Return None (No bet)
    └─ YES → Continue
        ↓
simulate_bet_min_range(pred_min, actual, bet_amount, balance)
    ↓
Check: actual >= pred_min?
    ├─ YES → WIN → Profit = pred_min × 100
    └─ NO → LOSS → Loss = 100
        ↓
Update current_balance
    ↓
Record in betting_history
```

---

## 🎯 Rules Summary

| Rule | Condition | Action |
|------|-----------|--------|
| **RULE 1** | `pred_min < 2.0` | No bet, no profit/loss calculation |
| **RULE 2** | `pred_min > 2.0` | Place bet (₹100) |
| **RULE 3** | `actual >= pred_min` | WIN → Profit = `pred_min × 100` |
| **RULE 4** | `actual < pred_min` | LOSS → Loss = ₹100 |

---

## 📈 Performance Metrics

The application tracks:
- **Total Bets**: Number of bets placed
- **Total Wins**: Number of winning bets
- **Total Losses**: Number of losing bets
- **Win Rate**: Percentage of winning bets
- **Total Profit/Loss**: Sum of all profit_loss values
- **Wallet P/L**: Current balance - Initial balance (₹50,000)

---

## 🔐 Safety Features

1. **Balance Bounds**: Wallet cannot go below ₹0 or above ₹50,000
2. **Insufficient Balance Check**: Bet blocked if balance < ₹100
3. **Duplicate Prevention**: Tracks processed multipliers to avoid duplicate bets
4. **Betting Enabled Flag**: Can disable betting system if needed

---

## 🚀 Running the Application

1. **Start Data Collection**: `python main.py`
   - Monitors Aviator game and saves multipliers to CSV

2. **Start Prediction App**: `python predictor_app.py`
   - Trains model, makes predictions, and processes bets
   - Web interface: `http://localhost:5000`

3. **View Results**: Open browser to see:
   - Current predictions
   - Betting history
   - Profit/loss statistics
   - Wallet balance

---

## 📝 Notes

- Profit is calculated based on **predicted minimum range** (pred_min), NOT actual multiplier
- Loss is always ₹100 (the bet amount)
- Wallet balance reflects actual net change after each bet
- All bets are recorded with timestamp, prediction, actual result, and profit/loss
