# 🚀 Dual Approach Aviator Predictor - Run Instructions

## Overview
This system implements two prediction approaches:
1. **Binary Classification**: Predicts probability of next multiplier > 2.0x (bets when prob > 0.7)
2. **Multi-Class + LLM**: 3-class classification (<1.5x, 1.5x-3x, >3x) with LLM reasoning layer

## Prerequisites

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

Key dependencies:
- `h2o` - H2O AutoML for model training
- `flask` & `flask-socketio` - Web server
- `pandas`, `numpy`, `scikit-learn` - Data processing
- `langchain-openai` - LLM reasoning (optional, has fallback)
- `selenium` - Data scraping (for main.py)

### 2. Set Up Environment Variables (Optional)
If using LLM reasoning, set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

**Note**: LLM is optional - the system has a rule-based fallback if LLM is not available.

## Step-by-Step Setup

### Step 1: Prepare Data
Ensure you have multiplier data in CSV format:
- Location: `data/aviator_payouts_global.csv`
- Required columns: `timestamp`, `multiplier`
- Format: CSV with timestamp and multiplier values

If you don't have data yet, you can:
1. Run `main.py` to scrape data (requires Selenium setup)
2. Or manually create a CSV file with the required columns

### Step 2: Start the Flask Application
```bash
python transformer_app.py
```

The server will start on:
- **URL**: http://localhost:5002
- **Port**: 5002 (configurable via environment variable)

You should see:
```
✅ H2O initialized
✅ Found CSV: data/aviator_payouts_global.csv (XXXX records)
🔄 Training H2O Binary Classification model...
✅ H2O Binary model ready!
🔄 Training H2O Multi-Class + LLM model...
✅ H2O Multi-Class + LLM model ready!
System initialized

🚀 Starting Flask server...
📊 Access the UI at: http://localhost:5002
```

### Step 3: Access the Web UI
Open your browser and navigate to:
```
http://localhost:5002
```

## Using the Web Interface

### Tab 1: Live Predictions
- **Binary Classification**: Shows probability of >2.0x, betting action, AUC, confidence
- **Multi-Class + LLM**: Shows class probabilities, LLM decision, expected value, reasoning

### Tab 2: Simulation Results
- Compare performance of both approaches
- View metrics: Final Balance, Total Return, Sharpe Ratio, Max Drawdown, Win Rate

### Tab 3: Betting Stats
- Real-time betting statistics for both approaches
- Bankroll, Total Bets, Wins/Losses, Total P/L

## Available Actions

### 1. Train Models
- **Train Both Models**: Trains both Binary and Multi-Class models
- **Train Binary Only**: Trains only Binary Classification model
- **Train Multi-Class+LLM**: Trains only Multi-Class + LLM model

**Note**: First-time training may take 3-5 minutes per model.

### 2. Run Simulation
- Simulates betting strategies on historical data
- Compares both approaches side-by-side
- Results saved to `models/simulation_results.json`

### 3. Start/Stop Monitoring
- **Start Monitoring**: Begins real-time prediction and betting
- **Stop Monitoring**: Stops the monitoring loop

## File Structure

```
4raBet-Predict/
├── data/
│   └── aviator_payouts_global.csv    # Input data
├── models/
│   ├── h2o_automl_model/             # Binary model files
│   ├── h2o_multiclass_model/         # Multi-Class model files
│   ├── h2o_metadata.json            # Binary model metadata
│   ├── h2o_multiclass_metadata.json # Multi-Class model metadata
│   ├── h2o_bet_history.csv          # Binary betting logs
│   ├── h2o_multiclass_bet_history.csv # Multi-Class betting logs
│   └── simulation_results.json      # Simulation results
├── templates/
│   └── transformer_index.html       # Web UI
├── h2o_automl_model.py              # Model implementations
├── model_integration.py             # Model integration layer
├── transformer_app.py               # Flask application
├── main.py                          # Data scraper (optional)
└── requirements.txt                 # Dependencies
```

## API Endpoints

### Training
```bash
# Train both models
curl -X POST http://localhost:5002/api/train \
  -H "Content-Type: application/json" \
  -d '{"model": "both"}'

# Train binary only
curl -X POST http://localhost:5002/api/train \
  -H "Content-Type: application/json" \
  -d '{"model": "binary"}'

# Train multiclass only
curl -X POST http://localhost:5002/api/train \
  -H "Content-Type: application/json" \
  -d '{"model": "multiclass"}'
```

### Predictions
```bash
# Get binary prediction
curl http://localhost:5002/api/predict

# Get multiclass prediction
curl http://localhost:5002/api/predict_multiclass
```

### Simulations
```bash
# Run simulation
curl -X POST http://localhost:5002/api/simulate

# Get simulation results
curl http://localhost:5002/api/simulation_results
```

### Status
```bash
# Get system status
curl http://localhost:5002/api/status
```

## Troubleshooting

### Issue: H2O initialization fails
**Solution**: 
- Ensure Java is installed (H2O requires Java)
- Check available memory (H2O needs at least 2GB RAM)
- Try: `h2o.init(max_mem_size='2G')`

### Issue: LLM not working
**Solution**:
- Check if `OPENAI_API_KEY` is set
- System will use rule-based fallback if LLM unavailable
- Check console for LLM error messages

### Issue: Models not training
**Solution**:
- Ensure data file exists: `data/aviator_payouts_global.csv`
- Check data has at least 100 samples
- Verify data format (timestamp, multiplier columns)

### Issue: Port already in use
**Solution**:
- Change port in `transformer_app.py`: `socketio.run(app, host='0.0.0.0', port=5003)`
- Or kill process using port 5002

### Issue: Insufficient data
**Solution**:
- Need at least 100 samples for training
- More data = better model performance
- Use `main.py` to scrape more data

## Model Details

### Binary Classification Approach
- **Target**: Probability that next multiplier > 2.0x
- **Betting Threshold**: prob > 0.7
- **Features**: lag_1-3, rolling stats, EMA, volatility ratio, streaks
- **Model**: H2O AutoML (Binary Classification)
- **Metrics**: AUC, LogLoss, Precision, Recall, F1

### Multi-Class + LLM Approach
- **Target**: 3 classes (<1.5x, 1.5x-3x, >3x)
- **Features**: lag_1-2, rolling_mean_10, rolling_std_10, ema_10, volatility_ratio, streaks
- **Model**: H2O AutoML (Multi-Class Classification)
- **Decision Layer**: LLM reasoning with expected value calculation
- **Metrics**: LogLoss, Accuracy, Precision, Recall, F1

## Betting Strategy

### Binary Classification
- **BET**: When prob > 0.7, bet ₹100 on multiplier > 2.0x
- **SMALL_BET**: When prob ≥ 0.55, bet ₹50 on multiplier > 2.0x
- **NO_BET**: When prob < 0.55

### Multi-Class + LLM
- **BET**: When LLM recommends BET and EV > 0, bet ₹100 on multiplier ≥ 1.5x
- **SMALL_BET**: When LLM recommends SMALL_BET and EV > 0, bet ₹50 on multiplier ≥ 1.5x
- **NO_BET**: When LLM recommends NO_BET or EV ≤ 0

## Performance Expectations

### Realistic Results
- **AUC (Binary)**: 0.52-0.58 (expected for IID data)
- **LogLoss**: Slightly better than baseline
- **Live Hit Rate**: ~55%
- **Profitability**: Depends on bankroll management

**Important**: Aviator multipliers are designed to be near-IID (independent and identically distributed). Perfect prediction is impossible. The goal is risk management and probability-based decision making.

## Next Steps

1. **Train Models**: Click "Train Both Models" in the UI
2. **Run Simulation**: Click "Run Simulation" to see historical performance
3. **Start Monitoring**: Click "Start Monitoring" for live predictions
4. **Analyze Results**: Check betting logs and simulation results

## Support

For issues or questions:
- Check console logs for error messages
- Verify all dependencies are installed
- Ensure data file exists and is properly formatted
- Review model metadata files for training details

---

**Happy Predicting! 🚀**

