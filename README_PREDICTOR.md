# Aviator Multiplier Predictor - Auto-Sync System

## Overview
This system runs two processes in parallel:
1. **Data Collection** (`main.py`) - Collects multiplier data from Aviator game
2. **Predictor App** (`predictor_app.py`) - Trains ML model and predicts next outcomes with auto-sync

## Features

### Auto-Sync Capabilities
- ✅ **Automatic CSV Monitoring**: Watches for new data in CSV files
- ✅ **Auto-Retraining**: Model automatically retrains when new data arrives
- ✅ **Real-time Predictions**: Predictions update automatically every 3 seconds
- ✅ **File System Watching**: Uses watchdog to detect CSV file changes
- ✅ **Background Processing**: Runs in parallel threads without blocking

### Machine Learning
- **Model**: Ensemble (Random Forest + Gradient Boosting + Linear Regression)
- **Features**: 25 advanced time-series features (trends, volatility, momentum, patterns, percentiles)
- **Auto-Training**: Instant sync - retrains with ALL data when ≥5 new records arrive
- **Real-time Updates**: Predictions refresh automatically with instant data sync
- **Data Usage**: Uses 100% of available data for training (not random subset)

## Installation

```bash
# Install all dependencies
pip install -r requirements.txt
```

## Usage

### Option 1: Run Both Processes Separately (Recommended)

**Terminal 1 - Data Collection:**
```bash
python main.py
```

**Terminal 2 - Predictor App:**
```bash
python predictor_app.py
```

Then open browser: `http://localhost:5001`

### Option 2: Run Both in Parallel (Single Command)

**Python Script (Cross-platform):**
```bash
python run_parallel.py
```

**Shell Script (macOS/Linux):**
```bash
chmod +x start_parallel.sh
./start_parallel.sh
```

**Batch Script (Windows):**
```bash
start_parallel.bat
```

All methods start both processes and manage them together. The shell/batch scripts open processes in separate windows/terminals for better visibility.

## How Auto-Sync Works

1. **CSV File Monitoring**: 
   - Watches for `aviator_payouts_*.csv` files
   - Detects when file size or modification time changes
   - Checks every 2 seconds

2. **Auto-Training (Instant Sync)**:
   - When new data is detected and file has ≥30 records
   - Model automatically retrains with ALL available data (instant sync)
   - Retrains when ≥5 new records or 5% increase in data
   - Uses ensemble model for better accuracy
   - No manual intervention needed

3. **Auto-Prediction**:
   - After training, automatically generates new prediction
   - Updates every time new data arrives
   - Web interface refreshes every 3 seconds

4. **Real-time Web Updates**:
   - Frontend polls `/api/status` endpoint every 3 seconds
   - Displays latest prediction, model accuracy, and statistics
   - No page refresh needed

## API Endpoints

- `GET /` - Web interface
- `GET /api/load_csv` - Load CSV data
- `POST /api/train` - Manually train model
- `GET /api/predict` - Get prediction
- `GET /api/status` - Get current status (for auto-refresh)
- All endpoints use instant data sync (100% of available data)

## Status Indicators

- 🟢 **Auto-Sync Active**: System is monitoring and updating
- 🤖 **Model Trained**: Ready for predictions
- 🔄 **Auto-Training**: Model is retraining in background
- 📊 **Data Synced**: Latest CSV data loaded

## Troubleshooting

### Model Not Training Automatically
- Ensure CSV has at least 30 records for initial training
- Model retrains automatically with ≥5 new records
- Check that `auto_train_enabled = True` in code
- Verify file watcher is running (check console logs)
- Model uses ALL data (instant sync) - not a random subset

### Predictions Not Updating
- Check browser console for errors
- Verify `/api/status` endpoint is accessible
- Ensure model is trained (check model info card)

### CSV Not Detected
- Verify CSV files are in same directory as scripts
- Check file naming: `aviator_payouts_YYYYMMDD_HHMMSS.csv`
- Ensure file permissions allow reading

## Console Output

You'll see messages like:
```
🔄 New data detected in aviator_payouts_20251210_193235.csv (405 records)
🤖 [AUTO-TRAIN] Model retrained! MAE: 2.345, RMSE: 3.678, Records: 405
📝 CSV file updated: ./aviator_payouts_20251210_193235.csv
```

## Notes

- The predictor app runs on port 5001 by default
- Both processes can run simultaneously without conflicts
- CSV files are read-only (main.py writes, predictor reads)
- Model training happens in background thread
- Web interface auto-refreshes every 3 seconds
