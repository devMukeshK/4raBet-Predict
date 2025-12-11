from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import glob
import threading
import time
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost and LightGBM for better performance
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    XGBOOST_AVAILABLE = False
    print(f"⚠️ XGBoost not available: {str(e)[:100]}")
    print("   Install with: pip install xgboost")
    print("   On macOS, you may also need: brew install libomp")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, Exception) as e:
    LIGHTGBM_AVAILABLE = False
    print(f"⚠️ LightGBM not available: {str(e)[:100]}")
    print("   Install with: pip install lightgbm")
import joblib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

app = Flask(__name__)

# Global model storage
trained_model = None
scaler = None
feature_selector = None  # Store feature selector for prediction
model_accuracy = None
target_transform_func = None  # For target transformation
target_inverse_func = None  # For inverse transformation
last_training_file = None
last_csv_size = 0
last_csv_mtime = 0
last_training_record_count = 0
last_retrain_record_count = 0  # Track when model was last retrained
retrain_interval = 10  # Retrain model every 10 new records (for faster predictions)
current_prediction = None
prediction_history = []  # Store last 10 predictions
last_prediction_multiplier = None  # Track last multiplier used for prediction update (prevents updates during flight)
auto_train_enabled = True
training_lock = threading.Lock()
all_training_data = None  # Store all data for instant sync

# ============================================================================
# GAMBLING APPLICATION - PROFIT/LOSS SYSTEM
# ============================================================================
# HOW THE APPLICATION WORKS:
#
# 1. DATA COLLECTION (main.py):
#    - Continuously monitors Aviator game for new payout multipliers
#    - Saves each multiplier with timestamp to CSV file
#    - Example: "2025-12-10 23:45:17, 2.5x"
#
# 2. MODEL TRAINING (train_model):
#    - Loads historical data from CSV
#    - Extracts 70+ time-series features (rolling stats, trends, patterns)
#    - Trains ensemble ML model (XGBoost, LightGBM, RandomForest, etc.)
#    - Model learns patterns to predict next multiplier
#
# 3. PREDICTION (predict_next):
#    - Uses trained model to predict next multiplier range
#    - Example: "2.3x - 3.8x" (min_range = 2.3x, max_range = 3.8x)
#    - Stores prediction in current_prediction
#
# 4. BETTING DECISION (simulate_bet):
#    - RULE 1: If pred_min < 2.0 → NO BET (skip this round)
#    - RULE 2: If pred_min > 2.0 → PLACE BET (₹100)
#
# 5. ACTUAL RESULT ARRIVES (check_and_process_bet):
#    - New multiplier appears in CSV (e.g., 3.2x)
#    - Compares actual vs predicted minimum range
#    - RULE 3: If actual >= pred_min → WIN → Profit = pred_min × 100
#    - RULE 4: If actual < pred_min → LOSS → Loss = 100
#
# 6. PROFIT/LOSS CALCULATION:
#    WIN Example:
#    - pred_min = 2.5x, actual = 3.2x, bet = ₹100
#    - Profit = 2.5 × 100 = ₹250 (displayed)
#    - Wallet: balance - ₹100 (bet) + ₹250 (return) = balance + ₹150 (net)
#
#    LOSS Example:
#    - pred_min = 2.5x, actual = 1.8x, bet = ₹100
#    - Loss = ₹100
#    - Wallet: balance - ₹100 (bet) = balance - ₹100 (net)
#
# 7. WALLET UPDATE:
#    - Starting balance: ₹50,000
#    - Each bet updates current_balance
#    - Total P/L = current_balance - initial_balance (₹50,000)
#    - All bets recorded in betting_history for tracking
#
# ============================================================================

# Betting simulation storage
betting_history = []  # List of bets: {timestamp, bet_amount, predicted_range, actual_multiplier, profit_loss, balance}
current_balance = 50000.0  # Starting balance (max amount)
base_bet_amount = 100.0  # Base bet amount
max_balance = 50000.0  # Maximum balance
betting_enabled = True
last_bet_timestamp = None
last_bet_record_count = 0  # Track which record we last bet on
processed_multipliers = set()  # Track processed (timestamp, multiplier) pairs

# Separate tracking for minimum range and prediction value bets
min_range_bets = []  # Bets based on minimum range value
prediction_bets = []  # Bets based on prediction value
min_range_balance = 50000.0  # Balance for minimum range bets
prediction_balance = 50000.0  # Balance for prediction value bets

def load_latest_csv():
    """Load the most recent CSV file from the directory."""
    csv_files = glob.glob("aviator_payouts_*.csv")
    if not csv_files:
        return None, None
    latest_file = max(csv_files, key=os.path.getctime)
    try:
        df = pd.read_csv(latest_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        # Validate multipliers: must be >= 1.0 (never negative, minimum is 1.0)
        df['multiplier'] = pd.to_numeric(df['multiplier'], errors='coerce')
        df = df[df['multiplier'] >= 1.0]  # Filter out invalid values (< 1.0)
        df = df.dropna().reset_index(drop=True)
        return df, latest_file
    except Exception as e:
        return None, str(e)

def check_csv_updated():
    """Check if CSV file has been updated - ALWAYS detect new records."""
    global last_csv_size, last_csv_mtime, last_training_record_count
    df, file_info = load_latest_csv()
    if df is None or file_info is None:
        return False, None, None
    
    try:
        current_size = os.path.getsize(file_info)
        current_mtime = os.path.getmtime(file_info)
        current_records = len(df)
        
        # Always check if record count increased (most reliable indicator)
        records_increased = current_records > last_training_record_count
        
        # Also check file modification (for immediate detection)
        file_changed = (current_size != last_csv_size or current_mtime != last_csv_mtime)
        
        # If records increased OR file changed, it's an update
        if records_increased or file_changed:
            # Update tracking immediately
            last_csv_size = current_size
            last_csv_mtime = current_mtime
            # Note: last_training_record_count will be updated after processing
            return True, df, file_info
        
        return False, df, file_info
    except Exception as e:
        print(f"⚠️ Error checking CSV update: {e}")
        return False, df, file_info

def handle_outliers(df, multiplier_col='multiplier', method='winsorize', percentile_low=1, percentile_high=99):
    """Handle outliers in multiplier data."""
    df = df.copy()
    multipliers = df[multiplier_col].values
    
    if method == 'winsorize':
        # Cap extreme values at percentiles
        low_threshold = np.percentile(multipliers, percentile_low)
        high_threshold = np.percentile(multipliers, percentile_high)
        df[multiplier_col] = df[multiplier_col].clip(lower=low_threshold, upper=high_threshold)
    elif method == 'log_transform':
        # Log transform to reduce impact of extreme values
        df[multiplier_col] = np.log1p(df[multiplier_col])
    
    return df

def prepare_features(df, lookback=20, use_all_data=True):
    """Prepare SUPER ADVANCED features that analyze ENTIRE dataset patterns, not just recent values."""
    if len(df) < lookback + 1:
        return None, None
    
    # Ensure data is clean and sorted
    df = df.copy()
    df['multiplier'] = pd.to_numeric(df['multiplier'], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Handle outliers - cap extreme values at 99th percentile
    df = handle_outliers(df, method='winsorize', percentile_low=1, percentile_high=99)
    
    if len(df) < lookback + 1:
        return None, None
    
    # Extract time components for better temporal features
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['second'] = df['timestamp'].dt.second
    df['time_of_day'] = df['hour'] * 3600 + df['minute'] * 60 + df['second']
    
    # ===== COMPUTE GLOBAL STATISTICS FROM ENTIRE DATASET =====
    # These are computed once and used for all predictions
    global_mean = df['multiplier'].mean()
    global_std = df['multiplier'].std()
    global_median = df['multiplier'].median()
    global_min = df['multiplier'].min()
    global_max = df['multiplier'].max()
    global_p25 = df['multiplier'].quantile(0.25)
    global_p75 = df['multiplier'].quantile(0.75)
    global_p90 = df['multiplier'].quantile(0.90)
    global_p95 = df['multiplier'].quantile(0.95)
    
    # Distribution analysis from entire dataset
    global_skew = df['multiplier'].skew()
    global_kurtosis = df['multiplier'].kurtosis()
    
    # Pattern analysis from entire dataset
    high_count_global = (df['multiplier'] > 5.0).sum()
    very_high_count_global = (df['multiplier'] > 10.0).sum()
    low_count_global = (df['multiplier'] < 1.5).sum()
    extreme_count_global = (df['multiplier'] > 50.0).sum()
    
    # Long-term trend from entire dataset
    if len(df) > 50:
        # Fit polynomial trend to entire dataset
        x_indices = np.arange(len(df))
        global_trend_coef = np.polyfit(x_indices, df['multiplier'].values, 2)  # Quadratic trend
        global_trend_slope = global_trend_coef[1]  # Linear component
        global_trend_curve = global_trend_coef[0]  # Quadratic component
    else:
        global_trend_slope = 0
        global_trend_curve = 0
    
    # Cyclical pattern detection (if we have enough data)
    if len(df) > 100:
        # Autocorrelation at different lags across entire dataset
        autocorr_lag5 = df['multiplier'].autocorr(lag=5) if len(df) > 5 else 0
        autocorr_lag10 = df['multiplier'].autocorr(lag=10) if len(df) > 10 else 0
        autocorr_lag20 = df['multiplier'].autocorr(lag=20) if len(df) > 20 else 0
    else:
        autocorr_lag5 = 0
        autocorr_lag10 = 0
        autocorr_lag20 = 0
    
    # Frequency analysis - how often do patterns repeat?
    multiplier_values = df['multiplier'].values
    if len(multiplier_values) > 50:
        # Count transitions between high/low states
        high_low_transitions = sum(1 for i in range(1, len(multiplier_values)) 
                                  if (multiplier_values[i-1] < 2.0 and multiplier_values[i] > 5.0) or
                                     (multiplier_values[i-1] > 5.0 and multiplier_values[i] < 2.0))
        transition_rate = high_low_transitions / len(multiplier_values)
    else:
        transition_rate = 0
    
    features = []
    targets = []
    
    # Use ALL data for training, not just recent
    start_idx = lookback if use_all_data else max(lookback, len(df) - 100)
    
    # Pre-compute rolling statistics for efficiency
    df['mult_rolling_mean_3'] = df['multiplier'].rolling(window=3, min_periods=1).mean()
    df['mult_rolling_mean_5'] = df['multiplier'].rolling(window=5, min_periods=1).mean()
    df['mult_rolling_mean_10'] = df['multiplier'].rolling(window=10, min_periods=1).mean()
    df['mult_rolling_mean_20'] = df['multiplier'].rolling(window=20, min_periods=1).mean()
    df['mult_rolling_std_5'] = df['multiplier'].rolling(window=5, min_periods=1).std()
    df['mult_rolling_std_10'] = df['multiplier'].rolling(window=10, min_periods=1).std()
    df['mult_rolling_std_20'] = df['multiplier'].rolling(window=20, min_periods=1).std()
    df['mult_rolling_min_5'] = df['multiplier'].rolling(window=5, min_periods=1).min()
    df['mult_rolling_max_5'] = df['multiplier'].rolling(window=5, min_periods=1).max()
    
    # Exponential moving averages
    df['mult_ema_3'] = df['multiplier'].ewm(span=3, adjust=False).mean()
    df['mult_ema_5'] = df['multiplier'].ewm(span=5, adjust=False).mean()
    df['mult_ema_10'] = df['multiplier'].ewm(span=10, adjust=False).mean()
    df['mult_ema_20'] = df['multiplier'].ewm(span=20, adjust=False).mean()
    
    for i in range(start_idx, len(df)):
        window = df['multiplier'].iloc[i-lookback:i].values
        current = df['multiplier'].iloc[i]
        
        # Basic statistics
        mean_val = np.mean(window)
        std_val = np.std(window) if len(window) > 1 else 0
        min_val = np.min(window)
        max_val = np.max(window)
        median_val = np.median(window)
        
        # Time-based features (extract from timestamp)
        current_time = df['timestamp'].iloc[i]
        time_diff = (current_time - df['timestamp'].iloc[i-1]).total_seconds() if i > 0 else 0
        hour = df['hour'].iloc[i]
        minute = df['minute'].iloc[i]
        time_of_day = df['time_of_day'].iloc[i]
        
        # Time since last high/low
        recent_highs = [j for j in range(max(0, i-10), i) if df['multiplier'].iloc[j] > 5.0]
        time_since_high = (i - recent_highs[-1]) if recent_highs else 20
        recent_lows = [j for j in range(max(0, i-10), i) if df['multiplier'].iloc[j] < 1.5]
        time_since_low = (i - recent_lows[-1]) if recent_lows else 20
        
        # Use pre-computed rolling statistics
        rolling_mean_3 = df['mult_rolling_mean_3'].iloc[i-1] if i > 0 else mean_val
        rolling_mean_5 = df['mult_rolling_mean_5'].iloc[i-1] if i > 0 else mean_val
        rolling_mean_10 = df['mult_rolling_mean_10'].iloc[i-1] if i > 0 else mean_val
        rolling_mean_20 = df['mult_rolling_mean_20'].iloc[i-1] if i > 0 else mean_val
        rolling_std_5 = df['mult_rolling_std_5'].iloc[i-1] if i > 0 else std_val
        rolling_std_10 = df['mult_rolling_std_10'].iloc[i-1] if i > 0 else std_val
        rolling_std_20 = df['mult_rolling_std_20'].iloc[i-1] if i > 0 else std_val
        rolling_min_5 = df['mult_rolling_min_5'].iloc[i-1] if i > 0 else min_val
        rolling_max_5 = df['mult_rolling_max_5'].iloc[i-1] if i > 0 else max_val
        
        # Exponential moving averages
        ema_3 = df['mult_ema_3'].iloc[i-1] if i > 0 else mean_val
        ema_5 = df['mult_ema_5'].iloc[i-1] if i > 0 else mean_val
        ema_10 = df['mult_ema_10'].iloc[i-1] if i > 0 else mean_val
        ema_20 = df['mult_ema_20'].iloc[i-1] if i > 0 else mean_val
        
        # Sequence features (last N values) - DEFINE FIRST before using in global comparison
        last_1 = window[-1] if len(window) > 0 else mean_val
        last_2 = window[-2] if len(window) > 1 else last_1
        last_3 = window[-3] if len(window) > 2 else last_2
        last_5_mean = np.mean(window[-5:]) if len(window) >= 5 else mean_val
        last_10_mean = np.mean(window[-10:]) if len(window) >= 10 else mean_val
        
        # ===== COMPARE CURRENT WINDOW TO GLOBAL DATASET =====
        # How does current window compare to entire dataset?
        window_vs_global_mean = mean_val / global_mean if global_mean > 0 else 1
        window_vs_global_std = std_val / global_std if global_std > 0 else 1
        last_vs_global_mean = last_1 / global_mean if global_mean > 0 else 1
        last_vs_global_median = last_1 / global_median if global_median > 0 else 1
        
        # Position in global distribution
        percentile_in_global = (df['multiplier'].iloc[:i+1] < last_1).sum() / (i + 1) if i > 0 else 0.5
        
        # Distance from global statistics
        dist_from_global_mean = last_1 - global_mean
        dist_from_global_median = last_1 - global_median
        z_score_global = (last_1 - global_mean) / global_std if global_std > 0 else 0
        
        # Trend features (polynomial fit)
        recent_5 = window[-5:] if len(window) >= 5 else window
        recent_10 = window[-10:] if len(window) >= 10 else window
        trend_5 = np.polyfit(range(len(recent_5)), recent_5, 1)[0] if len(recent_5) > 1 else 0
        trend_10 = np.polyfit(range(len(recent_10)), recent_10, 1)[0] if len(recent_10) > 1 else 0
        
        # Volatility features
        volatility_5 = np.std(window[-5:]) if len(window) >= 5 else std_val
        volatility_10 = np.std(window[-10:]) if len(window) >= 10 else std_val
        
        # Pattern features
        high_ratio = len([x for x in window if x > 2.0]) / len(window)
        very_high_ratio = len([x for x in window if x > 10.0]) / len(window)
        low_ratio = len([x for x in window if x < 1.5]) / len(window)
        extreme_high_ratio = len([x for x in window if x > 50.0]) / len(window)
        
        # Note: last_1, last_2, etc. are already defined above before global comparison
        
        # Momentum and acceleration
        momentum_1 = last_1 - last_2 if len(window) > 1 else 0
        momentum_2 = last_2 - last_3 if len(window) > 2 else 0
        momentum_3 = last_3 - (window[-4] if len(window) > 3 else last_3)
        acceleration = momentum_1 - momentum_2
        
        # Rate of change features
        roc_1 = (last_1 - last_2) / last_2 if len(window) > 1 and last_2 > 0 else 0
        roc_2 = (last_2 - last_3) / last_3 if len(window) > 2 and last_3 > 0 else 0
        roc_5 = (last_1 - window[-6]) / window[-6] if len(window) > 5 and window[-6] > 0 else 0
        
        # Autocorrelation features
        if len(window) >= 5:
            autocorr_1 = np.corrcoef(window[:-1], window[1:])[0, 1] if len(window) > 1 and not np.isnan(np.corrcoef(window[:-1], window[1:])[0, 1]) else 0
        else:
            autocorr_1 = 0
        
        # Range and spread features
        range_val = max_val - min_val
        range_ratio = range_val / mean_val if mean_val > 0 else 0
        cv = std_val / mean_val if mean_val > 0 else 0  # Coefficient of variation
        
        # Percentile features
        p10 = np.percentile(window, 10)
        p25 = np.percentile(window, 25)
        p50 = np.percentile(window, 50)
        p75 = np.percentile(window, 75)
        p90 = np.percentile(window, 90)
        iqr = p75 - p25
        
        # MA ratios and crossovers
        ma_ratio_3_5 = rolling_mean_3 / rolling_mean_5 if rolling_mean_5 > 0 else 1
        ma_ratio_5_10 = rolling_mean_5 / rolling_mean_10 if rolling_mean_10 > 0 else 1
        ema_ratio_3_5 = ema_3 / ema_5 if ema_5 > 0 else 1
        
        # Distance from moving averages
        dist_from_ma3 = last_1 - rolling_mean_3
        dist_from_ma5 = last_1 - rolling_mean_5
        dist_from_ma10 = last_1 - rolling_mean_10
        dist_from_ema3 = last_1 - ema_3
        
        # Z-score features
        z_score_5 = (last_1 - rolling_mean_5) / rolling_std_5 if rolling_std_5 > 0 else 0
        z_score_10 = (last_1 - rolling_mean_10) / rolling_std_10 if rolling_std_10 > 0 else 0
        
        # Feature vector (ANALYZES ENTIRE DATASET + LOCAL PATTERNS - 70+ features)
        feature_row = [
            # ===== GLOBAL DATASET FEATURES (learns from entire history) =====
            # Global statistics (10)
            global_mean, global_std, global_median, global_min, global_max,
            global_p25, global_p75, global_p90, global_p95,
            global_skew,  # Distribution shape
            # Global patterns (4)
            high_count_global / len(df), very_high_count_global / len(df),
            low_count_global / len(df), extreme_count_global / len(df),
            # Global trends (2)
            global_trend_slope, global_trend_curve,
            # Global autocorrelations (3)
            autocorr_lag5, autocorr_lag10, autocorr_lag20,
            # Global transition patterns (1)
            transition_rate,
            # ===== LOCAL WINDOW FEATURES (recent patterns) =====
            # Basic statistics (5)
            mean_val, std_val, min_val, max_val, median_val,
            # Last values (5)
            last_1, last_2, last_3, last_5_mean, last_10_mean,
            # Rolling means (4)
            rolling_mean_3, rolling_mean_5, rolling_mean_10, rolling_mean_20,
            # Rolling std (3)
            rolling_std_5, rolling_std_10, rolling_std_20,
            # Rolling min/max (2)
            rolling_min_5, rolling_max_5,
            # EMA (4)
            ema_3, ema_5, ema_10, ema_20,
            # Trends (2)
            trend_5, trend_10,
            # Volatility (2)
            volatility_5, volatility_10,
            # Patterns (4)
            high_ratio, very_high_ratio, low_ratio, extreme_high_ratio,
            # Momentum (4)
            momentum_1, momentum_2, momentum_3, acceleration,
            # Rate of change (3)
            roc_1, roc_2, roc_5,
            # Range features (3)
            range_val, range_ratio, cv,
            # Percentiles (6)
            p10, p25, p50, p75, p90, iqr,
            # MA ratios (3)
            ma_ratio_3_5, ma_ratio_5_10, ema_ratio_3_5,
            # Distance from MA (4)
            dist_from_ma3, dist_from_ma5, dist_from_ma10, dist_from_ema3,
            # Z-scores (2)
            z_score_5, z_score_10,
            # ===== COMPARISON: LOCAL vs GLOBAL (learns relationships) =====
            # Window vs Global (4)
            window_vs_global_mean, window_vs_global_std,
            last_vs_global_mean, last_vs_global_median,
            # Position in global distribution (1)
            percentile_in_global,
            # Distance from global (3)
            dist_from_global_mean, dist_from_global_median, z_score_global,
            # Time features (5)
            time_diff, hour, minute, time_of_day, time_since_high,
            # Autocorrelation (1)
            autocorr_1,
        ]
        
        features.append(feature_row)
        targets.append(current)
    
    return np.array(features), np.array(targets)

def train_model(df, auto=False):
    """Train an ensemble model on ALL available data with advanced features."""
    global trained_model, scaler, feature_selector, model_accuracy, last_training_file, last_training_record_count, all_training_data
    
    with training_lock:
        # Store all data for instant sync
        all_training_data = df.copy()
        
        # IMPROVED: Use longer lookback for better pattern recognition (30 instead of 25)
        features, targets = prepare_features(df, lookback=30, use_all_data=True)
        
        if features is None or len(features) < 15:
            return False, f"Not enough data. Need at least 30 records for training. Current: {len(df)}"
        
        print(f"📊 Training on {len(features)} samples from {len(df)} total records...")
        
        # IMPROVED: Better target handling - use winsorization instead of log transform
        # Log transform can cause issues with evaluation metrics
        targets_array = np.array(targets)
        
        # Use winsorization (cap extreme values) - more stable than log transform
        target_q99 = np.percentile(targets_array, 99)
        target_q01 = np.percentile(targets_array, 1)
        targets_capped = np.clip(targets_array, target_q01, target_q99)
        
        # Store transform functions (identity for winsorization)
        global target_transform_func, target_inverse_func
        target_transform_func = lambda x: np.clip(x, np.percentile(x, 1), np.percentile(x, 99))
        target_inverse_func = lambda x: x  # No inverse needed for clipping
        
        print(f"📊 Target range: [{np.min(targets_capped):.2f}, {np.max(targets_capped):.2f}] (winsorized at 1st-99th percentile)")
        
        # IMPROVED: Use RobustScaler (more stable than PowerTransformer for this use case)
        # PowerTransformer can sometimes cause issues with prediction
        global scaler
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features)
        print("✅ Using RobustScaler for feature scaling (robust to outliers)")
        
        # IMPROVED: Better feature selection - use mutual information for non-linear relationships
        global feature_selector
        from sklearn.feature_selection import mutual_info_regression
        
        if len(features_scaled) > 50 and features_scaled.shape[1] > 30:
            # Use mutual information for better feature selection (captures non-linear relationships)
            try:
                # Select top features using mutual information (better for non-linear patterns)
                k_features = min(70, features_scaled.shape[1])  # Use more features for better accuracy
                mi_scores = mutual_info_regression(features_scaled, targets_capped, random_state=42, n_neighbors=3)
                top_indices = np.argsort(mi_scores)[-k_features:]
                
                # Create a proper feature selector object
                class CustomFeatureSelector:
                    def __init__(self, indices):
                        self.indices = indices
                    def transform(self, X):
                        return X[:, self.indices]
                
                feature_selector = CustomFeatureSelector(top_indices)
                features_scaled = features_scaled[:, top_indices]
                print(f"✅ Selected {k_features} best features using Mutual Information (captures non-linear patterns)")
            except Exception as e:
                # Fallback to SelectKBest
                k_features = min(70, features_scaled.shape[1])
                feature_selector = SelectKBest(f_regression, k=k_features)
                features_scaled = feature_selector.fit_transform(features_scaled, targets_capped)
                print(f"✅ Selected {features_scaled.shape[1]} best features using F-regression (fallback)")
        else:
            feature_selector = None
            print(f"📊 Using all {features_scaled.shape[1]} features")
        
        # Use time-series split for validation (no shuffling for time series)
        if len(features) > 50:
            # Use last 20% for testing
            split_idx = int(len(features) * 0.8)
            X_train, X_test = features_scaled[:split_idx], features_scaled[split_idx:]
            y_train, y_test = targets_capped[:split_idx], targets_capped[split_idx:]
        else:
            # If very little data, use all for training
            X_train, X_test = features_scaled, features_scaled[-5:] if len(features) > 5 else features_scaled
            y_train, y_test = targets_capped, targets_capped[-5:] if len(targets) > 5 else targets_capped
        
        # Create SUPER ADVANCED ensemble model - prioritize XGBoost/LightGBM
        models = []
        weights = []
        
        # IMPROVED: XGBoost with early stopping and better hyperparameters
        if XGBOOST_AVAILABLE:
            try:
                # Split for early stopping
                if len(X_train) > 100:
                    X_train_fit, X_val = X_train[:-50], X_train[-50:]
                    y_train_fit, y_val = y_train[:-50], y_train[-50:]
                else:
                    X_train_fit, X_val = X_train, X_train
                    y_train_fit, y_val = y_train, y_train
                
                xgb_model = xgb.XGBRegressor(
                    n_estimators=1000,  # More trees with early stopping
                    max_depth=10,  # Deeper trees for complex patterns
                    learning_rate=0.01,  # Lower learning rate for better convergence
                    subsample=0.85,
                    colsample_bytree=0.85,
                    colsample_bylevel=0.85,
                    min_child_weight=5,
                    gamma=0.2,
                    reg_alpha=0.2,  # L1 regularization
                    reg_lambda=2.0,  # L2 regularization
                    random_state=42,
                    n_jobs=-1,
                    tree_method='hist',
                    objective='reg:squarederror',
                    eval_metric='mae'
                )
                xgb_model.fit(
                    X_train_fit, y_train_fit,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
                models.append(('xgb', xgb_model))
                weights.append(4)  # Highest weight
                print("✅ Using XGBoost with early stopping (optimized)")
            except Exception as e:
                print(f"⚠️ XGBoost model creation failed: {str(e)[:100]}")
        
        # IMPROVED: LightGBM with early stopping and better hyperparameters
        if LIGHTGBM_AVAILABLE:
            try:
                # Split for early stopping
                if len(X_train) > 100:
                    X_train_fit, X_val = X_train[:-50], X_train[-50:]
                    y_train_fit, y_val = y_train[:-50], y_train[-50:]
                else:
                    X_train_fit, X_val = X_train, X_train
                    y_train_fit, y_val = y_train, y_train
                
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=1000,  # More trees with early stopping
                    max_depth=10,
                    learning_rate=0.01,  # Lower learning rate
                    num_leaves=50,  # More leaves for complex patterns
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_samples=30,
                    reg_alpha=0.2,
                    reg_lambda=2.0,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                    objective='regression',
                    metric='mae'
                )
                lgb_model.fit(
                    X_train_fit, y_train_fit,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
                models.append(('lgb', lgb_model))
                weights.append(4)  # Highest weight
                print("✅ Using LightGBM with early stopping (optimized)")
            except Exception as e:
                print(f"⚠️ LightGBM model creation failed: {str(e)[:100]}")
        
        # Ensure we have at least some models
        if len(models) == 0:
            print("⚠️ No advanced models available, using standard ensemble...")
        
        # IMPROVED: Random Forest with better hyperparameters
        rf_model = RandomForestRegressor(
            n_estimators=800,  # More trees
            max_depth=20,  # Deeper trees
            min_samples_split=4,  # More splits
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            oob_score=True,
            bootstrap=True
        )
        models.append(('rf', rf_model))
        weights.append(3 if len(models) == 1 else 2)
        
        # IMPROVED: Gradient Boosting with better hyperparameters
        gb_model = GradientBoostingRegressor(
            n_estimators=600,  # More trees
            max_depth=10,  # Deeper trees
            learning_rate=0.02,  # Lower learning rate
            min_samples_split=4,
            min_samples_leaf=1,
            subsample=0.9,
            random_state=42,
            validation_fraction=0.1,
            n_iter_no_change=30,  # More patience
            tol=1e-4
        )
        models.append(('gb', gb_model))
        weights.append(3 if len(models) == 2 else 2)
        
        # IMPROVED: Extra Trees with better hyperparameters
        et_model = ExtraTreesRegressor(
            n_estimators=600,  # More trees
            max_depth=20,  # Deeper trees
            min_samples_split=4,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            bootstrap=True
        )
        models.append(('et', et_model))
        weights.append(2)
        
        # Regularized linear models (for diversity)
        ridge_model = Ridge(alpha=0.3, random_state=42)
        models.append(('ridge', ridge_model))
        weights.append(1)
        
        # ElasticNet for additional regularization
        elastic_model = ElasticNet(alpha=0.3, l1_ratio=0.5, random_state=42, max_iter=2000)
        models.append(('elastic', elastic_model))
        weights.append(1)
        
        # Ensemble model (weighted voting regressor)
        ensemble_model = VotingRegressor(models, weights=weights)
        
        # Train ensemble
        model_names = [name for name, _ in models]
        print(f"🔄 Training SUPER ADVANCED ensemble model: {', '.join(model_names)}...")
        print(f"   Using {len(X_train)} training samples, {len(X_test)} test samples")
        print(f"   Features: {X_train.shape[1]}, Target range: [{np.min(y_train):.2f}, {np.max(y_train):.2f}]")
        
        # Train ensemble (this fits all individual models)
        ensemble_model.fit(X_train, y_train)
        
        # IMPROVED: Evaluate individual models AFTER ensemble is fitted
        # VotingRegressor fits models in place, access them via named_estimators_
        model_performances = {}
        
        # Get fitted models from the ensemble (VotingRegressor stores them in named_estimators_)
        if hasattr(ensemble_model, 'named_estimators_'):
            fitted_models = ensemble_model.named_estimators_
            for name, _ in models:
                if name in fitted_models:
                    try:
                        fitted_model = fitted_models[name]
                        y_pred_model = fitted_model.predict(X_test)
                        y_pred_model = np.maximum(1.0, y_pred_model)
                        mae_model = mean_absolute_error(y_test, y_pred_model)
                        model_performances[name] = mae_model
                    except Exception as e:
                        print(f"⚠️ Error evaluating {name}: {e}")
                        model_performances[name] = 999
                else:
                    model_performances[name] = 999
        else:
            # Fallback: Skip individual model evaluation, use default weights
            print("⚠️ Cannot access individual models for performance evaluation, using default weights")
            for name, _ in models:
                model_performances[name] = 999  # Skip performance-based weighting
        
        # Update weights based on performance (lower MAE = higher weight)
        if model_performances and len([m for m in model_performances.values() if m < 999]) > 1:
            valid_performances = {k: v for k, v in model_performances.items() if v < 999}
            if valid_performances:
                max_mae = max(valid_performances.values())
                min_mae = min(valid_performances.values())
                if max_mae > min_mae:
                    # Normalize weights: better performance = higher weight
                    performance_weights = []
                    for name, _ in models:
                        if name in valid_performances:
                            mae = valid_performances[name]
                            weight = (max_mae - mae) / (max_mae - min_mae) * 3 + 1
                            performance_weights.append(weight)
                        else:
                            performance_weights.append(1)  # Default weight for failed models
                    weights = performance_weights
                    print(f"📊 Performance-based weights: {dict(zip([name for name, _ in models], [f'{w:.2f}' for w in weights]))}")
        
        # Evaluate ensemble on test set
        y_pred = ensemble_model.predict(X_test)
        # No inverse transform needed (winsorization doesn't require inverse)
        # Enforce constraint: predictions must be >= 1.0
        y_pred = np.maximum(1.0, y_pred)
        y_test_clipped = np.maximum(1.0, y_test)  # Test targets are already winsorized
        
        mae = mean_absolute_error(y_test_clipped, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_clipped, y_pred))
        r2 = r2_score(y_test_clipped, y_pred)
        
        # Time-series cross-validation for better accuracy estimate
        if len(X_train) > 50:
            tscv = TimeSeriesSplit(n_splits=min(5, len(X_train)//20))
            cv_scores = cross_val_score(ensemble_model, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            cv_std = cv_scores.std()
            print(f"📊 Cross-validation MAE: {cv_mae:.3f} (+/- {cv_std:.3f})")
        else:
            cv_mae = mae
            cv_std = 0
        
        trained_model = ensemble_model
        model_accuracy = {
            'mae': round(mae, 3),
            'rmse': round(rmse, 3),
            'r2_score': round(r2, 3),
            'cv_mae': round(cv_mae, 3),
            'cv_std': round(cv_std, 3) if 'cv_std' in locals() else 0,
            'samples_trained': len(X_train),
            'samples_tested': len(X_test),
            'total_records': len(df),
            'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Print detailed diagnostics
        print(f"📈 Model Performance:")
        print(f"   MAE: {mae:.3f} (lower is better)")
        print(f"   RMSE: {rmse:.3f} (lower is better)")
        print(f"   R² Score: {r2:.3f} (higher is better, 1.0 is perfect)")
        if r2 < 0:
            print(f"   ⚠️  Negative R² indicates model performs worse than baseline mean prediction")
        print(f"   CV MAE: {cv_mae:.3f} (+/- {cv_std:.3f})")
        last_training_file = df
        last_training_record_count = len(df)
        
        if auto:
            print(f"✅ [AUTO-TRAIN] Ensemble model trained! MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}, Records: {len(df)}")
        else:
            print(f"✅ Ensemble model trained! MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
        
        return True, f"Ensemble model trained successfully! MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}"

def predict_next(df):
    """Predict the next multiplier value using trained ensemble model with ALL data."""
    global trained_model, scaler, feature_selector, current_prediction, all_training_data, prediction_history
    
    if trained_model is None:
        return None, "Model not trained yet. Please train the model first."
    
    if scaler is None:
        return None, "Scaler not initialized. Please retrain the model."
    
    # Check if scaler has been fitted
    # Different scalers have different fitted attributes:
    # - RobustScaler: 'center_'
    # - StandardScaler: 'mean_'
    # - PowerTransformer: 'lambdas_'
    try:
        # Try to check if scaler is fitted by accessing a fitted attribute
        if hasattr(scaler, 'center_'):
            _ = scaler.center_  # RobustScaler
        elif hasattr(scaler, 'mean_'):
            _ = scaler.mean_  # StandardScaler
        elif hasattr(scaler, 'lambdas_'):
            _ = scaler.lambdas_  # PowerTransformer
        else:
            # If no known attribute exists, try to transform a dummy value
            try:
                scaler.transform([[0] * features.shape[1] if len(features) > 0 else [0] * 10])
            except:
                return None, "Scaler not fitted yet. Please retrain the model."
    except (AttributeError, ValueError, IndexError):
        return None, "Scaler not fitted yet. Please retrain the model."
    
    # Use all available data for prediction (instant sync)
    prediction_df = all_training_data if all_training_data is not None else df
    prediction_df = prediction_df.copy()
    prediction_df['multiplier'] = pd.to_numeric(prediction_df['multiplier'], errors='coerce')
    # Validate: multipliers must be >= 1.0 (never negative, minimum is 1.0)
    prediction_df = prediction_df[prediction_df['multiplier'] >= 1.0]
    prediction_df = prediction_df.dropna().reset_index(drop=True)
    prediction_df = prediction_df.sort_values('timestamp').reset_index(drop=True)
    
    # Use same lookback as training to ensure consistency (30 to match training)
    features, _ = prepare_features(prediction_df, lookback=30, use_all_data=True)
    
    if features is None or len(features) == 0:
        return None, "Not enough data for prediction."
    
    # Use the last window for prediction
    last_features = features[-1].reshape(1, -1)
    
    # Scale features - ensure scaler is fitted
    try:
        last_features_scaled = scaler.transform(last_features)
    except (AttributeError, ValueError) as e:
        return None, f"Scaler error: {str(e)}. Please retrain the model."
    
    # Apply feature selection if it was used during training
    global feature_selector
    if feature_selector is not None:
        try:
            last_features_scaled = feature_selector.transform(last_features_scaled)
        except:
            # If custom selector, use indices directly
            if hasattr(feature_selector, 'indices'):
                last_features_scaled = last_features_scaled[:, feature_selector.indices]
            else:
                last_features_scaled = feature_selector.transform(last_features_scaled)
    
    # Get prediction from ensemble
    prediction = trained_model.predict(last_features_scaled)[0]
    
    # No inverse transform needed (we use winsorization, not log transform)
    # Enforce constraint: multiplier values start from 1.0, never negative
    prediction = max(1.0, prediction)  # Minimum value is 1.0
    
    # Get predictions from individual models for confidence calculation
    individual_predictions = [prediction]  # Start with ensemble prediction
    if hasattr(trained_model, 'named_estimators_'):
        for name, estimator in trained_model.named_estimators_.items():
            try:
                # Use the same transformed features for individual models
                pred = estimator.predict(last_features_scaled)[0]
                # No inverse transform needed (winsorization doesn't require inverse)
                pred = max(1.0, pred)  # Enforce minimum 1.0 constraint
                individual_predictions.append(pred)
            except Exception as e:
                print(f"⚠️ Error getting prediction from {name}: {e}")
                pass
    
    # Calculate confidence based on agreement between models
    if len(individual_predictions) > 1:
        std_dev = np.std(individual_predictions)
        mean_pred = np.mean(individual_predictions)
        # Lower std = higher confidence (models agree)
        confidence = max(50, min(95, 100 - (std_dev / mean_pred * 100) if mean_pred > 0 else 70))
    else:
        # Use model accuracy for confidence
        if model_accuracy:
            confidence = max(60, min(90, 100 - (model_accuracy.get('mae', 2.0) / prediction * 100) if prediction > 0 else 75))
        else:
            confidence = 75
    
    # Calculate additional insights
    last_actual = float(prediction_df['multiplier'].iloc[-1])
    trend = 'increasing' if prediction > last_actual else 'decreasing'
    
    # Calculate TIGHT prediction range using improved method
    if len(individual_predictions) > 1:
        # Use percentiles instead of min/max for tighter range
        pred_sorted = sorted(individual_predictions)
        # Use 25th and 75th percentiles for tighter range (interquartile range)
        p25_idx = int(len(pred_sorted) * 0.25)
        p75_idx = int(len(pred_sorted) * 0.75)
        pred_min = max(1.0, pred_sorted[p25_idx])  # 25th percentile, min 1.0
        pred_max = pred_sorted[p75_idx]  # 75th percentile
        
        # Further tighten based on model agreement
        std_dev = np.std(individual_predictions)
        mean_pred = np.mean(individual_predictions)
        
        # If models agree well (low std), use tighter range
        if std_dev < mean_pred * 0.1:  # Less than 10% variation
            # Use 40th-60th percentile for very tight range
            p40_idx = int(len(pred_sorted) * 0.40)
            p60_idx = int(len(pred_sorted) * 0.60)
            pred_min = max(1.0, pred_sorted[p40_idx])
            pred_max = pred_sorted[p60_idx]
        elif std_dev < mean_pred * 0.2:  # Less than 20% variation
            # Use 30th-70th percentile for tight range
            p30_idx = int(len(pred_sorted) * 0.30)
            p70_idx = int(len(pred_sorted) * 0.70)
            pred_min = max(1.0, pred_sorted[p30_idx])
            pred_max = pred_sorted[p70_idx]
        
        # Ensure range is not too wide (max 30% of prediction)
        max_range = prediction * 0.30  # Maximum 30% range
        current_range = pred_max - pred_min
        if current_range > max_range:
            # Tighten to ±15% of prediction
            pred_min = max(1.0, prediction - max_range / 2)
            pred_max = prediction + max_range / 2
        
        pred_range = f"{pred_min:.2f} - {pred_max:.2f}"
    else:
        # Estimate range based on model accuracy with tighter margins
        if model_accuracy:
            # Use a fraction of MAE for tighter range (e.g., 50-70% of MAE)
            mae = model_accuracy.get('mae', 1.0)
            # Use smaller error margin for tighter range
            error_margin = mae * 0.5  # Use 50% of MAE instead of full MAE
            
            # Further tighten based on R² score (better model = tighter range)
            r2 = model_accuracy.get('r2_score', 0)
            if r2 > 0.3:  # Good model
                error_margin = mae * 0.4  # Even tighter (40% of MAE)
            elif r2 > 0.2:  # Decent model
                error_margin = mae * 0.5  # 50% of MAE
            else:  # Poor model
                error_margin = mae * 0.6  # 60% of MAE
            
            # Cap maximum range to 25% of prediction
            max_range = prediction * 0.25
            error_margin = min(error_margin, max_range / 2)
            
            pred_min = max(1.0, prediction - error_margin)
            pred_max = prediction + error_margin
            pred_range = f"{pred_min:.2f} - {pred_max:.2f}"
        else:
            # Very tight default range (±10% of prediction)
            error_margin = prediction * 0.10
            pred_min = max(1.0, prediction - error_margin)
            pred_max = prediction + error_margin
            pred_range = f"{pred_min:.2f} - {pred_max:.2f}"
    
    result = {
        'predicted_multiplier': round(prediction, 2),
        'prediction_range': pred_range,
        'confidence': round(confidence, 1),
        'last_actual': round(last_actual, 2),
        'trend': trend,
        'predicted_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_points_used': len(prediction_df)
    }
    
    current_prediction = result
    
    # Add to prediction history (keep last 10, avoid duplicates)
    # Only add if this is a new prediction (different from the last one)
    if not prediction_history or prediction_history[-1].get('predicted_multiplier') != result.get('predicted_multiplier') or prediction_history[-1].get('predicted_at') != result.get('predicted_at'):
        prediction_history.append(result.copy())
        if len(prediction_history) > 10:
            prediction_history.pop(0)  # Remove oldest prediction
    
    return result, None

def simulate_bet_min_range(pred_min, actual_multiplier, bet_amount, balance):
    """Simulate a bet based on minimum range value.
    
    EXACT RULES (as per user requirement):
    1. When pred_min < 2.0: Do NOT bet (handled in simulate_bet function)
    2. When pred_min > 2.0: Bet and calculate profit/loss
    3. When bet placed AND actual >= pred_min (WIN): Profit = pred_min × 100
    4. When bet placed AND actual < pred_min (LOSS): Loss = 100
    
    Examples:
    - pred_min = 2.5x, actual = 3.0x, bet = ₹100 → WIN → Profit = ₹250, Wallet: balance - 100 + 250 = balance + 150
    - pred_min = 2.5x, actual = 1.8x, bet = ₹100 → LOSS → Loss = ₹100, Wallet: balance - 100
    """
    # Rule 3 & 4: Check if win (actual must be >= pred_min)
    is_win = actual_multiplier >= pred_min
    
    if is_win:
        # Rule 3: WIN → Profit = pred_min × 100
        # Example: pred_min = 2.5x → Profit = 2.5 × 100 = ₹250
        profit_loss = pred_min * bet_amount
        
        # Wallet calculation: Deduct bet, then add return
        # balance - bet_amount + (pred_min × bet_amount)
        # = balance + (pred_min - 1) × bet_amount
        # Example: balance - 100 + 250 = balance + 150
        new_balance = balance - bet_amount + (pred_min * bet_amount)
    else:
        # Rule 4: LOSS → Loss = 100
        # Example: Loss = ₹100
        profit_loss = -bet_amount
        
        # Wallet calculation: Only deduct bet, no return
        # balance - bet_amount
        # Example: balance - 100
        new_balance = balance - bet_amount
    
    # Ensure balance stays within bounds [0, max_balance]
    new_balance = min(max_balance, max(0, new_balance))
    
    return {
        'is_win': is_win,
        'profit_loss': round(profit_loss, 2),  # Positive for win (pred_min × 100), negative for loss (-100)
        'balance_after': round(new_balance, 2)
    }

def simulate_bet_prediction(predicted_value, actual_multiplier, bet_amount, balance):
    """Simulate a bet based on prediction value."""
    # Win if actual >= predicted (or within small tolerance)
    tolerance = 0.1  # 10% tolerance
    is_win = actual_multiplier >= (predicted_value * (1 - tolerance))
    
    if is_win:
        profit_loss = (actual_multiplier - 1) * bet_amount
    else:
        profit_loss = -bet_amount
    
    new_balance = min(max_balance, balance + profit_loss)
    new_balance = max(0, new_balance)
    
    return {
        'is_win': is_win,
        'profit_loss': round(profit_loss, 2),
        'balance_after': round(new_balance, 2)
    }

def simulate_bet(prediction_data, actual_multiplier):
    """Simulate bets based on both minimum range and prediction value.
    IMPORTANT: Only places bets when pred_min > 2.0 (strictly greater than 2.0).
    """
    global current_balance, base_bet_amount, max_balance, betting_history, last_bet_timestamp
    global min_range_bets, prediction_bets, min_range_balance, prediction_balance
    
    if not betting_enabled:
        return None
    
    # Parse prediction range
    pred_range = prediction_data.get('prediction_range', '')
    predicted_value = prediction_data.get('predicted_multiplier', 1.0)
    
    try:
        if ' - ' in pred_range:
            pred_min, pred_max = map(float, pred_range.split(' - '))
        else:
            pred_min = pred_max = float(pred_range.replace('x', '')) if pred_range else predicted_value
    except:
        pred_min = pred_max = predicted_value
    
    # Ensure pred_min is at least 1.0
    pred_min = max(1.0, pred_min)
    predicted_value = max(1.0, predicted_value)
    
    # RULE 1: When pred_min < 2.0: Do NOT bet, do NOT calculate profit/loss, do NOT update wallet
    # RULE 2: When pred_min > 2.0: Bet and calculate profit/loss
    # Must be strictly > 2.0 (not >= 2.0)
    if pred_min <= 2.0:
        print(f"⏭️  RULE 1: Bet BLOCKED - pred_min ({pred_min:.2f}x) ≤ 2.0x. No bet, no profit/loss calculation, no wallet update.")
        return None  # Return None immediately - no bet processing, no profit/loss calculation, no wallet update
    
    # Bet amount (base 100 rupees)
    bet_amount_per_strategy = base_bet_amount  # Always ₹100
    
    # Check: Ensure sufficient balance in main wallet
    if current_balance < bet_amount_per_strategy:
        print(f"⚠️  Bet BLOCKED: Insufficient balance (₹{current_balance:.2f} < ₹{bet_amount_per_strategy:.2f}).")
        return None  # Return None if insufficient balance
    
    # RULE 2: At this point, pred_min > 2.0 and balance is sufficient - proceed with bet
    print(f"✅ RULE 2: Bet PLACED - pred_min ({pred_min:.2f}x) > 2.0x. Proceeding with bet and profit/loss calculation.")
    
    # Use current_balance (main wallet) for the bet calculation
    # This will apply RULE 3 (WIN: Profit = pred_min × 100) or RULE 4 (LOSS: Loss = 100)
    min_range_result = simulate_bet_min_range(pred_min, actual_multiplier, bet_amount_per_strategy, current_balance)
    
    # Update min_range_balance for tracking (but main wallet is current_balance)
    min_range_balance = min_range_result['balance_after']
    
    # Simulate bet based on prediction value
    prediction_result = None
    if prediction_balance >= bet_amount_per_strategy:
        prediction_result = simulate_bet_prediction(predicted_value, actual_multiplier, bet_amount_per_strategy, prediction_balance)
        prediction_balance = prediction_result['balance_after']
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    min_range_bet = None
    prediction_bet = None
    
    # Update wallet balance based on profit/loss from minimum range bets
    # This is the main wallet that tracks P/L from minimum range strategy
    # Note: min_range_result is guaranteed to exist here because we checked pred_min > 2.0 above
    if min_range_result:
        # Store balance before bet for record keeping
        wallet_balance_before = current_balance
        
        # Get profit/loss amount (already calculated correctly in simulate_bet_min_range)
        profit_loss_amount = min_range_result['profit_loss']
        
        # Update wallet balance: Use the balance_after from simulate_bet_min_range
        # This already accounts for: deduct bet, then add return if win
        current_balance = min_range_result['balance_after']
        
        # Log the result with clear rule indication
        if min_range_result['is_win']:
            # RULE 3: WIN → Profit = pred_min × 100
            net_change = current_balance - wallet_balance_before
            print(f"✅ RULE 3 (WIN): pred_min={pred_min:.2f}x, actual={actual_multiplier:.2f}x, Profit=₹{profit_loss_amount:.2f} (pred_min×100), Net Change=₹{net_change:.2f}, Balance: ₹{wallet_balance_before:.2f} → ₹{current_balance:.2f}")
        else:
            # RULE 4: LOSS → Loss = 100
            net_change = current_balance - wallet_balance_before
            print(f"❌ RULE 4 (LOSS): pred_min={pred_min:.2f}x, actual={actual_multiplier:.2f}x, Loss=₹{abs(profit_loss_amount):.2f}, Net Change=₹{net_change:.2f}, Balance: ₹{wallet_balance_before:.2f} → ₹{current_balance:.2f}")
        
        # Record minimum range bet (only if bet was placed, which it was since we passed the pred_min > 2.0 check)
        # Calculate profit based on minimum range value (pred_min), not actual multiplier
        calculated_profit = min_range_result['profit_loss']  # Already calculated from pred_min
        
        min_range_bet = {
            'timestamp': timestamp,
            'bet_amount': round(bet_amount_per_strategy, 2),
            'bet_type': 'min_range',
            'predicted_value': round(pred_min, 2),  # Minimum range value
            'actual_multiplier': round(actual_multiplier, 2),  # Actual outcome (for reference only, NOT used for profit)
            'profit_loss': calculated_profit,  # Profit calculated ONLY from pred_min, NOT from actual_multiplier
            'balance_before': round(wallet_balance_before, 2),  # Wallet balance before bet
            'balance_after': round(current_balance, 2),  # Wallet balance after bet (updated above)
            'is_win': min_range_result['is_win'],
            'confidence': prediction_data.get('confidence', 0),
            'profit_based_on': 'min_range_value'  # Explicitly indicate profit is based on minimum range, not actual
        }
        min_range_bets.append(min_range_bet)
        betting_history.append(min_range_bet)
    
    # Record prediction value bet (optional - separate strategy)
    if prediction_result:
        balance_before_pred = prediction_balance - prediction_result['profit_loss']
        prediction_bet = {
            'timestamp': timestamp,
            'bet_amount': round(bet_amount_per_strategy, 2),
            'bet_type': 'prediction',
            'predicted_value': round(predicted_value, 2),
            'actual_multiplier': round(actual_multiplier, 2),
            'profit_loss': prediction_result['profit_loss'],
            'balance_before': round(balance_before_pred, 2),
            'balance_after': prediction_result['balance_after'],
            'is_win': prediction_result['is_win'],
            'confidence': prediction_data.get('confidence', 0)
        }
        prediction_bets.append(prediction_bet)
        betting_history.append(prediction_bet)
    
    last_bet_timestamp = timestamp
    
    # Return combined result (only if at least one bet was placed)
    # Since we already checked pred_min > 2.0, min_range_bet should exist
    if min_range_bet is None:
        return None  # Safety check - should not happen if logic is correct
    
    return {
        'timestamp': timestamp,
        'min_range_bet': min_range_bet,
        'prediction_bet': prediction_bet,
        'total_profit_loss': (min_range_result['profit_loss'] if min_range_result else 0) + (prediction_result['profit_loss'] if prediction_result else 0)
    }

def check_and_process_bet():
    """Check for new actual multiplier and process bet if prediction exists.
    
    GAMBLING APPLICATION FLOW:
    ===========================
    1. This function runs automatically when new data arrives in CSV
    2. Checks if we have a prediction ready (current_prediction)
    3. Gets the latest actual multiplier from CSV
    4. Calls simulate_bet() which:
       - Checks if pred_min > 2.0 (RULE 1 & 2)
       - If yes, places bet and calculates profit/loss (RULE 3 & 4)
       - Updates wallet balance
    5. Records bet in betting_history for tracking
    
    Example Flow:
    - Prediction: "2.5x - 3.8x" (pred_min = 2.5x)
    - New actual result: 3.2x
    - Since 3.2x >= 2.5x → WIN
    - Profit = 2.5 × 100 = ₹250
    - Wallet: ₹50,000 - ₹100 + ₹250 = ₹50,150
    """
    global current_prediction, all_training_data, last_bet_record_count, processed_multipliers
    
    if current_prediction is None or all_training_data is None or not betting_enabled:
        return
    
    # Get the latest actual multiplier
    df = all_training_data.copy()
    if len(df) == 0:
        return
    
    current_record_count = len(df)
    
    # Process bets for any new records we haven't bet on yet
    # Start from the record after last_bet_record_count
    start_idx = last_bet_record_count if last_bet_record_count > 0 else 0
    
    for idx in range(start_idx, current_record_count):
        actual_multiplier = float(df['multiplier'].iloc[idx])
        actual_timestamp = df['timestamp'].iloc[idx].strftime('%Y-%m-%d %H:%M:%S')
        
        # Create unique key for this multiplier
        multiplier_key = (actual_timestamp, actual_multiplier)
        
        # Skip if already processed
        if multiplier_key in processed_multipliers:
            continue
        
        # Use the prediction that was available at the time (current prediction)
        # In a real scenario, you'd want to use the prediction that existed before this multiplier
        # For now, we use current prediction as approximation
        bet_result = simulate_bet(current_prediction, actual_multiplier)
        if bet_result:
            min_bet = bet_result.get('min_range_bet')
            pred_bet = bet_result.get('prediction_bet')
            
            if min_bet:
                status_min = "⭐ WIN" if min_bet['is_win'] else "LOSS"
                # Clarify that profit is calculated from minimum range value, not actual multiplier
                profit_source = f"Min Range ({min_bet['predicted_value']}x)" if min_bet['is_win'] else "Loss"
                print(f"💰 Min Range Bet: {status_min} | Bet: ₹{min_bet['bet_amount']:.2f} | Predicted Min: {min_bet['predicted_value']}x | Actual: {actual_multiplier}x | P/L: ₹{min_bet['profit_loss']:.2f} (from {profit_source}) | Balance: ₹{min_bet['balance_after']:.2f}")
            
            if pred_bet:
                status_pred = "⭐ WIN" if pred_bet['is_win'] else "LOSS"
                print(f"💰 Prediction Bet: {status_pred} | Bet: ₹{pred_bet['bet_amount']:.2f} | Predicted: {pred_bet['predicted_value']}x | Actual: {actual_multiplier}x | P/L: ₹{pred_bet['profit_loss']:.2f} | Balance: ₹{pred_bet['balance_after']:.2f}")
            
            processed_multipliers.add(multiplier_key)
    
    # Update last bet record count
    last_bet_record_count = current_record_count

@app.route('/')
def index():
    return render_template('predictor.html')

@app.route('/api/load_csv', methods=['GET'])
def api_load_csv():
    """API endpoint to load CSV data."""
    df, file_info = load_latest_csv()
    
    if df is None:
        return jsonify({
            'success': False,
            'error': f'No CSV file found or error loading: {file_info}'
        })
    
    return jsonify({
        'success': True,
        'filename': file_info,
        'total_records': len(df),
        'latest_multiplier': float(df['multiplier'].iloc[-1]),
        'latest_timestamp': df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S'),
        'statistics': {
            'mean': float(df['multiplier'].mean()),
            'std': float(df['multiplier'].std()),
            'min': float(df['multiplier'].min()),
            'max': float(df['multiplier'].max()),
            'median': float(df['multiplier'].median())
        },
        'recent_data': df.tail(20).to_dict('records')
    })

@app.route('/api/train', methods=['POST'])
def api_train():
    """API endpoint to train the model."""
    df, file_info = load_latest_csv()
    
    if df is None:
        return jsonify({
            'success': False,
            'error': f'No CSV file found: {file_info}'
        })
    
    success, message = train_model(df)
    
    return jsonify({
        'success': success,
        'message': message,
        'accuracy': model_accuracy if success else None
    })

@app.route('/api/predict', methods=['GET'])
def api_predict():
    """API endpoint to get prediction."""
    df, file_info = load_latest_csv()
    
    if df is None:
        return jsonify({
            'success': False,
            'error': f'No CSV file found: {file_info}'
        })
    
    prediction, error = predict_next(df)
    
    if error:
        return jsonify({
            'success': False,
            'error': error
        })
    
    return jsonify({
        'success': True,
        'prediction': prediction
    })

@app.route('/api/status', methods=['GET'])
def api_status():
    """API endpoint to get current status (for real-time updates)."""
    df, file_info = load_latest_csv()
    
    # Check and process bet if new data available
    if trained_model is not None and current_prediction is not None:
        check_and_process_bet()
    
    # Calculate total P&L for min range bets
    min_range_total_pl = sum(bet['profit_loss'] for bet in min_range_bets)
    min_range_wins = sum(1 for bet in min_range_bets if bet['is_win'])
    min_range_losses = len(min_range_bets) - min_range_wins
    min_range_win_rate = (min_range_wins / len(min_range_bets) * 100) if min_range_bets else 0
    
    # Calculate total P&L for prediction bets
    prediction_total_pl = sum(bet['profit_loss'] for bet in prediction_bets)
    prediction_wins = sum(1 for bet in prediction_bets if bet['is_win'])
    prediction_losses = len(prediction_bets) - prediction_wins
    prediction_win_rate = (prediction_wins / len(prediction_bets) * 100) if prediction_bets else 0
    
    # Overall totals
    total_profit_loss = min_range_total_pl + prediction_total_pl
    total_bets = len(min_range_bets) + len(prediction_bets)
    
    # Calculate wallet P/L: current_balance - initial_balance (50000)
    # If positive = profit, if negative = loss
    wallet_profit_loss = current_balance - max_balance
    
    status = {
        'model_trained': trained_model is not None,
        'model_accuracy': model_accuracy,
        'current_prediction': current_prediction,
        'prediction_history': prediction_history[-10:],  # Last 10 predictions
        'auto_train_enabled': auto_train_enabled,
        'csv_file': file_info,
        'csv_records': len(df) if df is not None else 0,
        'latest_multiplier': float(df['multiplier'].iloc[-1]) if df is not None and len(df) > 0 else None,
        'betting': {
            'current_balance': round(current_balance, 2),
            'max_balance': max_balance,
            'base_bet_amount': base_bet_amount,
            'wallet_profit_loss': round(wallet_profit_loss, 2),  # P/L based on current vs initial balance
            'total_profit_loss': round(total_profit_loss, 2),
            'total_bets': total_bets,
            'betting_enabled': bool(betting_enabled),  # Convert to bool for JSON
            'min_range': {
                'balance': round(min_range_balance, 2),
                'total_profit_loss': round(min_range_total_pl, 2),
                'total_bets': len(min_range_bets),
                'total_wins': int(min_range_wins),  # Ensure int
                'total_losses': int(min_range_losses),  # Ensure int
                'win_rate': round(min_range_win_rate, 2)
            },
            'prediction': {
                'balance': round(prediction_balance, 2),
                'total_profit_loss': round(prediction_total_pl, 2),
                'total_bets': len(prediction_bets),
                'total_wins': int(prediction_wins),  # Ensure int
                'total_losses': int(prediction_losses),  # Ensure int
                'win_rate': round(prediction_win_rate, 2)
            }
        }
    }
    
    return jsonify(status)

@app.route('/api/betting_history', methods=['GET'])
def api_betting_history():
    """API endpoint to get betting history."""
    # Convert betting history to JSON-serializable format (convert bool to int)
    def make_json_serializable(bet):
        bet_copy = bet.copy()
        bet_copy['is_win'] = int(bet['is_win'])  # Convert bool to int (0 or 1)
        return bet_copy
    
    history_serializable = [make_json_serializable(bet) for bet in betting_history[-100:]]
    min_range_serializable = [make_json_serializable(bet) for bet in min_range_bets[-50:]]
    prediction_serializable = [make_json_serializable(bet) for bet in prediction_bets[-50:]]
    
    return jsonify({
        'success': True,
        'history': history_serializable,
        'min_range_bets': min_range_serializable,
        'prediction_bets': prediction_serializable,
        'summary': {
            'current_balance': round(current_balance, 2),
            'total_profit_loss': round(sum(bet['profit_loss'] for bet in betting_history), 2),
            'total_bets': len(betting_history),
            'min_range': {
                'balance': round(min_range_balance, 2),
                'total_profit_loss': round(sum(bet['profit_loss'] for bet in min_range_bets), 2),
                'total_bets': len(min_range_bets),
                'total_wins': sum(1 for bet in min_range_bets if bet['is_win']),
                'total_losses': sum(1 for bet in min_range_bets if not bet['is_win']),
                'win_rate': round((sum(1 for bet in min_range_bets if bet['is_win']) / len(min_range_bets) * 100) if min_range_bets else 0, 2)
            },
            'prediction': {
                'balance': round(prediction_balance, 2),
                'total_profit_loss': round(sum(bet['profit_loss'] for bet in prediction_bets), 2),
                'total_bets': len(prediction_bets),
                'total_wins': sum(1 for bet in prediction_bets if bet['is_win']),
                'total_losses': sum(1 for bet in prediction_bets if not bet['is_win']),
                'win_rate': round((sum(1 for bet in prediction_bets if bet['is_win']) / len(prediction_bets) * 100) if prediction_bets else 0, 2)
            }
        }
    })

@app.route('/api/reset_betting', methods=['POST'])
def api_reset_betting():
    """API endpoint to reset betting simulation."""
    global current_balance, betting_history, last_bet_timestamp, last_bet_record_count, processed_multipliers
    global min_range_bets, prediction_bets, min_range_balance, prediction_balance
    current_balance = max_balance
    min_range_balance = max_balance
    prediction_balance = max_balance
    betting_history = []
    min_range_bets = []
    prediction_bets = []
    last_bet_timestamp = None
    last_bet_record_count = 0
    processed_multipliers = set()
    return jsonify({
        'success': True,
        'message': 'Betting simulation reset',
        'balance': current_balance
    })

class CSVFileHandler(FileSystemEventHandler):
    """Handler for CSV file changes - FAST response."""
    def on_modified(self, event):
        if event.src_path.endswith('.csv') and 'aviator_payouts_' in event.src_path:
            print(f"📝 CSV file modified: {event.src_path}")
            # Wait minimal time for file to be fully written (reduced from 0.8s to 0.2s for faster response)
            time.sleep(0.2)
            # Trigger immediate processing in a separate thread to avoid blocking
            threading.Thread(target=process_new_data_immediately, daemon=True).start()
    
    def on_created(self, event):
        """Handle new CSV file creation - FAST response."""
        if event.src_path.endswith('.csv') and 'aviator_payouts_' in event.src_path:
            print(f"📝 New CSV file created: {event.src_path}")
            # Wait minimal time for file to be fully written (reduced from 0.8s to 0.2s for faster response)
            time.sleep(0.2)
            threading.Thread(target=process_new_data_immediately, daemon=True).start()

def process_new_data_immediately():
    """Process new data immediately when CSV is updated - Fast predictions, periodic retraining."""
    global auto_train_enabled, last_training_record_count, all_training_data, last_prediction_multiplier, last_retrain_record_count
    
    try:
        updated, df, file_info = check_csv_updated()
        
        if updated and df is not None:
            # Update all_training_data for betting system
            all_training_data = df.copy()
            
            # Calculate new records properly
            if last_training_record_count > 0:
                new_records = len(df) - last_training_record_count
            else:
                new_records = len(df)  # First time loading
            
            # Get the latest multiplier (most recent outcome)
            latest_multiplier = None
            if len(df) > 0:
                latest_multiplier = float(df['multiplier'].iloc[-1])
            
            # Only update prediction if a NEW outcome is finalized (new multiplier recorded)
            # This prevents prediction changes during flight (while CSV is being written)
            should_update_prediction = False
            if latest_multiplier is not None:
                if last_prediction_multiplier is None:
                    # First time - always update
                    should_update_prediction = True
                elif latest_multiplier != last_prediction_multiplier:
                    # New outcome finalized - update prediction
                    should_update_prediction = True
                    print(f"🎯 New outcome finalized: {latest_multiplier}x (previous: {last_prediction_multiplier}x)")
            
            print(f"🔄 New data detected in {file_info} ({len(df)} total records, {new_records} new)")
            
            # Auto-train if enabled - PREDICT ONLY AFTER ACTUAL RESULT IS RECORDED AND ANALYZED
            if auto_train_enabled:
                # Train if model doesn't exist and we have enough data
                if trained_model is None and len(df) >= 30:
                    print("🤖 Training model with all available data...")
                    success, msg = train_model(df, auto=True)
                    if success:
                        # Predict after training (first time)
                        prediction, error = predict_next(df)
                        if not error and latest_multiplier is not None:
                            last_prediction_multiplier = latest_multiplier
                    # Update record count after training
                    last_training_record_count = len(df)
                # If model exists, ONLY predict when new actual result is finalized
                elif trained_model is not None:
                    # STEP 1: Wait for new actual result to be finalized
                    if should_update_prediction and new_records > 0:
                        print(f"🎯 New actual result: {latest_multiplier}x → Predicting IMMEDIATELY...")
                        
                        # STEP 2: Predict IMMEDIATELY with existing model (FAST - uses latest data)
                        # Update all_training_data first so prediction uses new result (FAST - no delay)
                        all_training_data = df.copy()
                        
                        # FAST PREDICTION - immediate response, no waiting
                        prediction, error = predict_next(df)
                        if error:
                            print(f"⚠️ Prediction error: {error}")
                        else:
                            print(f"✅ FAST Prediction: {prediction['predicted_multiplier']}x (confidence: {prediction['confidence']}%)")
                            # Update last prediction multiplier
                            if latest_multiplier is not None:
                                last_prediction_multiplier = latest_multiplier
                            # Check and process bet with new prediction (non-blocking for speed)
                            threading.Thread(target=check_and_process_bet, daemon=True).start()
                        
                        # STEP 3: Retrain model in background (periodically, not every time)
                        # Only retrain if we have enough new records since last retrain
                        records_since_retrain = len(df) - last_retrain_record_count
                        if len(df) >= 30 and records_since_retrain >= retrain_interval:
                            # Retrain in background thread (non-blocking)
                            def retrain_in_background():
                                print(f"🔄 Retraining model in background ({records_since_retrain} new records since last retrain)...")
                                success, msg = train_model(df, auto=True)
                                if success:
                                    global last_retrain_record_count
                                    last_retrain_record_count = len(df)
                                    print(f"✅ Model retrained in background")
                                else:
                                    print(f"⚠️ Background retraining failed: {msg}")
                            
                            threading.Thread(target=retrain_in_background, daemon=True).start()
                        
                        # Update record count after prediction
                        last_training_record_count = len(df)
                    elif new_records > 0:
                        print(f"⏸️  Waiting for new actual result to finalize (current: {latest_multiplier}x, last processed: {last_prediction_multiplier}x)")
                # If model exists but not enough data, still try to predict after new result
                elif trained_model is not None and len(df) >= 15:
                    if should_update_prediction:
                        print(f"🎯 New actual result received: {latest_multiplier}x")
                        print(f"📊 Analyzing data with new result...")
                        print("🔮 Predicting next outcome...")
                        prediction, error = predict_next(df)
                        if not error and latest_multiplier is not None:
                            last_prediction_multiplier = latest_multiplier
                        last_training_record_count = len(df)
    except Exception as e:
        print(f"⚠️ Error processing new data: {e}")
        import traceback
        traceback.print_exc()

def background_worker():
    """Background worker that checks for CSV updates and auto-trains/predicts - FAST POLLING."""
    global auto_train_enabled
    
    print("🔄 Background worker polling every 0.2 seconds for CSV updates (FAST MODE)...")
    
    while True:
        try:
            # Check for updates continuously (this is the primary detection method)
            updated, df, file_info = check_csv_updated()
            
            if updated and df is not None:
                print(f"🔔 Background worker detected update in {file_info}")
                # Process immediately
                process_new_data_immediately()
            else:
                # Even if not detected as updated, check if we should update prediction
                # (fallback for edge cases)
                if trained_model is not None:
                    df, file_info = load_latest_csv()
                    if df is not None and len(df) > last_training_record_count:
                        print(f"🔔 Fallback detection: Found {len(df) - last_training_record_count} new records")
                        process_new_data_immediately()
            
            time.sleep(0.2)  # Check every 0.2 seconds (FAST - for instant predictions)
            
        except Exception as e:
            print(f"⚠️ Background worker error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

def initialize_betting_system():
    """Initialize betting system on app startup - reset all P/L to zero and track from this point forward."""
    global current_balance, betting_history, last_bet_record_count, processed_multipliers, all_training_data
    global min_range_balance, prediction_balance, min_range_bets, prediction_bets
    
    print("💰 Initializing Profit & Loss Tracking System...")
    print(f"   Starting Balance (Each Strategy): ₹{max_balance:,.2f}")
    print(f"   Base Bet Amount: ₹{base_bet_amount:,.2f}")
    print(f"   Strategies: Minimum Range Value & Prediction Value")
    
    # RESET all betting data to zero when app starts
    current_balance = max_balance
    min_range_balance = max_balance
    prediction_balance = max_balance
    betting_history = []
    min_range_bets = []
    prediction_bets = []
    processed_multipliers = set()
    last_bet_timestamp = None
    
    print(f"   🔄 Reset: All P/L, wins, losses set to ZERO")
    print(f"   📊 Initial State: Balance=₹{max_balance:,.2f}, Total P/L=₹0.00, Wins=0, Losses=0")
    
    # Load current data to set starting point
    df, file_info = load_latest_csv()
    if df is not None:
        # Set last_bet_record_count to current record count
        # This ensures we only track bets for NEW records that arrive AFTER app startup
        last_bet_record_count = len(df)
        all_training_data = df.copy()
        print(f"   📊 Found {len(df)} existing records - P/L tracking will start from NEW records only")
        print(f"   ✅ Starting point set: Will track bets from record #{len(df) + 1} onwards")
    else:
        last_bet_record_count = 0
        print(f"   📊 No existing data - Will track all bets from the start")
    
    print("✅ Betting system initialized - P/L tracking starts from ZERO at app startup time!")

def start_background_tasks():
    """Start background tasks for file watching and auto-training."""
    # Initialize betting system first
    initialize_betting_system()
    
    # Start background worker thread
    worker_thread = threading.Thread(target=background_worker, daemon=True)
    worker_thread.start()
    print("✅ Background worker started (auto-sync enabled)")
    
    # Initialize CSV tracking (betting system already loaded data in initialize_betting_system)
    df, file_info = load_latest_csv()
    if df is not None:
        global last_csv_size, last_csv_mtime, last_training_record_count, all_training_data
        last_csv_size = os.path.getsize(file_info)
        last_csv_mtime = os.path.getmtime(file_info)
        last_training_record_count = len(df)  # Initialize record count for model training
        # all_training_data is already set in initialize_betting_system
        if all_training_data is None:
            all_training_data = df.copy()  # Store for betting system if not already set
        print(f"📊 Tracking CSV: {file_info} ({len(df)} records)")
        
        # Auto-train on startup if enough data (use ALL data)
        if len(df) >= 30:
            print(f"🤖 Auto-training model on startup with ALL {len(df)} records...")
            success, msg = train_model(df, auto=True)
            if success:
                prediction, error = predict_next(df)
                if not error and len(df) > 0:
                    # Initialize last prediction multiplier with latest outcome
                    global last_prediction_multiplier, last_retrain_record_count
                    last_prediction_multiplier = float(df['multiplier'].iloc[-1])
                    last_retrain_record_count = len(df)  # Initialize retrain counter
                    print(f"📌 Prediction initialized for next outcome (last finalized: {last_prediction_multiplier}x)")
            else:
                print(f"⚠️ Training failed: {msg}")
            # NOTE: Do NOT call check_and_process_bet() here - we only want to track NEW bets
            # The betting system is already initialized with last_bet_record_count = len(df)
            # So it will only process records that arrive AFTER app startup
            print("💰 Betting system ready - will track P/L for new multipliers only (from app startup)")
        elif len(df) >= 15:
            print(f"⚠️  Only {len(df)} records available. Need at least 30 for optimal training.")
            # Still try to train with available data
            try:
                train_model(df, auto=True)
                if trained_model:
                    predict_next(df)
                    # NOTE: Do NOT call check_and_process_bet() here - only track new bets
                    print("💰 Betting system ready - will track P/L for new multipliers only (from app startup)")
            except:
                pass
        else:
            # Even if not enough for training, set the record count
            last_training_record_count = len(df)
            print(f"⏳ Waiting for more data (need at least 15 records for training)...")
    
    # Start file watcher - watch current directory for CSV changes
    event_handler = CSVFileHandler()
    observer = Observer()
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
    observer.schedule(event_handler, path=current_dir, recursive=False)
    observer.start()
    print(f"👁️  File watcher started (monitoring CSV changes in: {current_dir})")
    print(f"🔍 Watching for files matching: aviator_payouts_*.csv")
    
    return observer

if __name__ == '__main__':
    print("🚀 Starting Aviator Predictor Web App with Auto-Sync...")
    print("=" * 60)
    
    # Start background tasks
    observer = start_background_tasks()
    
    print("📊 Access the interface at: http://localhost:5001")
    print("💡 Using Ensemble Model: Random Forest + Gradient Boosting + Linear Regression")
    print("💡 Features: 25 advanced time-series features with instant data sync")
    print("🔄 Auto-training and prediction updates enabled")
    print("💰 Profit & Loss tracking: ACTIVE (Starting from ₹50,000)")
    print("=" * 60)
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5001, use_reloader=False)
    except KeyboardInterrupt:
        print("\n⚠️  Shutting down...")
        observer.stop()
    observer.join()
