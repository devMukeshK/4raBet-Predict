from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import glob
import threading
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from scipy import stats
from scipy.optimize import minimize
from collections import deque

# Try to import hmmlearn for HMM regime detection
HMM_AVAILABLE = False
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except (ImportError, Exception) as e:
    HMM_AVAILABLE = False
    print(f"⚠️ hmmlearn not available: {str(e)[:100]}")
    print("   Install with: pip install hmmlearn")

# Try to import H2O AutoML
H2O_AVAILABLE = False
try:
    import h2o
    from h2o.automl import H2OAutoML
    H2O_AVAILABLE = True
    print("✅ H2O AutoML available")
except (ImportError, Exception) as e:
    H2O_AVAILABLE = False
    print(f"⚠️ H2O AutoML not available: {str(e)[:100]}")
    print("   Install with: pip install h2o")

app = Flask(__name__)

# Global variables (Scikit-Learn removed - using H2O AutoML only)
last_csv_size = 0
last_csv_mtime = 0
last_training_record_count = 0  # Track how many records were processed last time
last_prediction_multiplier = None  # Track last multiplier used for prediction update (prevents updates during flight)

# ===== H2O AutoML Global Variables =====
h2o_model = None
h2o_accuracy = None
h2o_current_prediction = None
h2o_prediction_history = []
h2o_last_retrain_record_count = 0
h2o_retrain_interval = 10  # Retrain H2O model every 10 new records
h2o_training_history = []
h2o_performance_history = deque(maxlen=100)
h2o_initialized = False
h2o_training_in_progress = False  # Track if training is currently running
h2o_last_training_attempt = None  # Track last training attempt timestamp

# H2O Betting simulation storage
h2o_min_range_bets = []  # H2O bets based on minimum range value
h2o_min_range_balance = 50000.0  # Starting balance for H2O betting
h2o_last_bet_record_count = 0  # Track last processed record for H2O betting
auto_train_enabled = True
training_lock = threading.Lock()
all_training_data = None  # Store all data for instant sync

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

# File to persist every bet decision
bet_log_file = os.path.join(os.path.dirname(__file__), "betting_log.csv")

# ===== ML PROBABILITY SYSTEM GLOBALS =====
# Bayesian smoothing
bayesian_prior_alpha = 2.0  # Prior for beta distribution
bayesian_prior_beta = 2.0
bayesian_history = deque(maxlen=100)  # Recent prediction accuracy

# Probability calibration
calibration_data = deque(maxlen=200)  # (predicted_prob, actual_outcome)
calibration_model = None

# HMM regime detection
hmm_model = None
current_regime = None  # 'bull', 'bear', 'neutral'
regime_history = deque(maxlen=50)

# Risk management
stop_loss_threshold = -0.20  # Stop betting if down 20% from peak
take_profit_threshold = 0.30  # Take profit if up 30% from start
peak_balance = 50000.0  # Track peak balance for stop-loss
starting_balance = 50000.0  # Starting balance for take-profit

# Variance circuit breaker
variance_threshold = 2.5  # Pause if variance exceeds 2.5x normal
recent_variances = deque(maxlen=20)
normal_variance = None

# Confidence decay
prediction_timestamps = {}  # Track when predictions were made
confidence_decay_rate = 0.05  # 5% decay per minute

# Kelly Criterion
kelly_fraction_history = deque(maxlen=50)
max_kelly_fraction = 0.25  # Cap at 25% of bankroll

# Monte-Carlo
monte_carlo_simulations = 1000
monte_carlo_risk_threshold = 0.15  # Max 15% probability of ruin

# EV + Sharpe
ev_history = deque(maxlen=50)
sharpe_history = deque(maxlen=50)

# Model training progress tracking
model_performance_history = deque(maxlen=100)  # Track model performance over time
training_history = []  # Track training events: {timestamp, records_count, mae, r2, win_rate}


def append_bet_to_csv(bet_record: dict):
    """Persist a single bet decision to betting_log.csv."""
    try:
        # Define stable header order
        headers = [
            "timestamp",
            "bet_type",
            "bet_placed",
            "predicted_range",
            "pred_min",
            "pred_max",
            "predicted_value",
            "actual_multiplier",
            "bet_amount",
            "payout",          # total return when win (0 when loss)
            "profit_loss",     # net change to wallet
            "balance_before",
            "balance_after",
            "is_win",
            "confidence",
        ]

        # Write header if file does not exist
        write_header = not os.path.exists(bet_log_file)
        with open(bet_log_file, "a", newline="", encoding="utf-8") as f:
            import csv

            writer = csv.DictWriter(f, fieldnames=headers)
            if write_header:
                writer.writeheader()
            # Filter to known headers; default missing to ''
            row = {h: bet_record.get(h, "") for h in headers}
            writer.writerow(row)
    except Exception as e:
        print(f"⚠️  Failed to append bet to CSV: {e}")

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

# ===== ML PROBABILITY SYSTEM FUNCTIONS =====

def bayesian_smoothing(raw_probability, alpha_prior=2.0, beta_prior=2.0):
    """Apply Bayesian smoothing to probability estimates.
    
    Uses Beta distribution as conjugate prior for binomial outcomes.
    Updates prior based on historical accuracy.
    """
    global bayesian_history, bayesian_prior_alpha, bayesian_prior_beta
    
    # Update priors based on recent history
    if len(bayesian_history) > 10:
        # Calculate success rate from history
        successes = sum(1 for x in bayesian_history if x > 0.5)  # Assuming >0.5 means "win"
        total = len(bayesian_history)
        
        # Update Beta parameters (conjugate prior)
        alpha = alpha_prior + successes
        beta = beta_prior + (total - successes)
    else:
        alpha = alpha_prior
        beta = beta_prior
    
    # Bayesian update: combine prior with current observation
    # Use Beta-Binomial conjugate update
    smoothed_prob = (alpha + raw_probability * 10) / (alpha + beta + 10)
    
    # Ensure probability is in [0, 1]
    smoothed_prob = max(0.0, min(1.0, smoothed_prob))
    
    return smoothed_prob

def calibrate_probability(raw_probability, actual_outcome=None):
    """Calibrate probability using Platt scaling (sigmoid) or isotonic regression.
    
    If actual_outcome is provided, update calibration model.
    """
    global calibration_data, calibration_model
    
    if actual_outcome is not None:
        # Add to calibration data
        calibration_data.append((raw_probability, 1.0 if actual_outcome >= 2.0 else 0.0))
    
    if len(calibration_data) < 20:
        # Not enough data for calibration, return raw
        return raw_probability
    
    # Simple Platt scaling: P_calibrated = 1 / (1 + exp(A * P_raw + B))
    # Fit A and B using logistic regression on calibration data
    try:
        X = np.array([[p] for p, _ in calibration_data])
        y = np.array([a for _, a in calibration_data])
        
        # Simple logistic regression (Platt scaling)
        lr = LogisticRegression()
        lr.fit(X, y)
        
        # Predict calibrated probability
        calibrated = lr.predict_proba([[raw_probability]])[0][1]
        return max(0.0, min(1.0, calibrated))
    except:
        # Fallback: linear interpolation
        sorted_data = sorted(calibration_data)
        if raw_probability <= sorted_data[0][0]:
            return sorted_data[0][1]
        if raw_probability >= sorted_data[-1][0]:
            return sorted_data[-1][1]
        
        # Linear interpolation
        for i in range(len(sorted_data) - 1):
            if sorted_data[i][0] <= raw_probability <= sorted_data[i+1][0]:
                x1, y1 = sorted_data[i]
                x2, y2 = sorted_data[i+1]
                calibrated = y1 + (y2 - y1) * (raw_probability - x1) / (x2 - x1)
                return max(0.0, min(1.0, calibrated))
    
    return raw_probability

def detect_hmm_regime(df, n_states=3):
    """Detect market regime using Hidden Markov Model.
    
    States: 0=Bear (low), 1=Neutral, 2=Bull (high)
    """
    global hmm_model, current_regime, regime_history
    
    if len(df) < 30:
        return 'neutral'
    
    try:
        # Prepare observations (multipliers)
        observations = df['multiplier'].values[-50:].reshape(-1, 1)
        
        if HMM_AVAILABLE:
            # Use hmmlearn
            if hmm_model is None:
                # Initialize HMM
                hmm_model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)
                hmm_model.fit(observations)
            else:
                # Update model with new data
                hmm_model.fit(observations)
            
            # Predict current state
            state = hmm_model.predict(observations[-1:])[0]
            
            # Map state to regime
            regime_map = {0: 'bear', 1: 'neutral', 2: 'bull'}
            if n_states == 3:
                current_regime = regime_map.get(state, 'neutral')
            else:
                # For other n_states, use state index
                if state == 0:
                    current_regime = 'bear'
                elif state == n_states - 1:
                    current_regime = 'bull'
                else:
                    current_regime = 'neutral'
        else:
            # Simple heuristic-based regime detection
            recent_mean = np.mean(observations[-10:])
            recent_std = np.std(observations[-10:])
            global_mean = np.mean(observations)
            
            if recent_mean < global_mean - recent_std:
                current_regime = 'bear'
            elif recent_mean > global_mean + recent_std:
                current_regime = 'bull'
            else:
                current_regime = 'neutral'
        
        regime_history.append(current_regime)
        return current_regime
    except Exception as e:
        print(f"⚠️ HMM regime detection error: {e}")
        return 'neutral'

def calculate_ev_and_sharpe(predicted_prob, predicted_multiplier, bet_amount, recent_returns=None):
    """Calculate Expected Value (EV) and Sharpe ratio.
    
    EV = P(win) * (multiplier - 1) * bet - P(loss) * bet
    Sharpe = (mean_return - risk_free_rate) / std_return
    """
    global ev_history, sharpe_history
    
    # Calculate EV
    win_prob = predicted_prob
    loss_prob = 1 - win_prob
    
    # Expected return per bet
    expected_return = win_prob * (predicted_multiplier - 1) * bet_amount - loss_prob * bet_amount
    ev = expected_return / bet_amount if bet_amount > 0 else 0  # EV as fraction
    
    ev_history.append(ev)
    
    # Calculate Sharpe ratio
    if recent_returns is not None and len(recent_returns) > 10:
        returns = np.array(recent_returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns) if len(returns) > 1 else 0.01
        
        # Risk-free rate = 0 (no risk-free alternative in betting)
        sharpe = mean_return / std_return if std_return > 0 else 0
    else:
        # Estimate Sharpe from EV and variance
        # Assume variance scales with bet size
        estimated_std = bet_amount * 0.5  # Rough estimate
        sharpe = ev * bet_amount / estimated_std if estimated_std > 0 else 0
    
    sharpe_history.append(sharpe)
    
    return {
        'ev': ev,
        'ev_percentage': ev * 100,
        'sharpe': sharpe,
        'expected_return': expected_return
    }

def monte_carlo_risk_filter(predicted_prob, predicted_multiplier, bet_amount, balance, n_simulations=1000):
    """Monte-Carlo simulation to assess risk of ruin.
    
    Returns probability of ruin and recommended action.
    """
    global monte_carlo_simulations, monte_carlo_risk_threshold
    
    n_simulations = monte_carlo_simulations
    
    # Simulate betting outcomes
    ruin_count = 0
    final_balances = []
    
    for _ in range(n_simulations):
        sim_balance = balance
        consecutive_losses = 0
        max_consecutive_losses = 0
        
        # Simulate 100 bets
        for _ in range(100):
            if sim_balance < bet_amount:
                ruin_count += 1
                break
            
            # Simulate bet outcome
            win = np.random.random() < predicted_prob
            if win:
                sim_balance += (predicted_multiplier - 1) * bet_amount
                consecutive_losses = 0
            else:
                sim_balance -= bet_amount
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            
            # Stop if balance drops below 20% of starting
            if sim_balance < balance * 0.2:
                ruin_count += 1
                break
        
        final_balances.append(sim_balance)
    
    prob_ruin = ruin_count / n_simulations
    avg_final_balance = np.mean(final_balances)
    
    # Risk assessment
    risk_level = 'low' if prob_ruin < 0.05 else 'medium' if prob_ruin < 0.15 else 'high'
    
    return {
        'prob_ruin': prob_ruin,
        'risk_level': risk_level,
        'avg_final_balance': avg_final_balance,
        'should_bet': prob_ruin < monte_carlo_risk_threshold
    }

def kelly_criterion(predicted_prob, predicted_multiplier, balance):
    """Calculate optimal bet size using Kelly Criterion.
    
    Kelly % = (P * W - L) / W
    where P = win probability, W = win amount (multiplier - 1), L = loss amount (1)
    """
    global kelly_fraction_history, max_kelly_fraction
    
    if predicted_prob <= 0 or predicted_multiplier <= 1.0:
        return {
            'kelly_fraction': 0.0,
            'kelly_bet': 0.0,
            'full_kelly': 0.0
        }
    
    # Kelly formula
    win_amount = predicted_multiplier - 1.0
    loss_amount = 1.0
    
    kelly_fraction = (predicted_prob * win_amount - (1 - predicted_prob) * loss_amount) / win_amount
    
    # Ensure positive
    kelly_fraction = max(0.0, kelly_fraction)
    
    # Apply fractional Kelly (use 50% of full Kelly for safety)
    fractional_kelly = kelly_fraction * 0.5
    
    # Cap at maximum
    kelly_fraction = min(fractional_kelly, max_kelly_fraction)
    
    # Calculate bet amount
    kelly_bet = balance * kelly_fraction
    
    kelly_fraction_history.append(kelly_fraction)
    
    return {
        'kelly_fraction': kelly_fraction,
        'kelly_bet': kelly_bet,
        'full_kelly': kelly_fraction * 2  # Show what full Kelly would be
    }

def check_stop_loss_take_profit(current_balance):
    """Check if stop-loss or take-profit conditions are met."""
    global peak_balance, starting_balance, stop_loss_threshold, take_profit_threshold
    
    # Update peak balance
    if current_balance > peak_balance:
        peak_balance = current_balance
    
    # Calculate drawdown from peak
    drawdown = (current_balance - peak_balance) / peak_balance if peak_balance > 0 else 0
    
    # Calculate profit from start
    profit = (current_balance - starting_balance) / starting_balance if starting_balance > 0 else 0
    
    # Check stop-loss
    stop_loss_triggered = drawdown <= stop_loss_threshold
    
    # Check take-profit
    take_profit_triggered = profit >= take_profit_threshold
    
    return {
        'stop_loss_triggered': stop_loss_triggered,
        'take_profit_triggered': take_profit_triggered,
        'drawdown': drawdown,
        'profit': profit,
        'peak_balance': peak_balance
    }

def variance_circuit_breaker(df):
    """Pause betting if variance exceeds threshold (circuit breaker)."""
    global recent_variances, normal_variance, variance_threshold
    
    if len(df) < 20:
        return {'should_pause': False, 'variance_ratio': 1.0}
    
    # Calculate recent variance
    recent_window = df['multiplier'].values[-20:]
    current_variance = np.var(recent_window)
    
    recent_variances.append(current_variance)
    
    # Calculate normal variance (from longer history)
    if len(df) >= 100:
        normal_window = df['multiplier'].values[-100:]
        normal_variance = np.var(normal_window)
    elif normal_variance is None:
        normal_variance = current_variance
    
    # Calculate variance ratio
    variance_ratio = current_variance / normal_variance if normal_variance > 0 else 1.0
    
    # Circuit breaker: pause if variance is too high
    should_pause = variance_ratio > variance_threshold
    
    return {
        'should_pause': should_pause,
        'variance_ratio': variance_ratio,
        'current_variance': current_variance,
        'normal_variance': normal_variance
    }

def apply_confidence_decay(prediction_timestamp, initial_confidence):
    """Apply confidence decay over time.
    
    Confidence decreases as prediction ages.
    """
    global confidence_decay_rate
    
    if prediction_timestamp is None:
        return initial_confidence
    
    # Calculate time since prediction (in minutes)
    time_diff = (datetime.now() - prediction_timestamp).total_seconds() / 60.0
    
    # Apply exponential decay
    decay_factor = np.exp(-confidence_decay_rate * time_diff)
    decayed_confidence = initial_confidence * decay_factor
    
    return max(0.0, min(100.0, decayed_confidence))

def ml_probability_decision(prediction_data, df, balance, bet_amount):
    """Comprehensive ML probability-based betting decision.
    
    Integrates all components:
    - Bayesian smoothing
    - Calibration
    - HMM regime detection
    - EV + Sharpe
    - Monte-Carlo risk filter
    - Kelly sizing
    - Stop-loss / Take-profit
    - Variance circuit breaker
    - Confidence decay
    - Final BET/PAUSE decision
    """
    global current_regime, prediction_timestamps
    
    # Extract prediction data
    predicted_multiplier = prediction_data.get('predicted_multiplier', 1.0)
    pred_min = float(prediction_data.get('prediction_range', '1.0 - 1.0').split(' - ')[0])
    initial_confidence = prediction_data.get('confidence', 50.0)
    prediction_time = datetime.strptime(prediction_data.get('predicted_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S')), '%Y-%m-%d %H:%M:%S')
    
    # 1. Calculate raw win probability (probability that actual >= pred_min)
    # Use historical data to estimate
    if len(df) >= 20:
        recent_data = df['multiplier'].values[-50:]
        win_rate = np.sum(recent_data >= pred_min) / len(recent_data)
    else:
        win_rate = 0.5  # Default
    
    # 2. Bayesian smoothing
    raw_prob = win_rate
    bayesian_prob = bayesian_smoothing(raw_prob)
    
    # 3. Calibration
    calibrated_prob = calibrate_probability(bayesian_prob)
    
    # 4. HMM regime detection
    regime = detect_hmm_regime(df)
    
    # 5. Confidence decay
    decayed_confidence = apply_confidence_decay(prediction_time, initial_confidence)
    confidence_factor = decayed_confidence / 100.0
    
    # Adjust probability based on regime and confidence
    if regime == 'bear':
        calibrated_prob *= 0.8  # Reduce probability in bear market
    elif regime == 'bull':
        calibrated_prob *= 1.1  # Increase slightly in bull market
    calibrated_prob = min(1.0, calibrated_prob * confidence_factor)
    
    # 6. EV + Sharpe calculation
    recent_returns = []
    if len(min_range_bets) >= 5:
        recent_returns = [bet['profit_loss'] / bet['bet_amount'] if bet['bet_amount'] > 0 else 0 
                          for bet in min_range_bets[-20:]]
    
    ev_sharpe = calculate_ev_and_sharpe(calibrated_prob, predicted_multiplier, bet_amount, recent_returns)
    
    # 7. Monte-Carlo risk filter
    mc_risk = monte_carlo_risk_filter(calibrated_prob, predicted_multiplier, bet_amount, balance)
    
    # 8. Kelly Criterion
    kelly = kelly_criterion(calibrated_prob, predicted_multiplier, balance)
    
    # 9. Stop-loss / Take-profit
    sl_tp = check_stop_loss_take_profit(balance)
    
    # 10. Variance circuit breaker
    variance_cb = variance_circuit_breaker(df)
    
    # 11. Final BET/PAUSE decision
    should_bet = True
    
    # Decision factors
    decision_factors = {
        'calibrated_prob_high': calibrated_prob >= 0.6,
        'ev_positive': ev_sharpe['ev'] > 0,
        'sharpe_positive': ev_sharpe['sharpe'] > 0,
        'mc_safe': mc_risk['should_bet'],
        'regime_ok': regime != 'bear' or calibrated_prob > 0.7,
        'confidence_ok': decayed_confidence >= 50.0,
        'no_stop_loss': not sl_tp['stop_loss_triggered'],
        'no_take_profit': not sl_tp['take_profit_triggered'],
        'variance_ok': not variance_cb['should_pause'],
        'pred_min_ok': pred_min > 2.0
    }
    
    # Count positive factors
    positive_factors = sum(1 for v in decision_factors.values() if v)
    total_factors = len(decision_factors)
    
    # Require at least 70% of factors to be positive
    should_bet = (positive_factors / total_factors) >= 0.7
    
    # Override: always pause if stop-loss or take-profit triggered
    if sl_tp['stop_loss_triggered']:
        should_bet = False
        decision_factors['stop_loss_override'] = True
    
    if sl_tp['take_profit_triggered']:
        should_bet = False
        decision_factors['take_profit_override'] = True
    
    # Override: always pause if variance circuit breaker
    if variance_cb['should_pause']:
        should_bet = False
        decision_factors['variance_override'] = True
    
    # Calculate optimal bet size (use Kelly if should_bet, else 0)
    if should_bet:
        optimal_bet = min(kelly['kelly_bet'], bet_amount)  # Use Kelly or base bet, whichever is smaller
    else:
        optimal_bet = 0.0
    
    # Compile result
    result = {
        'should_bet': should_bet,
        'optimal_bet': optimal_bet,
        'raw_probability': raw_prob,
        'bayesian_probability': bayesian_prob,
        'calibrated_probability': calibrated_prob,
        'regime': regime,
        'confidence': decayed_confidence,
        'ev': ev_sharpe['ev'],
        'ev_percentage': ev_sharpe['ev_percentage'],
        'sharpe': ev_sharpe['sharpe'],
        'kelly_fraction': kelly['kelly_fraction'],
        'kelly_bet': kelly['kelly_bet'],
        'monte_carlo_risk': mc_risk['prob_ruin'],
        'monte_carlo_risk_level': mc_risk['risk_level'],
        'stop_loss_triggered': sl_tp['stop_loss_triggered'],
        'take_profit_triggered': sl_tp['take_profit_triggered'],
        'variance_circuit_breaker': variance_cb['should_pause'],
        'variance_ratio': variance_cb['variance_ratio'],
        'decision_factors': decision_factors,
        'positive_factors': positive_factors,
        'total_factors': total_factors,
        'decision_score': positive_factors / total_factors
    }
    
    return result

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
        
        # CRITICAL: Low-value pattern detection (< 2.0) - most outcomes are below 2.0
        below_2_ratio = len([x for x in window if x < 2.0]) / len(window)
        below_2_count = len([x for x in window if x < 2.0])
        below_2_streak = 0  # Count consecutive values < 2.0 from end
        for val in reversed(window):
            if val < 2.0:
                below_2_streak += 1
            else:
                break
        
        # Recent low-value frequency (last 3, 5, 10 values)
        recent_3_low = len([x for x in window[-3:] if x < 2.0]) / min(3, len(window))
        recent_5_low = len([x for x in window[-5:] if x < 2.0]) / min(5, len(window))
        recent_10_low = len([x for x in window[-10:] if x < 2.0]) / min(10, len(window))
        
        # Probability of next being < 2.0 based on recent history
        prob_low_next = recent_5_low  # Use recent 5 as proxy
        
        # Average of values < 2.0 in window
        low_values = [x for x in window if x < 2.0]
        avg_low_value = np.mean(low_values) if low_values else 1.5
        
        # Time since last value >= 2.0
        steps_since_above_2 = 0
        for j in range(i-1, max(0, i-20), -1):
            if j >= 0 and df['multiplier'].iloc[j] >= 2.0:
                steps_since_above_2 = i - j
                break
        if steps_since_above_2 == 0:
            steps_since_above_2 = 20  # Default if all recent values < 2.0
        
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
        
        # Feature vector (ANALYZES ENTIRE DATASET + LOCAL PATTERNS - 80+ features including low-value pattern detection)
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
            # Low-value pattern features (9) - CRITICAL for reducing losses
            below_2_ratio, below_2_count, below_2_streak,
            recent_3_low, recent_5_low, recent_10_low,
            prob_low_next, avg_low_value, steps_since_above_2,
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
    """Placeholder - Scikit-Learn model removed, using H2O AutoML only."""
    return False, "Scikit-Learn model removed. Please use H2O AutoML instead."

def predict_next(df):
    """Placeholder - Scikit-Learn prediction removed, using H2O AutoML only."""
    return None, "Scikit-Learn prediction removed. Please use H2O AutoML instead."

# ===== H2O AutoML Functions =====

def init_h2o():
    """Initialize H2O cluster if available with increased memory - thread-safe."""
    global h2o_initialized
    if not H2O_AVAILABLE:
        return False, "H2O AutoML not available. Install with: pip install h2o"
    
    # Use lock to prevent multiple initializations
    with training_lock:
        try:
            # Check if H2O is already connected
            try:
                if h2o.cluster() is not None:
                    h2o_initialized = True
                    return True, "H2O cluster already connected"
            except:
                pass  # Cluster not connected, need to initialize
            
            if not h2o_initialized:
                # Increase memory allocation to prevent OutOfMemoryError
                # Use 4G or more if available, with more threads for better performance
                try:
                    import psutil
                    available_memory_gb = psutil.virtual_memory().available / (1024**3)
                    if available_memory_gb >= 8:
                        mem_size = "6G"
                        nthreads = 4
                    elif available_memory_gb >= 4:
                        mem_size = "4G"
                        nthreads = 2
                    else:
                        mem_size = "2G"
                        nthreads = 2
                except ImportError:
                    # Fallback if psutil not available
                    mem_size = "4G"
                    nthreads = 2
                
                print(f"🔧 Initializing H2O with {mem_size} memory and {nthreads} threads...")
                h2o.init(nthreads=nthreads, max_mem_size=mem_size, ignore_config=True, strict_version_check=False)
                h2o.no_progress()
                h2o_initialized = True
                print(f"✅ H2O cluster initialized with {mem_size} memory")
            return True, "H2O cluster ready"
        except Exception as e:
            error_msg = str(e)
            # Check if cluster is already running
            if "already running" in error_msg.lower() or "already exists" in error_msg.lower():
                try:
                    h2o.cluster()  # Try to connect to existing cluster
                    h2o_initialized = True
                    return True, "H2O cluster already running, connected successfully"
                except:
                    pass
            return False, f"H2O initialization failed: {error_msg[:200]}"

def train_h2o_model(df, auto=False):
    """Train H2O AutoML model on ALL available data."""
    global h2o_model, h2o_accuracy, h2o_last_retrain_record_count, h2o_training_history, h2o_performance_history
    global h2o_training_in_progress, h2o_last_training_attempt
    
    if not H2O_AVAILABLE:
        return False, "H2O AutoML not available"
    
    # Prevent multiple simultaneous training attempts
    with training_lock:
        if h2o_training_in_progress:
            return False, "H2O training already in progress, please wait..."
        
        h2o_training_in_progress = True
        h2o_last_training_attempt = datetime.now()
    
    try:
        # Initialize H2O if needed
        init_success, init_msg = init_h2o()
        if not init_success:
            h2o_training_in_progress = False
            return False, init_msg
        
        if len(df) < 30:
            h2o_training_in_progress = False
            return False, f"Need at least 30 records, got {len(df)}"
        
        print(f"📊 Starting H2O training with {len(df)} records...")
        # Limit dataset size to prevent memory issues (use most recent data)
        max_rows = 10000  # Limit to 10k rows to prevent OOM
        if len(df) > max_rows:
            print(f"⚠️ Dataset has {len(df)} rows, limiting to most recent {max_rows} rows for H2O training")
            df = df.tail(max_rows).reset_index(drop=True)
        
        # Prepare features using the same function as sklearn model
        X, y = prepare_features(df, lookback=20, use_all_data=True)
        if X is None or y is None:
            return False, "Failed to prepare features"
        
        # Limit number of features if too many (to reduce memory usage)
        max_features = 100
        if X.shape[1] > max_features:
            print(f"⚠️ Dataset has {X.shape[1]} features, limiting to {max_features} most important features")
            # Use simple variance-based feature selection
            feature_variances = np.var(X, axis=0)
            top_feature_indices = np.argsort(feature_variances)[-max_features:]
            X = X[:, top_feature_indices]
        
        # Convert to H2O Frame
        feature_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        feature_df['target'] = y

        # Drop constant columns (zero variance) before sending to H2O
        variances = feature_df.var(numeric_only=True)
        constant_cols = variances[variances == 0].index.tolist()
        if constant_cols:
            print(f"⚠️ Dropping constant columns before H2O training: {len(constant_cols)} columns")
            feature_df = feature_df.drop(columns=constant_cols)

        # Ensure we still have at least one feature column besides target
        feature_cols = [c for c in feature_df.columns if c != 'target']
        if not feature_cols:
            return False, "No valid feature columns remaining after dropping constant features"

        print(f"📊 Preparing H2O frame: {len(feature_df)} rows, {len(feature_cols)} features")
        
        # Create H2O frame with error handling for memory issues
        try:
            h2o_frame = h2o.H2OFrame(feature_df)
        except Exception as e:
            error_msg = str(e)
            if "OutOfMemoryError" in error_msg or "heap space" in error_msg:
                return False, f"H2O memory error: Dataset too large. Try reducing data size or increasing H2O memory. Error: {error_msg[:200]}"
            else:
                return False, f"H2O frame creation failed: {error_msg[:200]}"

        # Split into train/test (80/20)
        train_frame, test_frame = h2o_frame.split_frame(ratios=[0.8], seed=42)

        # Guard: if split produced empty frames (can happen with tiny datasets)
        if train_frame.nrows == 0 or test_frame.nrows == 0:
            return False, f"Train/test split failed: train={train_frame.nrows}, test={test_frame.nrows}"

        # Define features and target
        target_col = 'target'
        
        # Train H2O AutoML (limit to 60 seconds for faster training)
        max_runtime_secs = 60 if auto else 120
        print(f"🤖 Training H2O AutoML model (max {max_runtime_secs}s)...")
        
        aml = H2OAutoML(
            max_runtime_secs=max_runtime_secs,
            max_models=10,
            seed=42,
            stopping_metric="MAE",
            stopping_tolerance=0.001,
            stopping_rounds=3,
            sort_metric="MAE"
        )
        
        try:
            print(f"🚀 Starting H2O AutoML training (this may take up to {max_runtime_secs} seconds)...")
            aml.train(
                x=feature_cols,
                y=target_col,
                training_frame=train_frame,
                validation_frame=test_frame
            )
            print("✅ H2O AutoML training completed")
        except Exception as e:
            error_msg = str(e)
            h2o_training_in_progress = False
            if "OutOfMemoryError" in error_msg or "heap space" in error_msg:
                return False, f"H2O training error: Out of memory. Try reducing dataset size or increasing H2O memory allocation. Error: {error_msg[:300]}"
            else:
                return False, f"H2O training error: {error_msg[:300]}"
        
        # Get best model
        best_model = aml.leader
        
        # Evaluate on test set
        test_predictions = best_model.predict(test_frame)
        test_actual = test_frame[target_col]
        
        # Convert to numpy for metrics
        pred_array = test_predictions.as_data_frame().values.flatten()
        actual_array = test_actual.as_data_frame().values.flatten()
        
        # Calculate metrics (using numpy since sklearn removed)
        mae = np.mean(np.abs(actual_array - pred_array))
        rmse = np.sqrt(np.mean((actual_array - pred_array) ** 2))
        ss_res = np.sum((actual_array - pred_array) ** 2)
        ss_tot = np.sum((actual_array - np.mean(actual_array)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Store model and metrics
        h2o_model = best_model
        h2o_accuracy = {
            'mae': round(mae, 3),
            'rmse': round(rmse, 3),
            'r2_score': round(r2, 3),
            'model_id': best_model.model_id,
            'samples_trained': train_frame.nrows,
            'samples_tested': test_frame.nrows,
            'total_records': len(df),
            'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Track performance history
        h2o_performance_history.append({
            'mae': mae,
            'r2': r2,
            'records': len(df),
            'timestamp': datetime.now()
        })
        
        # Calculate maturity
        records_count = len(df)
        if records_count < 100:
            maturity_level = "Early Stage"
            maturity_pct = min(100, (records_count / 100) * 100)
        elif records_count < 500:
            maturity_level = "Learning Phase"
            maturity_pct = min(100, 50 + ((records_count - 100) / 400) * 50)
        elif records_count < 1000:
            maturity_level = "Mature"
            maturity_pct = min(100, 75 + ((records_count - 500) / 500) * 25)
        else:
            maturity_level = "Fully Trained"
            maturity_pct = 100
        
        h2o_accuracy['maturity_level'] = maturity_level
        h2o_accuracy['maturity_pct'] = round(maturity_pct, 1)
        
        # Track training history
        h2o_training_history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'records_count': len(df),
            'mae': round(mae, 3),
            'r2_score': round(r2, 3),
            'model_id': best_model.model_id
        })
        
        if len(h2o_training_history) > 50:
            h2o_training_history = h2o_training_history[-50:]
        
        h2o_last_retrain_record_count = len(df)
        
        if auto:
            print(f"✅ [H2O AUTO-TRAIN] Model trained! MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}, Records: {len(df)}")
        else:
            print(f"✅ H2O AutoML model trained! MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
        
        h2o_training_in_progress = False
        return True, f"H2O AutoML trained successfully! MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}"
        
    except Exception as e:
        h2o_training_in_progress = False
        error_msg = f"H2O training error: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return False, error_msg

def predict_h2o_next(df):
    """Predict the next multiplier value using trained H2O AutoML model."""
    global h2o_model, h2o_current_prediction, h2o_prediction_history
    
    if h2o_model is None:
        return None, "H2O model not trained yet. Please train the model first."
    
    if not H2O_AVAILABLE:
        return None, "H2O AutoML not available"
    
    try:
        # Prepare features using the same function
        X, y = prepare_features(df, lookback=20, use_all_data=True)
        if X is None or y is None:
            return None, "Failed to prepare features"
        
        # Get the last row (most recent features)
        last_features = X[-1:].reshape(1, -1)
        
        # Convert to H2O Frame
        feature_df = pd.DataFrame(last_features, columns=[f'feature_{i}' for i in range(last_features.shape[1])])
        h2o_frame = h2o.H2OFrame(feature_df)
        
        # Predict
        prediction_frame = h2o_model.predict(h2o_frame)
        prediction = float(prediction_frame.as_data_frame().iloc[0, 0])
        
        # Ensure prediction is >= 1.0
        prediction = max(1.0, prediction)
        
        # Get last actual value for comparison
        last_actual = float(df['multiplier'].iloc[-1])
        
        # Calculate confidence based on model accuracy
        confidence = 60.0  # Base confidence
        if h2o_accuracy:
            r2 = h2o_accuracy.get('r2_score', 0)
            if r2 > 0.5:
                confidence = 85.0
            elif r2 > 0.3:
                confidence = 75.0
            elif r2 > 0.1:
                confidence = 65.0
            else:
                confidence = 55.0
        
        # Determine trend
        if prediction > last_actual * 1.1:
            trend = "📈 Increasing"
        elif prediction < last_actual * 0.9:
            trend = "📉 Decreasing"
        else:
            trend = "➡️ Stable"
        
        # Calculate prediction range based on model accuracy
        if h2o_accuracy:
            mae = h2o_accuracy.get('mae', 0.5)
            r2 = h2o_accuracy.get('r2_score', 0)
            
            if r2 > 0.3:
                error_margin = mae * 0.4
            elif r2 > 0.2:
                error_margin = mae * 0.5
            else:
                error_margin = mae * 0.6
            
            max_range = prediction * 0.25
            error_margin = min(error_margin, max_range / 2)
            
            pred_min = max(1.0, prediction - error_margin)
            pred_max = prediction + error_margin
            pred_range = f"{pred_min:.2f} - {pred_max:.2f}"
        else:
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
            'data_points_used': len(df)
        }
        
        h2o_current_prediction = result
        
        # Add to prediction history (keep all, no limit)
        if not h2o_prediction_history or h2o_prediction_history[-1].get('predicted_multiplier') != result.get('predicted_multiplier'):
            h2o_prediction_history.append(result.copy())
        
        return result, None
        
    except Exception as e:
        error_msg = f"H2O prediction error: {str(e)}"
        print(f"❌ {error_msg}")
        return None, error_msg

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
        # Rule 3: WIN → Profit (display payout) = pred_min × 100
        payout = pred_min * bet_amount
        # Net change to wallet = payout - bet_amount
        profit_loss = payout - bet_amount
        new_balance = balance + profit_loss
    else:
        # Rule 4: LOSS → Loss = 100
        payout = 0
        profit_loss = -bet_amount
        new_balance = balance + profit_loss
    
    # Ensure balance doesn't go below 0 (but allow it to go above max_balance for profits)
    new_balance = max(0, new_balance)
    
    return {
        'is_win': is_win,
        'bet_placed': True,
        'payout': round(payout, 2),           # Total return when win (0 when loss)
        'profit_loss': round(profit_loss, 2), # Net change to wallet
        'balance_after': round(new_balance, 2)
    }

def simulate_bet_prediction(predicted_value, actual_multiplier, bet_amount, balance):
    """Simulate a bet based on prediction value."""
    # Win if actual >= predicted (or within small tolerance)
    tolerance = 0.1  # 10% tolerance
    is_win = actual_multiplier >= (predicted_value * (1 - tolerance))
    
    if is_win:
        payout = actual_multiplier * bet_amount
        profit_loss = payout - bet_amount  # net change
    else:
        payout = 0
        profit_loss = -bet_amount
    
    # Calculate new balance (allow it to go above max_balance for profits, but not below 0)
    new_balance = balance + profit_loss
    new_balance = max(0, new_balance)  # Only cap at 0, allow unlimited growth
    
    return {
        'is_win': is_win,
        'bet_placed': True,
        'payout': round(payout, 2),
        'profit_loss': round(profit_loss, 2),
        'balance_after': round(new_balance, 2)
    }

def simulate_bet(prediction_data, actual_multiplier):
    """Simulate bets based on both minimum range and prediction value.
    IMPORTANT: Only places bets when pred_min > 2.0 (strictly greater than 2.0).
    Always saves a record to CSV, even when bet is NOT placed.
    """
    global current_balance, base_bet_amount, max_balance, betting_history, last_bet_timestamp
    global min_range_bets, prediction_bets, min_range_balance, prediction_balance, peak_balance
    
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
    pred_max = max(pred_min, pred_max)  # Ensure pred_max >= pred_min
    predicted_value = max(1.0, predicted_value)
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    confidence = prediction_data.get('confidence', 0)
    
    # Store balance before any bet processing
    wallet_balance_before = current_balance
    prediction_balance_before = prediction_balance
    
    # RULE 1: When pred_min <= 2.0: Do NOT bet, do NOT update profit/loss, just record the decision
    if pred_min <= 2.0:
        print(f"⏭️  RULE 1: Bet BLOCKED - pred_min ({pred_min:.2f}x) ≤ 2.0x. NO BET, NO PROFIT/LOSS UPDATE. Recording decision only.")
        
        # Save record for min_range strategy (bet NOT placed, NO profit/loss update)
        min_range_bet = {
            'timestamp': timestamp,
            'bet_type': 'min_range',
            'bet_placed': False,  # FLAG: Bet was NOT placed
            'predicted_range': pred_range,
            'pred_min': round(pred_min, 2),
            'pred_max': round(pred_max, 2),
            'predicted_value': round(predicted_value, 2),
            'actual_multiplier': round(actual_multiplier, 2),
            'bet_amount': 0.0,  # No bet placed
            'payout': 0.0,
            'profit_loss': 0.0,  # No profit/loss since no bet (balance unchanged)
            'balance_before': round(wallet_balance_before, 2),
            'balance_after': round(wallet_balance_before, 2),  # Balance unchanged - NO UPDATE
            'is_win': False,  # N/A since no bet
            'confidence': confidence
        }
        min_range_bets.append(min_range_bet)
        betting_history.append(min_range_bet)
        append_bet_to_csv(min_range_bet)
        
        # Save record for prediction strategy (bet NOT placed, NO profit/loss update)
        prediction_bet = {
            'timestamp': timestamp,
            'bet_type': 'prediction',
            'bet_placed': False,  # FLAG: Bet was NOT placed
            'predicted_range': pred_range,
            'pred_min': round(pred_min, 2),
            'pred_max': round(pred_max, 2),
            'predicted_value': round(predicted_value, 2),
            'actual_multiplier': round(actual_multiplier, 2),
            'bet_amount': 0.0,  # No bet placed
            'payout': 0.0,
            'profit_loss': 0.0,  # No profit/loss since no bet (balance unchanged)
            'balance_before': round(prediction_balance_before, 2),
            'balance_after': round(prediction_balance_before, 2),  # Balance unchanged - NO UPDATE
            'is_win': False,  # N/A since no bet
            'confidence': confidence
        }
        prediction_bets.append(prediction_bet)
        betting_history.append(prediction_bet)
        append_bet_to_csv(prediction_bet)
        
        last_bet_timestamp = timestamp
        return {
            'timestamp': timestamp,
            'min_range_bet': min_range_bet,
            'prediction_bet': prediction_bet,
            'total_profit_loss': 0.0
        }
    
    # Bet amount (base 100 rupees)
    bet_amount_per_strategy = base_bet_amount  # Always ₹100
    
    # Check: Ensure sufficient balance in main wallet
    if current_balance < bet_amount_per_strategy:
        print(f"⚠️  Bet BLOCKED: Insufficient balance (₹{current_balance:.2f} < ₹{bet_amount_per_strategy:.2f}). Recording decision.")
        
        # Save record for min_range strategy (bet NOT placed due to insufficient balance)
        min_range_bet = {
            'timestamp': timestamp,
            'bet_type': 'min_range',
            'bet_placed': False,  # FLAG: Bet was NOT placed (insufficient balance)
            'predicted_range': pred_range,
            'pred_min': round(pred_min, 2),
            'pred_max': round(pred_max, 2),
            'predicted_value': round(predicted_value, 2),
            'actual_multiplier': round(actual_multiplier, 2),
            'bet_amount': 0.0,
            'payout': 0.0,
            'profit_loss': 0.0,
            'balance_before': round(wallet_balance_before, 2),
            'balance_after': round(wallet_balance_before, 2),
            'is_win': False,
            'confidence': confidence
        }
        min_range_bets.append(min_range_bet)
        betting_history.append(min_range_bet)
        append_bet_to_csv(min_range_bet)
        
        last_bet_timestamp = timestamp
        return {
            'timestamp': timestamp,
            'min_range_bet': min_range_bet,
            'prediction_bet': None,
            'total_profit_loss': 0.0
        }
    
    # ===== ML PROBABILITY DECISION SYSTEM =====
    # Get ML probability-based decision
    global all_training_data
    df_for_ml = all_training_data if all_training_data is not None else load_latest_csv()[0]
    
    if df_for_ml is not None and len(df_for_ml) >= 20:
        ml_decision = ml_probability_decision(prediction_data, df_for_ml, current_balance, bet_amount_per_strategy)
        
        # Log ML decision
        print(f"🤖 ML Probability Decision:")
        print(f"   Calibrated Prob: {ml_decision['calibrated_probability']:.2%}")
        print(f"   EV: {ml_decision['ev_percentage']:.2f}%")
        print(f"   Sharpe: {ml_decision['sharpe']:.2f}")
        print(f"   Regime: {ml_decision['regime']}")
        print(f"   Kelly Bet: ₹{ml_decision['kelly_bet']:.2f}")
        print(f"   MC Risk: {ml_decision['monte_carlo_risk']:.2%} ({ml_decision['monte_carlo_risk_level']})")
        print(f"   Decision Score: {ml_decision['decision_score']:.2%} ({ml_decision['positive_factors']}/{ml_decision['total_factors']} factors)")
        print(f"   Should Bet: {'✅ YES' if ml_decision['should_bet'] else '❌ NO'}")
        
        # 🔧 DESIGN FLAW FIX: Override ML decision if pred_min > 2.0 (strict rule: bet when pred_min > 2.0)
        # ML can only reduce bet size, not block the bet
        if pred_min > 2.0:
            # Override: If pred_min > 2.0, we MUST bet (ML cannot block)
            if not ml_decision['should_bet']:
                print(f"⚠️  ML SYSTEM suggested NO bet, but pred_min ({pred_min:.2f}x) > 2.0x → OVERRIDING ML, bet will be placed")
                ml_decision['should_bet'] = True  # Force bet when pred_min > 2.0
        
        # If ML still says no bet (shouldn't happen if pred_min > 2.0, but handle edge case)
        if not ml_decision['should_bet']:
            print(f"⏸️  ML SYSTEM: Bet PAUSED - Decision score too low or risk factors detected")
            
            # Save record with bet_placed=False due to ML system
            min_range_bet = {
                'timestamp': timestamp,
                'bet_type': 'min_range',
                'bet_placed': False,  # ML system blocked bet
                'predicted_range': pred_range,
                'pred_min': round(pred_min, 2),
                'pred_max': round(pred_max, 2),
                'predicted_value': round(predicted_value, 2),
                'actual_multiplier': round(actual_multiplier, 2),
                'bet_amount': 0.0,
                'payout': 0.0,
                'profit_loss': 0.0,
                'balance_before': round(wallet_balance_before, 2),
                'balance_after': round(wallet_balance_before, 2),
                'is_win': False,
                'confidence': confidence,
                'ml_decision': ml_decision  # Store ML decision data
            }
            min_range_bets.append(min_range_bet)
            betting_history.append(min_range_bet)
            append_bet_to_csv(min_range_bet)
            
            last_bet_timestamp = timestamp
            return {
                'timestamp': timestamp,
                'min_range_bet': min_range_bet,
                'prediction_bet': None,
                'total_profit_loss': 0.0,
                'ml_decision': ml_decision
            }
        
        # Use Kelly-optimized bet size if ML says bet
        optimal_bet = ml_decision['optimal_bet']
        if optimal_bet > 0 and optimal_bet < bet_amount_per_strategy:
            bet_amount_per_strategy = max(optimal_bet, base_bet_amount * 0.5)  # At least 50% of base
            print(f"💰 Using Kelly-optimized bet size: ₹{bet_amount_per_strategy:.2f}")
    else:
        ml_decision = None
        print(f"⚠️  ML system unavailable (insufficient data), using default logic")
    
    # RULE 2: At this point, pred_min > 2.0, balance is sufficient - proceed with bet
    # 🔧 DESIGN FLAW FIX: ML cannot block when pred_min > 2.0 (already handled above)
    print(f"✅ RULE 2: Bet PLACED - pred_min ({pred_min:.2f}x) > 2.0x. Proceeding with bet and profit/loss calculation.")
    
    # CRITICAL: Only bet when pred_min > 2.0 (strictly greater than 2.0)
    # This ensures we only place bets and update profit/loss when pred_min > 2.0
    assert pred_min > 2.0, f"ERROR: Bet should only be placed when pred_min > 2.0, but got {pred_min:.2f}"
    
    # Use current_balance (main wallet) for the bet calculation
    # This will apply RULE 3 (WIN: Profit = pred_min × bet_amount) or RULE 4 (LOSS: Loss = bet_amount)
    min_range_result = simulate_bet_min_range(pred_min, actual_multiplier, bet_amount_per_strategy, current_balance)
    
    # CRITICAL: ALWAYS update balance and profit/loss when bet is placed (pred_min > 2.0)
    # This ensures profit/loss is ALWAYS updated after win or loss
    if min_range_result:
        # 🔧 BUG 3 FIX: Single source of truth for balance
        # current_balance IS the wallet balance (min_range_balance is just a reference)
        old_balance = current_balance
        current_balance = min_range_result['balance_after']
        min_range_balance = current_balance  # Keep in sync (min_range_balance = current_balance always)
        
        # Verify balance was updated
        if abs(current_balance - old_balance) < 0.01 and min_range_result['profit_loss'] != 0:
            print(f"⚠️  WARNING: Balance not updated! Old: ₹{old_balance:.2f}, New: ₹{current_balance:.2f}, P/L: ₹{min_range_result['profit_loss']:.2f}")
            # Force update balance
            current_balance = old_balance + min_range_result['profit_loss']
            min_range_balance = current_balance
            print(f"✅ FORCED Balance update: ₹{old_balance:.2f} → ₹{current_balance:.2f}")
        
        # Update peak balance for stop-loss tracking
        if current_balance > peak_balance:
            peak_balance = current_balance
    else:
        print(f"⚠️  ERROR: min_range_result is None! Bet should have been placed but result is missing.")
        # Don't proceed if result is None
        return {
            'timestamp': timestamp,
            'min_range_bet': None,
            'prediction_bet': None,
            'total_profit_loss': 0.0
        }
    
    # Simulate bet based on prediction value (separate strategy with separate balance)
    prediction_result = None
    if prediction_balance >= bet_amount_per_strategy:
        prediction_result = simulate_bet_prediction(predicted_value, actual_multiplier, bet_amount_per_strategy, prediction_balance)
        prediction_balance = prediction_result['balance_after']
    
    min_range_bet = None
    prediction_bet = None
    
    # Record minimum range bet (bet WAS placed - pred_min > 2.0)
    # CRITICAL: This block ALWAYS executes when pred_min > 2.0 and bet is placed
    if min_range_result:
        # Get profit/loss amount (net change) and payout
        profit_loss_amount = min_range_result['profit_loss']
        payout_amount = min_range_result.get('payout', 0)
        
        # Verify profit/loss calculation is correct
        # Expected: profit_loss = payout - bet_amount (for win) or -bet_amount (for loss)
        expected_profit_loss = (payout_amount - bet_amount_per_strategy) if min_range_result['is_win'] else -bet_amount_per_strategy
        if abs(profit_loss_amount - expected_profit_loss) > 0.01:
            print(f"⚠️  WARNING: Profit/Loss mismatch! Expected: ₹{expected_profit_loss:.2f}, Got: ₹{profit_loss_amount:.2f}")
            # Correct the profit_loss
            profit_loss_amount = expected_profit_loss
            print(f"✅ CORRECTED Profit/Loss to: ₹{profit_loss_amount:.2f}")
        
        # Verify balance calculation is correct - CRITICAL for ensuring profit/loss updates
        expected_balance = wallet_balance_before + profit_loss_amount
        if abs(current_balance - expected_balance) > 0.01:
            print(f"⚠️  WARNING: Balance mismatch! Expected: ₹{expected_balance:.2f}, Got: ₹{current_balance:.2f}")
            # Force correct balance
            current_balance = expected_balance
            min_range_balance = current_balance
            print(f"✅ CORRECTED Balance to: ₹{current_balance:.2f}")
        
        # Log the result with clear rule indication
        net_change = current_balance - wallet_balance_before
        if min_range_result['is_win']:
            # RULE 3: WIN → Profit = pred_min × bet_amount
            print(f"✅ RULE 3 (WIN): pred_min={pred_min:.2f}x, actual={actual_multiplier:.2f}x, Bet=₹{bet_amount_per_strategy:.2f}, Payout=₹{payout_amount:.2f}, Profit=₹{profit_loss_amount:.2f}, Balance: ₹{wallet_balance_before:.2f} → ₹{current_balance:.2f}")
        else:
            # RULE 4: LOSS → Loss = bet_amount (ALWAYS update balance and profit/loss)
            print(f"❌ RULE 4 (LOSS): pred_min={pred_min:.2f}x, actual={actual_multiplier:.2f}x, Bet=₹{bet_amount_per_strategy:.2f}, Loss=₹{abs(profit_loss_amount):.2f}, Balance: ₹{wallet_balance_before:.2f} → ₹{current_balance:.2f}")
            # Ensure loss is properly reflected (negative value)
            if profit_loss_amount >= 0:
                print(f"⚠️  ERROR: Loss should be negative! Correcting from ₹{profit_loss_amount:.2f} to ₹{-bet_amount_per_strategy:.2f}")
                profit_loss_amount = -bet_amount_per_strategy
                current_balance = wallet_balance_before + profit_loss_amount
                min_range_balance = current_balance
                print(f"✅ CORRECTED: Loss=₹{abs(profit_loss_amount):.2f}, Balance: ₹{wallet_balance_before:.2f} → ₹{current_balance:.2f}")
    else:
        print(f"⚠️  ERROR: min_range_result is None but we should have a result when pred_min > 2.0!")
        
        # CRITICAL: Ensure profit_loss is correctly calculated (negative for loss, positive for win)
        # Recalculate to be absolutely sure
        if min_range_result['is_win']:
            # WIN: profit_loss = payout - bet_amount (should be positive)
            recalculated_profit_loss = payout_amount - bet_amount_per_strategy
        else:
            # LOSS: profit_loss = -bet_amount (should be negative)
            recalculated_profit_loss = -bet_amount_per_strategy
        
        # Use recalculated value to ensure correctness
        if abs(profit_loss_amount - recalculated_profit_loss) > 0.01:
            print(f"⚠️  Recalculating profit_loss: Old=₹{profit_loss_amount:.2f}, New=₹{recalculated_profit_loss:.2f}")
            profit_loss_amount = recalculated_profit_loss
            # Recalculate balance
            current_balance = wallet_balance_before + profit_loss_amount
            min_range_balance = current_balance
        
        min_range_bet = {
            'timestamp': timestamp,
            'bet_type': 'min_range',
            'bet_placed': True,  # FLAG: Bet WAS placed (pred_min > 2.0)
            'predicted_range': pred_range,
            'pred_min': round(pred_min, 2),
            'pred_max': round(pred_max, 2),
            'predicted_value': round(pred_min, 2),  # Minimum range value used for betting
            'actual_multiplier': round(actual_multiplier, 2),
            'bet_amount': round(bet_amount_per_strategy, 2),
            'payout': round(payout_amount, 2),       # Total return when win (0 when loss)
            'profit_loss': round(profit_loss_amount, 2),  # Net change (ALWAYS updated when bet placed)
            'balance_before': round(wallet_balance_before, 2),  # Wallet balance before bet
            'balance_after': round(current_balance, 2),  # Wallet balance after bet (ALWAYS updated)
            'is_win': min_range_result['is_win'],
            'confidence': confidence,
            'ml_decision': ml_decision if ml_decision else None  # Store ML decision data
        }
        
        # Verify the bet record is correct before saving
        if min_range_bet['bet_placed'] and min_range_bet['profit_loss'] == 0 and min_range_bet['bet_amount'] > 0:
            print(f"⚠️  ERROR: Bet was placed but profit_loss is 0! This should not happen.")
            print(f"   Bet amount: ₹{min_range_bet['bet_amount']:.2f}, Is win: {min_range_bet['is_win']}, Payout: ₹{min_range_bet['payout']:.2f}")
        
        min_range_bets.append(min_range_bet)
        betting_history.append(min_range_bet)
        append_bet_to_csv(min_range_bet)
        
        # Final verification: Ensure balance was actually updated
        if abs(current_balance - wallet_balance_before) < 0.01 and profit_loss_amount != 0:
            print(f"⚠️  CRITICAL ERROR: Balance not updated after bet! Before: ₹{wallet_balance_before:.2f}, After: ₹{current_balance:.2f}, P/L: ₹{profit_loss_amount:.2f}")
            # Force update
            current_balance = wallet_balance_before + profit_loss_amount
            min_range_balance = current_balance
            # Update the bet record
            min_range_bet['balance_after'] = round(current_balance, 2)
            print(f"✅ FORCED Balance update in bet record: ₹{current_balance:.2f}")
    
    # Record prediction value bet (optional - separate strategy)
    if prediction_result:
        # Verify profit/loss calculation for prediction bet
        pred_profit_loss = prediction_result['profit_loss']
        pred_payout = prediction_result.get('payout', 0)
        expected_pred_profit = (pred_payout - bet_amount_per_strategy) if prediction_result['is_win'] else -bet_amount_per_strategy
        if abs(pred_profit_loss - expected_pred_profit) > 0.01:
            print(f"⚠️  WARNING: Prediction bet Profit/Loss mismatch! Expected: ₹{expected_pred_profit:.2f}, Got: ₹{pred_profit_loss:.2f}")
        
        prediction_bet = {
            'timestamp': timestamp,
            'bet_type': 'prediction',
            'bet_placed': True,  # FLAG: Bet WAS placed
            'predicted_range': pred_range,
            'pred_min': round(pred_min, 2),
            'pred_max': round(pred_max, 2),
            'predicted_value': round(predicted_value, 2),
            'actual_multiplier': round(actual_multiplier, 2),
            'bet_amount': round(bet_amount_per_strategy, 2),
            'payout': round(pred_payout, 2),
            'profit_loss': round(pred_profit_loss, 2),
            'balance_before': round(prediction_balance_before, 2),
            'balance_after': round(prediction_result['balance_after'], 2),
            'is_win': prediction_result['is_win'],
            'confidence': confidence
        }
        prediction_bets.append(prediction_bet)
        betting_history.append(prediction_bet)
        append_bet_to_csv(prediction_bet)
        
        # Log prediction bet result
        if prediction_result['is_win']:
            print(f"✅ Prediction Bet (WIN): predicted={predicted_value:.2f}x, actual={actual_multiplier:.2f}x, Profit=₹{pred_profit_loss:.2f}, Balance: ₹{prediction_balance_before:.2f} → ₹{prediction_result['balance_after']:.2f}")
        else:
            print(f"❌ Prediction Bet (LOSS): predicted={predicted_value:.2f}x, actual={actual_multiplier:.2f}x, Loss=₹{abs(pred_profit_loss):.2f}, Balance: ₹{prediction_balance_before:.2f} → ₹{prediction_result['balance_after']:.2f}")
    
    last_bet_timestamp = timestamp
    
    # Note: Calibration and Bayesian updates happen in check_and_process_bet after actual result is known
    
    # Return combined result
    return {
        'timestamp': timestamp,
        'min_range_bet': min_range_bet,
        'prediction_bet': prediction_bet,
        'total_profit_loss': (min_range_result['profit_loss'] if min_range_result else 0) + (prediction_result['profit_loss'] if prediction_result else 0),
        'ml_decision': ml_decision if ml_decision else None
    }

def check_and_process_h2o_bet():
    """Check for new actual multiplier and process H2O bet if prediction exists.
    
    Same logic as check_and_process_bet() but for H2O AutoML predictions.
    """
    global h2o_current_prediction, all_training_data, h2o_last_bet_record_count, processed_multipliers
    global h2o_min_range_bets, h2o_min_range_balance, base_bet_amount, max_balance
    
    if h2o_current_prediction is None or all_training_data is None or not betting_enabled:
        return
    
    if not H2O_AVAILABLE or h2o_model is None:
        return
    
    try:
        df = all_training_data
        if len(df) == 0:
            return
        
        # Get current record count
        current_record_count = len(df)
        
        # 🔧 BUG 1 FIX: Only process the latest finalized record
        # Process only the latest record to avoid multiple bets for one prediction
        if current_record_count <= h2o_last_bet_record_count:
            return  # No new records
        
        # Only process the latest record
        idx = current_record_count - 1
        
        # Get the actual multiplier for this record
        actual_multiplier = float(df['multiplier'].iloc[idx])
        
        # Get prediction range from H2O prediction
        pred_range = h2o_current_prediction.get('prediction_range', '')
        # Parse pred_min from range (format: "2.50 - 3.80")
        try:
            if ' - ' in pred_range:
                pred_min = float(pred_range.split(' - ')[0])
                pred_max = float(pred_range.split(' - ')[1])
            else:
                # Fallback: use predicted_multiplier
                pred_min = float(h2o_current_prediction.get('predicted_multiplier', 0))
                pred_max = pred_min
        except:
            pred_min = float(h2o_current_prediction.get('predicted_multiplier', 0))
            pred_max = pred_min
        
        # RULE 1: Only bet when pred_min > 2.0 (strictly greater than 2.0)
        if pred_min <= 2.0:
            print(f"⏸️  H2O: Bet SKIPPED - pred_min ({pred_min:.2f}x) <= 2.0x")
            
            # Save record with bet_placed=False
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            h2o_min_range_bet = {
                'timestamp': timestamp,
                'bet_type': 'h2o_min_range',
                'bet_placed': False,
                'predicted_range': pred_range,
                'pred_min': round(pred_min, 2),
                'pred_max': round(pred_max, 2),
                'predicted_value': round(h2o_current_prediction.get('predicted_multiplier', 0), 2),
                'actual_multiplier': round(actual_multiplier, 2),
                'bet_amount': 0.0,
                'payout': 0.0,
                'profit_loss': 0.0,
                'balance_before': round(h2o_min_range_balance, 2),
                'balance_after': round(h2o_min_range_balance, 2),
                'is_win': False,
                'confidence': h2o_current_prediction.get('confidence', 0)
            }
            h2o_min_range_bets.append(h2o_min_range_bet)
            append_bet_to_csv(h2o_min_range_bet)
            h2o_last_bet_record_count = idx + 1
            return
        
        # RULE 2: Bet when pred_min > 2.0
        print(f"✅ H2O: Bet PLACED - pred_min ({pred_min:.2f}x) > 2.0x")
        
        # Use H2O balance for the bet
        bet_amount_per_strategy = base_bet_amount
        wallet_balance_before = h2o_min_range_balance
        
        # Check if balance is sufficient
        if wallet_balance_before < bet_amount_per_strategy:
            print(f"⚠️  H2O: Insufficient balance: ₹{wallet_balance_before:.2f} < ₹{bet_amount_per_strategy:.2f}")
            h2o_last_bet_record_count = idx + 1
            return
        
        # Simulate bet
        min_range_result = simulate_bet_min_range(pred_min, actual_multiplier, bet_amount_per_strategy, wallet_balance_before)
        
        if min_range_result:
            # Update H2O balance
            old_balance = h2o_min_range_balance
            h2o_min_range_balance = min_range_result['balance_after']
            
            # Verify balance was updated
            if abs(h2o_min_range_balance - old_balance) < 0.01 and min_range_result['profit_loss'] != 0:
                h2o_min_range_balance = old_balance + min_range_result['profit_loss']
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            h2o_min_range_bet = {
                'timestamp': timestamp,
                'bet_type': 'h2o_min_range',
                'bet_placed': True,
                'predicted_range': pred_range,
                'pred_min': round(pred_min, 2),
                'pred_max': round(pred_max, 2),
                'predicted_value': round(h2o_current_prediction.get('predicted_multiplier', 0), 2),
                'actual_multiplier': round(actual_multiplier, 2),
                'bet_amount': round(bet_amount_per_strategy, 2),
                'payout': round(min_range_result.get('payout', 0), 2),
                'profit_loss': round(min_range_result['profit_loss'], 2),
                'balance_before': round(wallet_balance_before, 2),
                'balance_after': round(h2o_min_range_balance, 2),
                'is_win': min_range_result['is_win'],
                'confidence': h2o_current_prediction.get('confidence', 0)
            }
            h2o_min_range_bets.append(h2o_min_range_bet)
            append_bet_to_csv(h2o_min_range_bet)
            
            if min_range_result['is_win']:
                print(f"✅ H2O (WIN): pred_min={pred_min:.2f}x, actual={actual_multiplier:.2f}x, Profit=₹{min_range_result['profit_loss']:.2f}")
            else:
                print(f"❌ H2O (LOSS): pred_min={pred_min:.2f}x, actual={actual_multiplier:.2f}x, Loss=₹{abs(min_range_result['profit_loss']):.2f}")
        
        # 🔧 BUG 2 FIX: Update immediately after betting
        h2o_last_bet_record_count = idx + 1
        
    except Exception as e:
        print(f"⚠️  H2O betting error: {e}")
        import traceback
        traceback.print_exc()

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
    
    # 🔧 BUG 1 FIX: ONLY process the latest finalized record (1 prediction → 1 bet → 1 P/L update)
    # Do NOT loop over multiple records - this causes duplicate bets
    if current_record_count <= last_bet_record_count:
        return  # No new records to process
    
    # Process ONLY the latest record
    idx = current_record_count - 1
    actual_multiplier = float(df['multiplier'].iloc[idx])
    actual_timestamp = df['timestamp'].iloc[idx].strftime('%Y-%m-%d %H:%M:%S')
    
    # Create unique key for this multiplier
    multiplier_key = (actual_timestamp, actual_multiplier)
    
    # Skip if already processed
    if multiplier_key in processed_multipliers:
        return
    
    # Use the prediction that was available at the time (current prediction)
    bet_result = simulate_bet(current_prediction, actual_multiplier)
    
    # 🔧 BUG 2 FIX: Update last_bet_record_count IMMEDIATELY after betting (prevents race conditions)
    last_bet_record_count = idx + 1
    
    if bet_result:
        min_bet = bet_result.get('min_range_bet')
        pred_bet = bet_result.get('prediction_bet')
        
        if min_bet:
            status_min = "⭐ WIN" if min_bet['is_win'] else "LOSS"
            # Clarify that profit is calculated from minimum range value, not actual multiplier
            profit_source = f"Min Range ({min_bet['predicted_value']}x)" if min_bet['is_win'] else "Loss"
            print(f"💰 Min Range Bet: {status_min} | Bet: ₹{min_bet['bet_amount']:.2f} | Predicted Min: {min_bet['predicted_value']}x | Actual: {actual_multiplier}x | P/L: ₹{min_bet['profit_loss']:.2f} (from {profit_source}) | Balance: ₹{min_bet['balance_after']:.2f}")
            
            # Update calibration data if ML decision was used
            if min_bet.get('ml_decision'):
                ml_dec = min_bet['ml_decision']
                actual_outcome = 1.0 if min_bet['is_win'] else 0.0
                calibrate_probability(ml_dec.get('calibrated_probability', 0.5), actual_outcome)
                # Update Bayesian history
                bayesian_history.append(actual_outcome)
        
        if pred_bet:
            status_pred = "⭐ WIN" if pred_bet['is_win'] else "LOSS"
            print(f"💰 Prediction Bet: {status_pred} | Bet: ₹{pred_bet['bet_amount']:.2f} | Predicted: {pred_bet['predicted_value']}x | Actual: {actual_multiplier}x | P/L: ₹{pred_bet['profit_loss']:.2f} | Balance: ₹{pred_bet['balance_after']:.2f}")
        
        processed_multipliers.add(multiplier_key)

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
    """API endpoint to train the model - redirects to H2O AutoML."""
    return api_train_h2o()  # Use H2O AutoML instead

@app.route('/api/predict', methods=['GET'])
def api_predict():
    """API endpoint to get prediction - uses H2O AutoML."""
    if not H2O_AVAILABLE or h2o_model is None:
        return jsonify({
            'success': False,
            'error': 'H2O AutoML model not trained. Please train the model first.'
        })
    df, file_info = load_latest_csv()
    if df is None:
        return jsonify({
            'success': False,
            'error': f'No CSV file found: {file_info}'
        })
    prediction, error = predict_h2o_next(df)
    if error:
        return jsonify({
            'success': False,
            'error': error
        })
    return jsonify({
        'success': True,
        'prediction': prediction
    })

@app.route('/api/train_h2o', methods=['POST'])
def api_train_h2o():
    """API endpoint to manually train the H2O AutoML model."""
    if not H2O_AVAILABLE:
        return jsonify({
            'success': False,
            'message': 'H2O AutoML not available. Install with: pip install h2o'
        })
    
    df, file_info = load_latest_csv()
    
    if df is None:
        return jsonify({
            'success': False,
            'message': f'No CSV file found: {file_info}'
        })
    
    success, message = train_h2o_model(df, auto=False)
    
    return jsonify({
        'success': success,
        'message': message,
        'accuracy': h2o_accuracy if success else None
    })

@app.route('/api/status', methods=['GET'])
def api_status():
    """API endpoint to get current status (for real-time updates) - H2O AutoML only."""
    global h2o_training_in_progress, h2o_last_training_attempt
    
    df, file_info = load_latest_csv()
    
    # Check and process H2O bet if new data available (only H2O AutoML now)
    if H2O_AVAILABLE and h2o_model is not None and h2o_current_prediction is not None:
        check_and_process_h2o_bet()
    
    # Calculate total P&L for H2O min range bets (ONLY count bets that were actually placed)
    placed_h2o_min_range_bets = [bet for bet in h2o_min_range_bets if bet.get('bet_placed', False)]
    h2o_min_range_total_pl = sum(bet['profit_loss'] for bet in placed_h2o_min_range_bets)
    h2o_min_range_wins = sum(1 for bet in placed_h2o_min_range_bets if bet.get('is_win', False))
    h2o_min_range_losses = len(placed_h2o_min_range_bets) - h2o_min_range_wins
    h2o_min_range_win_rate = (h2o_min_range_wins / len(placed_h2o_min_range_bets) * 100) if placed_h2o_min_range_bets else 0
    
    # Calculate H2O performance trend
    h2o_performance_trend = None
    if H2O_AVAILABLE and len(h2o_performance_history) >= 5:
        recent_mae = [p['mae'] for p in list(h2o_performance_history)[-5:]]
        older_mae = [p['mae'] for p in list(h2o_performance_history)[-10:-5]] if len(h2o_performance_history) >= 10 else recent_mae
        avg_recent = np.mean(recent_mae)
        avg_older = np.mean(older_mae)
        if avg_older > 0:
            improvement_pct = ((avg_older - avg_recent) / avg_older) * 100
            h2o_performance_trend = {
                'improving': int(improvement_pct > 0),  # Convert bool to int for JSON
                'improvement_pct': round(improvement_pct, 2),
                'recent_avg_mae': round(avg_recent, 3),
                'older_avg_mae': round(avg_older, 3)
            }
    
    # Check if training is stuck (running for more than 5 minutes)
    if h2o_training_in_progress and h2o_last_training_attempt:
        time_since_attempt = (datetime.now() - h2o_last_training_attempt).total_seconds()
        if time_since_attempt > 300:  # 5 minutes
            print(f"⚠️ H2O training appears stuck (running for {time_since_attempt:.0f}s), resetting flag...")
            h2o_training_in_progress = False
    
    status = {
        'model_trained': int(H2O_AVAILABLE and h2o_model is not None),  # H2O AutoML only
        'model_accuracy': h2o_accuracy if H2O_AVAILABLE else None,  # H2O accuracy only
        'current_prediction': h2o_current_prediction if H2O_AVAILABLE else None,  # H2O prediction only
        'prediction_history': list(h2o_prediction_history) if H2O_AVAILABLE else [],  # H2O history only (all predictions)
        'auto_train_enabled': int(auto_train_enabled),  # Convert bool to int for JSON
        'csv_file': file_info,
        'csv_records': len(df) if df is not None else 0,
        'latest_multiplier': float(df['multiplier'].iloc[-1]) if df is not None and len(df) > 0 else None,
        'model_training_info': None,  # Removed Scikit-Learn training info
        'h2o_training_status': {
            'in_progress': int(h2o_training_in_progress),
            'last_attempt': h2o_last_training_attempt.strftime('%Y-%m-%d %H:%M:%S') if h2o_last_training_attempt else None,
            'has_enough_data': int(len(df) >= 30 if df is not None else 0)
        } if H2O_AVAILABLE else None,
        'betting': {
            'current_balance': round(h2o_min_range_balance, 2),
            'max_balance': max_balance,
            'base_bet_amount': base_bet_amount,
            'wallet_profit_loss': round(h2o_min_range_balance - max_balance, 2),
            'total_profit_loss': round(h2o_min_range_total_pl, 2),
            'total_bets': len(placed_h2o_min_range_bets),
            'total_wins': int(h2o_min_range_wins),
            'total_losses': int(h2o_min_range_losses),
            'total_win_rate': round(h2o_min_range_win_rate, 2),
            'betting_enabled': int(betting_enabled),  # Convert bool to int for JSON
            'h2o_min_range': {
                'balance': round(h2o_min_range_balance, 2),
                'total_profit_loss': round(h2o_min_range_total_pl, 2),
                'total_bets': len(placed_h2o_min_range_bets),
                'total_wins': int(h2o_min_range_wins),
                'total_losses': int(h2o_min_range_losses),
                'win_rate': round(h2o_min_range_win_rate, 2)
            }
        },
        'h2o_automl': {
            'available': int(H2O_AVAILABLE),  # Convert bool to int for JSON
            'model_trained': int(h2o_model is not None if H2O_AVAILABLE else False),  # Convert bool to int for JSON
            'model_accuracy': h2o_accuracy,
            'current_prediction': h2o_current_prediction,
            'prediction_history': list(h2o_prediction_history) if H2O_AVAILABLE else [],  # All predictions
            'training_info': {
                'retrain_interval': h2o_retrain_interval if H2O_AVAILABLE else 10,
                'records_since_last_retrain': len(df) - h2o_last_retrain_record_count if H2O_AVAILABLE and df is not None and h2o_last_retrain_record_count > 0 else (len(df) if H2O_AVAILABLE and df is not None else 0),
                'total_training_events': len(h2o_training_history) if H2O_AVAILABLE else 0,
                'last_training': h2o_training_history[-1] if H2O_AVAILABLE and h2o_training_history else None,
                'performance_trend': h2o_performance_trend,
                'training_history_recent': h2o_training_history[-5:] if H2O_AVAILABLE and h2o_training_history else [],
                'maturity_info': {
                    'maturity_level': h2o_accuracy.get('maturity_level', 'Unknown') if H2O_AVAILABLE and h2o_accuracy else 'Unknown',
                    'maturity_pct': h2o_accuracy.get('maturity_pct', 0) if H2O_AVAILABLE and h2o_accuracy else 0,
                    'current_records': len(df) if df is not None else 0
                }
            } if H2O_AVAILABLE else None
        }
    }
    
    return jsonify(status)

@app.route('/api/betting_history', methods=['GET'])
def api_betting_history():
    """API endpoint to get betting history."""
    # Convert betting history to JSON-serializable format (convert bool to int)
    def make_json_serializable(obj):
        """Recursively convert all booleans to ints for JSON serialization."""
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        elif isinstance(obj, bool):
            return int(obj)  # Convert bool to int (0 or 1)
        elif isinstance(obj, (int, float, str, type(None))):
            return obj
        else:
            # For any other type, try to convert to string
            return str(obj)
    
    # Return all bets (not limited)
    history_serializable = [make_json_serializable(bet) for bet in betting_history]
    min_range_serializable = [make_json_serializable(bet) for bet in min_range_bets]
    prediction_serializable = [make_json_serializable(bet) for bet in prediction_bets]
    h2o_min_range_serializable = [make_json_serializable(bet) for bet in h2o_min_range_bets]
    
    return jsonify({
        'success': True,
        'history': history_serializable,
        'min_range_bets': min_range_serializable,
        'prediction_bets': prediction_serializable,
        'h2o_min_range_bets': h2o_min_range_serializable,
        'summary': {
            'current_balance': round(current_balance, 2),
            'total_profit_loss': round(sum(bet['profit_loss'] for bet in betting_history if bet.get('bet_placed', False)), 2),
            'total_bets': sum(1 for bet in betting_history if bet.get('bet_placed', False)),
            'total_wins': sum(1 for bet in betting_history if bet.get('bet_placed', False) and bet.get('is_win', False)),
            'total_losses': sum(1 for bet in betting_history if bet.get('bet_placed', False) and not bet.get('is_win', False)),
            'total_win_rate': round((sum(1 for bet in betting_history if bet.get('bet_placed', False) and bet.get('is_win', False)) / sum(1 for bet in betting_history if bet.get('bet_placed', False)) * 100) if sum(1 for bet in betting_history if bet.get('bet_placed', False)) > 0 else 0, 2),
            'min_range': {
                'balance': round(min_range_balance, 2),
                'total_profit_loss': round(sum(bet['profit_loss'] for bet in min_range_bets if bet.get('bet_placed', False)), 2),
                'total_bets': sum(1 for bet in min_range_bets if bet.get('bet_placed', False)),
                'total_wins': sum(1 for bet in min_range_bets if bet.get('bet_placed', False) and bet.get('is_win', False)),
                'total_losses': sum(1 for bet in min_range_bets if bet.get('bet_placed', False) and not bet.get('is_win', False)),
                'win_rate': round((sum(1 for bet in min_range_bets if bet.get('bet_placed', False) and bet.get('is_win', False)) / sum(1 for bet in min_range_bets if bet.get('bet_placed', False)) * 100) if sum(1 for bet in min_range_bets if bet.get('bet_placed', False)) > 0 else 0, 2)
            },
            'prediction': {
                'balance': round(prediction_balance, 2),
                'total_profit_loss': round(sum(bet['profit_loss'] for bet in prediction_bets if bet.get('bet_placed', False)), 2),
                'total_bets': sum(1 for bet in prediction_bets if bet.get('bet_placed', False)),
                'total_wins': sum(1 for bet in prediction_bets if bet.get('bet_placed', False) and bet.get('is_win', False)),
                'total_losses': sum(1 for bet in prediction_bets if bet.get('bet_placed', False) and not bet.get('is_win', False)),
                'win_rate': round((sum(1 for bet in prediction_bets if bet.get('bet_placed', False) and bet.get('is_win', False)) / sum(1 for bet in prediction_bets if bet.get('bet_placed', False)) * 100) if sum(1 for bet in prediction_bets if bet.get('bet_placed', False)) > 0 else 0, 2)
            }
        }
    })

@app.route('/api/ml_probability', methods=['GET'])
def api_ml_probability():
    """API endpoint to get ML probability decision for current prediction."""
    global current_prediction, all_training_data, current_balance, base_bet_amount
    
    if current_prediction is None:
        return jsonify({
            'success': False,
            'error': 'No prediction available'
        })
    
    df = all_training_data if all_training_data is not None else load_latest_csv()[0]
    
    if df is None or len(df) < 20:
        return jsonify({
            'success': False,
            'error': 'Insufficient data for ML probability system'
        })
    
    try:
        ml_decision = ml_probability_decision(current_prediction, df, current_balance, base_bet_amount)
        
        # Make decision factors JSON-serializable
        decision_factors = {k: bool(v) for k, v in ml_decision['decision_factors'].items()}
        
        return jsonify({
            'success': True,
            'ml_decision': {
                'should_bet': ml_decision['should_bet'],
                'optimal_bet': ml_decision['optimal_bet'],
                'raw_probability': ml_decision['raw_probability'],
                'bayesian_probability': ml_decision['bayesian_probability'],
                'calibrated_probability': ml_decision['calibrated_probability'],
                'regime': ml_decision['regime'],
                'confidence': ml_decision['confidence'],
                'ev': ml_decision['ev'],
                'ev_percentage': ml_decision['ev_percentage'],
                'sharpe': ml_decision['sharpe'],
                'kelly_fraction': ml_decision['kelly_fraction'],
                'kelly_bet': ml_decision['kelly_bet'],
                'monte_carlo_risk': ml_decision['monte_carlo_risk'],
                'monte_carlo_risk_level': ml_decision['monte_carlo_risk_level'],
                'stop_loss_triggered': ml_decision['stop_loss_triggered'],
                'take_profit_triggered': ml_decision['take_profit_triggered'],
                'variance_circuit_breaker': ml_decision['variance_circuit_breaker'],
                'variance_ratio': ml_decision['variance_ratio'],
                'decision_factors': decision_factors,
                'positive_factors': ml_decision['positive_factors'],
                'total_factors': ml_decision['total_factors'],
                'decision_score': ml_decision['decision_score']
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/reset_betting', methods=['POST'])
def api_reset_betting():
    """API endpoint to reset betting simulation."""
    global current_balance, betting_history, last_bet_timestamp, last_bet_record_count, processed_multipliers
    global min_range_bets, prediction_bets, min_range_balance, prediction_balance
    global peak_balance, starting_balance, bayesian_history, calibration_data
    global h2o_min_range_bets, h2o_min_range_balance, h2o_last_bet_record_count
    current_balance = max_balance
    min_range_balance = max_balance
    prediction_balance = max_balance
    peak_balance = max_balance
    starting_balance = max_balance
    betting_history = []
    min_range_bets = []
    prediction_bets = []
    last_bet_timestamp = None
    last_bet_record_count = 0
    processed_multipliers = set()
    bayesian_history.clear()
    calibration_data.clear()
    # Reset H2O betting
    h2o_min_range_bets = []
    h2o_min_range_balance = max_balance
    h2o_last_bet_record_count = 0
    return jsonify({
        'success': True,
        'message': 'Betting simulation reset (including H2O)',
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
    """Process new data immediately when CSV is updated - H2O AutoML only."""
    global auto_train_enabled, all_training_data, last_prediction_multiplier, last_training_record_count
    global h2o_last_retrain_record_count, h2o_current_prediction
    
    try:
        updated, df, file_info = check_csv_updated()
        
        if updated and df is not None:
            # Update all_training_data for betting system
            all_training_data = df.copy()
            
            # Calculate new records count
            new_records_count = len(df) - last_training_record_count if last_training_record_count > 0 else len(df)
            has_new_records = new_records_count > 0
            
            # Get the latest multiplier (most recent outcome)
            latest_multiplier = None
            if len(df) > 0:
                latest_multiplier = float(df['multiplier'].iloc[-1])
            
            # Check if multiplier changed (for logging)
            multiplier_changed = False
            if latest_multiplier is not None:
                if last_prediction_multiplier is None:
                    multiplier_changed = True
                elif latest_multiplier != last_prediction_multiplier:
                    multiplier_changed = True
                    print(f"🎯 New outcome finalized: {latest_multiplier}x (previous: {last_prediction_multiplier}x)")
            
            print(f"🔄 New data detected in {file_info} ({len(df)} total records, {new_records_count} new)")
            
            # Auto-train H2O AutoML if enabled
            if auto_train_enabled:
                # Train H2O AutoML if model doesn't exist and we have enough data
                if H2O_AVAILABLE and h2o_model is None and len(df) >= 30:
                    def train_h2o_first_time():
                        print("🤖 Training H2O AutoML model with all available data...")
                        success, msg = train_h2o_model(df, auto=True)
                        if success:
                            # Predict after training
                            h2o_prediction, h2o_error = predict_h2o_next(df)
                            if not h2o_error:
                                print(f"✅ H2O AutoML ready: {h2o_prediction['predicted_multiplier']}x")
                                global last_prediction_multiplier
                                if latest_multiplier is not None:
                                    last_prediction_multiplier = latest_multiplier
                    threading.Thread(target=train_h2o_first_time, daemon=True).start()
                # If H2O model exists, predict when we have new records
                elif H2O_AVAILABLE and h2o_model is not None:
                    # Predict whenever we have new records (not just when multiplier changes)
                    if has_new_records:
                        print(f"🎯 New records detected ({new_records_count} new) → Predicting with H2O AutoML...")
                        
                        # Update all_training_data first
                        all_training_data = df.copy()
                        
                        # H2O AutoML Prediction
                        def predict_h2o_thread():
                            try:
                                h2o_prediction, h2o_error = predict_h2o_next(df)
                                if not h2o_error:
                                    print(f"✅ H2O AutoML Prediction: {h2o_prediction['predicted_multiplier']}x (Range: {h2o_prediction['prediction_range']}, confidence: {h2o_prediction['confidence']}%)")
                                    # Update last prediction multiplier
                                    global last_prediction_multiplier
                                    if latest_multiplier is not None:
                                        last_prediction_multiplier = latest_multiplier
                                    # Process H2O bet (non-blocking)
                                    threading.Thread(target=check_and_process_h2o_bet, daemon=True).start()
                                else:
                                    print(f"⚠️ H2O prediction error: {h2o_error}")
                            except Exception as e:
                                print(f"⚠️ H2O prediction thread error: {e}")
                        
                        # Start prediction thread
                        h2o_thread = threading.Thread(target=predict_h2o_thread, daemon=True)
                        h2o_thread.start()
                        print(f"🚀 Started H2O AutoML prediction thread")
                        
                        # Retrain H2O AutoML in background (periodically)
                        h2o_records_since_retrain = len(df) - h2o_last_retrain_record_count
                        if len(df) >= 30 and h2o_records_since_retrain >= h2o_retrain_interval:
                            def retrain_h2o_in_background():
                                print(f"🤖 Retraining H2O AutoML in background ({h2o_records_since_retrain} new records)...")
                                success, msg = train_h2o_model(df, auto=True)
                                if success:
                                    global h2o_last_retrain_record_count
                                    h2o_last_retrain_record_count = len(df)
                                    print(f"✅ H2O AutoML retrained in background")
                                else:
                                    print(f"⚠️ H2O background retraining failed: {msg}")
                            
                            threading.Thread(target=retrain_h2o_in_background, daemon=True).start()
                    else:
                        print(f"⏸️  No new records detected (current: {len(df)} records, last processed: {last_training_record_count})")
                # If H2O model exists but not enough data, still try to predict after new result
                elif H2O_AVAILABLE and h2o_model is not None and len(df) >= 15:
                    if has_new_records:
                        print(f"🎯 New records detected ({new_records_count} new) → Predicting with H2O AutoML...")
                        
                        def predict_h2o_early():
                            try:
                                h2o_prediction, h2o_error = predict_h2o_next(df)
                                if not h2o_error:
                                    print(f"✅ H2O Early Prediction: {h2o_prediction['predicted_multiplier']}x")
                                    global last_prediction_multiplier
                                    if latest_multiplier is not None:
                                        last_prediction_multiplier = latest_multiplier
                            except Exception as e:
                                print(f"⚠️ H2O early prediction error: {e}")
                        
                        threading.Thread(target=predict_h2o_early, daemon=True).start()
            
            # Update the processed record count after handling new data
            last_training_record_count = len(df)
    except Exception as e:
        print(f"⚠️ Error processing new data: {e}")
        import traceback
        traceback.print_exc()

def background_worker():
    """Background worker that checks for CSV updates and auto-trains/predicts - FAST POLLING."""
    global auto_train_enabled
    
    print("🔄 Background worker polling every 0.2 seconds for CSV updates (FAST MODE)...")
    
    last_fallback_check = 0
    fallback_check_interval = 5  # Run fallback check every 5 seconds instead of every 0.2 seconds
    
    while True:
        try:
            # Check for updates continuously (this is the primary detection method)
            updated, df, file_info = check_csv_updated()
            
            if updated and df is not None:
                print(f"🔔 Background worker detected update in {file_info}")
                # Process immediately
                process_new_data_immediately()
            else:
                # Fallback check: Only run occasionally (every 5 seconds) to avoid spam
                current_time = time.time()
                if current_time - last_fallback_check >= fallback_check_interval:
                    last_fallback_check = current_time
                    # Even if not detected as updated, check if we should update prediction
                    # (fallback for edge cases - only check, don't spam logs)
                    if H2O_AVAILABLE and h2o_model is not None:
                        df, file_info = load_latest_csv()
                        if df is not None:
                            # Silently check - process_new_data_immediately will handle logging if needed
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
    global peak_balance, starting_balance
    
    print("💰 Initializing Profit & Loss Tracking System...")
    print(f"   Starting Balance (Each Strategy): ₹{max_balance:,.2f}")
    print(f"   Base Bet Amount: ₹{base_bet_amount:,.2f}")
    print(f"   Strategies: Minimum Range Value & Prediction Value")
    
    # RESET all betting data to zero when app starts
    current_balance = max_balance
    min_range_balance = max_balance
    prediction_balance = max_balance
    peak_balance = max_balance  # Initialize peak balance
    starting_balance = max_balance  # Initialize starting balance
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
        
        # Auto-train H2O AutoML on startup if enough data (use ALL data)
        if H2O_AVAILABLE and len(df) >= 30:
            print(f"🤖 Auto-training H2O AutoML model on startup with ALL {len(df)} records...")
            success, msg = train_h2o_model(df, auto=True)
            if success:
                h2o_prediction, h2o_error = predict_h2o_next(df)
                if not h2o_error and len(df) > 0:
                    # Initialize last prediction multiplier with latest outcome
                    global last_prediction_multiplier
                    last_prediction_multiplier = float(df['multiplier'].iloc[-1])
                    print(f"📌 H2O AutoML prediction initialized for next outcome (last finalized: {last_prediction_multiplier}x)")
            else:
                print(f"⚠️ H2O AutoML training failed: {msg}")
            # NOTE: Do NOT call check_and_process_h2o_bet() here - we only want to track NEW bets
            # The betting system is already initialized with last_bet_record_count = len(df)
            # So it will only process records that arrive AFTER app startup
            print("💰 Betting system ready - will track P/L for new multipliers only (from app startup)")
        elif H2O_AVAILABLE and len(df) >= 15:
            print(f"⚠️  Only {len(df)} records available. Need at least 30 for optimal H2O AutoML training.")
            # Still try to train with available data
            try:
                train_h2o_model(df, auto=True)
                if h2o_model:
                    predict_h2o_next(df)
                    # NOTE: Do NOT call check_and_process_h2o_bet() here - only track new bets
                    print("💰 Betting system ready - will track P/L for new multipliers only (from app startup)")
            except:
                pass
        else:
            if not H2O_AVAILABLE:
                print(f"⚠️ H2O AutoML not available. Install with: pip install h2o")
            else:
                print(f"⏳ Waiting for more data (need at least 15 records for H2O AutoML training)...")
    
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
    print("💡 Using H2O AutoML (Best Model Selection)")
    print("💡 Features: Advanced time-series features with instant data sync")
    print("🔄 Auto-training and prediction updates enabled")
    print("💰 Profit & Loss tracking: ACTIVE (Starting from ₹50,000)")
    print("=" * 60)
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5001, use_reloader=False)
    except KeyboardInterrupt:
        print("\n⚠️  Shutting down...")
        observer.stop()
    observer.join()
