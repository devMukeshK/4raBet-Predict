"""
H2O AutoML Model for Multiplier Prediction - Dual Approach System
1. Binary Classification: Predicts probability of next multiplier > 2.0 (prob > 0.7 threshold)
2. Multi-Class Classification + LLM Reasoning: 3 classes (<1.5x, 1.5x-3x, >3x) with LLM decision layer
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import warnings
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, log_loss, precision_score, recall_score, f1_score
warnings.filterwarnings('ignore')

# LLM setup
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️ LangChain OpenAI not available. Install with: pip install langchain-openai")

# H2O AutoML setup
try:
    import h2o
    from h2o.automl import H2OAutoML
    H2O_AVAILABLE = True
except ImportError:
    H2O_AVAILABLE = False
    print("⚠️ H2O AutoML not available. Install with: pip install h2o")


class MinimalFeatureEngineer:
    """Minimal, clean feature engineering - only strongest signals"""
    
    def __init__(self, lookback=30, prediction_threshold=2.0):
        self.lookback = lookback
        self.prediction_threshold = prediction_threshold
        self.scaler = RobustScaler()
        self.feature_columns = []
        self.target_column = 'target_gt_2'  # Classification target
        self.is_fitted = False
        
    def create_features(self, df):
        """Create minimal, clean features - only strongest signals"""
        df = df.copy()
        
        # Ensure proper data types
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Handle outliers (cap at 99th percentile)
        p99 = df['multiplier'].quantile(0.99)
        p01 = df['multiplier'].quantile(0.01)
        df['multiplier_capped'] = df['multiplier'].clip(lower=p01, upper=p99)
        
        # ========== KEEP ONLY: LAG FEATURES ==========
        for lag in [1, 2, 3]:
            df[f'lag_{lag}'] = df['multiplier_capped'].shift(lag)
        
        # ========== KEEP ONLY: ROLLING STATS (fixed min_periods) ==========
        KEEP_WINDOWS = [5, 10, 20]
        for window in KEEP_WINDOWS:
            # FIX: Use min_periods=window to prevent leakage
            df[f'rolling_mean_{window}'] = df['multiplier_capped'].rolling(window=window, min_periods=window).mean()
            df[f'rolling_std_{window}'] = df['multiplier_capped'].rolling(window=window, min_periods=window).std()
        
        # ========== KEEP ONLY: EMA ==========
        for span in [5, 10]:
            df[f'ema_{span}'] = df['multiplier_capped'].ewm(span=span, adjust=False).mean()
        
        # ========== KEEP ONLY: PATTERN FEATURES ==========
        # Streak detection
        df['is_high'] = (df['multiplier_capped'] > self.prediction_threshold).astype(int)
        df['high_streak'] = df['is_high'].groupby((df['is_high'] != df['is_high'].shift()).cumsum()).cumsum()
        df['is_low'] = (df['multiplier_capped'] < 1.5).astype(int)
        df['low_streak'] = df['is_low'].groupby((df['is_low'] != df['is_low'].shift()).cumsum()).cumsum()
        
        # Time since last high/low
        df['time_since_high'] = 0
        df['time_since_low'] = 0
        last_high_idx = 0
        last_low_idx = 0
        for i in range(1, len(df)):
            if df['is_high'].iloc[i]:
                last_high_idx = i
            if df['is_low'].iloc[i]:
                last_low_idx = i
            df.loc[i, 'time_since_high'] = i - last_high_idx
            df.loc[i, 'time_since_low'] = i - last_low_idx
        
        # ========== KEEP ONLY: PROBABILITY FEATURES ==========
        for window in [10, 20]:
            df[f'prob_gt_{self.prediction_threshold}_{window}'] = df['is_high'].rolling(window=window, min_periods=window).mean()
        
        # ========== KEEP ONLY: VOLATILITY RATIO ==========
        if 'rolling_std_5' in df.columns and 'rolling_std_20' in df.columns:
            df['volatility_ratio'] = df['rolling_std_5'] / (df['rolling_std_20'] + 1e-10)
        else:
            df['volatility_ratio'] = 1.0
        
        # ========== CLASSIFICATION TARGET (NOT REGRESSION) ==========
        # Predict probability of next multiplier > threshold
        df[self.target_column] = (df['multiplier_capped'].shift(-1) > self.prediction_threshold).astype(int)
        
        # Remove rows with NaN values in target or features
        df = df.dropna().reset_index(drop=True)
        
        # Store feature columns (exclude target and metadata)
        exclude_cols = ['timestamp', 'multiplier', 'multiplier_capped', 
                       self.target_column, 'is_high', 'is_low']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        return df
    
    def scale_features(self, df, fit=True):
        """Scale features using RobustScaler"""
        if len(self.feature_columns) == 0:
            return df
        
        # Get feature values
        feature_data = df[self.feature_columns].values
        
        # Handle NaN values
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        if fit:
            scaled_features = self.scaler.fit_transform(feature_data)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler not fitted. Call scale_features with fit=True first.")
            scaled_features = self.scaler.transform(feature_data)
        
        # Create new dataframe with scaled features
        df_scaled = df.copy()
        df_scaled[self.feature_columns] = scaled_features
        
        return df_scaled
    
    def save_scaler(self, path):
        """Save scaler to file"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_scaler(self, path):
        """Load scaler from file"""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.is_fitted = True
            return True
        return False


def train_h2o_automl(data, config, progress_callback=None):
    """
    Train H2O AutoML CLASSIFICATION model for multiplier prediction.
    
    Args:
        data: DataFrame with 'timestamp' and 'multiplier' columns
        config: Config object with training parameters
        progress_callback: Optional callback function
    
    Returns:
        dict with model, feature_engineer, metrics, etc.
    """
    if not H2O_AVAILABLE:
        return None
    
    # Initialize H2O if not already initialized
    try:
        if not h2o.cluster().is_running():
            h2o.init()
            print("✅ H2O initialized")
    except:
        try:
            h2o.init()
            print("✅ H2O initialized")
        except Exception as e:
            print(f"❌ H2O initialization failed: {e}")
            return None
    
    print(f"🔄 Preparing features for H2O AutoML Classification...")
    
    # Initialize feature engineer
    feature_engineer = MinimalFeatureEngineer(
        lookback=config.LOOKBACK,
        prediction_threshold=2.0
    )
    
    # Create features
    df_with_features = feature_engineer.create_features(data)
    
    if len(df_with_features) < config.MIN_SAMPLES_FOR_TRAINING:
        print(f"⚠️ Insufficient data: {len(df_with_features)} samples")
        return None
    
    print(f"📊 Features created: {len(feature_engineer.feature_columns)} features, {len(df_with_features)} samples")
    
    # Scale features
    df_scaled = feature_engineer.scale_features(df_with_features, fit=True)
    
    # Add split column for time-series aware splitting (preserve order)
    total_rows = len(df_scaled)
    train_size = int(total_rows * 0.8)
    df_scaled['_split'] = ['train'] * train_size + ['valid'] * (total_rows - train_size)
    
    # Convert to H2O Frame
    h2o_df = h2o.H2OFrame(df_scaled)
    
    # Define features and target
    features = feature_engineer.feature_columns
    target = feature_engineer.target_column
    
    # Convert target to factor (required for classification)
    h2o_df[target] = h2o_df[target].asfactor()
    
    # Split data (time-series aware - preserve order)
    train = h2o_df[h2o_df['_split'] == 'train', :]
    valid = h2o_df[h2o_df['_split'] == 'valid', :]
    
    # Remove split column from features list (it's not a real feature)
    features_clean = [f for f in features if f != '_split']
    
    print(f"📈 Training data: {h2o_df.nrows} rows, {len(features_clean)} features")
    print(f"📊 Train samples: {train.nrows}, Validation samples: {valid.nrows}")
    
    # Run AutoML - CLASSIFICATION MODE
    max_runtime_secs = getattr(config, 'H2O_MAX_RUNTIME_SECS', 300)
    max_models = getattr(config, 'H2O_MAX_MODELS', 20)
    
    aml = H2OAutoML(
        max_models=max_models,
        max_runtime_secs=max_runtime_secs,
        seed=42,
        balance_classes=True,  # Important for imbalanced data
        stopping_metric="AUC",  # Classification metric
        sort_metric="AUC",
        nfolds=0,  # IMPORTANT: No cross-validation for time series
        verbosity="info"
    )
    
    print(f"🚀 Starting H2O AutoML Classification training (max {max_runtime_secs}s, max {max_models} models)...")
    start_time = datetime.now()
    
    # Train
    aml.train(
        x=features_clean,
        y=target,
        training_frame=train,
        validation_frame=valid
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Get best model
    best_model = aml.leader
    leaderboard = aml.leaderboard.as_data_frame()
    
    # Evaluate on validation set - CLASSIFICATION METRICS
    perf = best_model.model_performance(valid)
    
    # Get predictions for detailed metrics
    predictions = best_model.predict(valid)
    pred_probs = predictions['p1'].as_data_frame().values.flatten()
    y_true = valid[target].as_data_frame().astype(int).values.flatten()
    
    # Calculate metrics
    auc = perf.auc()
    logloss = perf.logloss()
    
    # Binary predictions at 0.5 threshold
    y_pred_binary = (pred_probs >= 0.5).astype(int)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    accuracy = np.mean(y_true == y_pred_binary)
    
    metrics = {
        'auc': round(auc, 4),
        'logloss': round(logloss, 4),
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'training_time_sec': round(training_time, 2),
        'total_records': len(data),
        'train_samples': train.nrows,
        'test_samples': valid.nrows,
        'features_used': len(features_clean),
        'best_model': best_model.model_id
    }
    
    print(f"✅ H2O AutoML Classification training complete!")
    print(f"   Best Model: {best_model.model_id}")
    print(f"   AUC: {auc:.4f} (target: 0.55-0.62)")
    print(f"   LogLoss: {logloss:.4f}")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"   Training time: {training_time:.2f} seconds")
    
    return {
        'model': best_model,
        'feature_engineer': feature_engineer,
        'metrics': metrics,
        'leaderboard': leaderboard,
        'input_dim': len(features_clean),
        'task_type': 'classification'
    }


def predict_h2o_automl(model_dict, data, lookback=30):
    """
    Predict probability using H2O AutoML CLASSIFICATION model.
    
    Args:
        model_dict: Dict with 'model', 'feature_engineer'
        data: DataFrame with 'timestamp' and 'multiplier' columns
        lookback: Lookback window size
    
    Returns:
        float: Probability that next multiplier > threshold (0.0 to 1.0)
    """
    if model_dict is None or 'model' not in model_dict:
        return None
    
    model = model_dict['model']
    feature_engineer = model_dict['feature_engineer']
    
    # Create features for latest data
    df_features = feature_engineer.create_features(data)
    
    # Scale features
    df_scaled = feature_engineer.scale_features(df_features, fit=False)
    
    # Get the latest row for prediction
    if len(df_scaled) == 0:
        return None
    
    latest_features = df_scaled.iloc[-1:][feature_engineer.feature_columns]
    
    # Convert to H2O Frame
    h2o_features = h2o.H2OFrame(latest_features)
    
    # Make prediction (classification returns probabilities)
    prediction = model.predict(h2o_features)
    
    # Extract probability of class 1 (multiplier > threshold)
    prob = prediction['p1'].as_data_frame().iloc[0, 0]
    
    return float(prob)


# ============================================================================
# MULTI-CLASS CLASSIFICATION MODEL (ChatGPT Approach)
# ============================================================================

class StrictFeatureEngineer:
    """Strict feature engineering for multi-class - only strongest signals"""
    
    def __init__(self, lookback=30):
        self.lookback = lookback
        self.scaler = RobustScaler()
        self.feature_columns = []
        self.target_column = 'target'  # Multi-class target
        self.is_fitted = False
        
    def create_features(self, df):
        """Create strict features - only lag, rolling stats, EMA, volatility ratio, streaks"""
        df = df.copy()
        
        # Ensure proper data types
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Cap outliers
        df['multiplier'] = df['multiplier'].clip(0.5, 10.0)
        
        # ========== LAG FEATURES ==========
        df['lag_1'] = df['multiplier'].shift(1)
        df['lag_2'] = df['multiplier'].shift(2)
        
        # ========== ROLLING STATS (fixed min_periods) ==========
        df['rolling_mean_10'] = df['multiplier'].rolling(10, min_periods=10).mean()
        df['rolling_std_10'] = df['multiplier'].rolling(10, min_periods=10).std()
        
        # ========== EMA ==========
        df['ema_10'] = df['multiplier'].ewm(span=10, adjust=False).mean()
        
        # ========== VOLATILITY RATIO ==========
        rolling_std_20 = df['multiplier'].rolling(20, min_periods=20).std()
        df['volatility_ratio'] = df['rolling_std_10'] / (rolling_std_20 + 1e-6)
        
        # ========== STREAK FEATURES ==========
        df['is_high'] = (df['multiplier'] > 2.0).astype(int)
        df['is_low'] = (df['multiplier'] < 1.5).astype(int)
        df['high_streak'] = df['is_high'].groupby((df['is_high'] != df['is_high'].shift()).cumsum()).cumsum()
        df['low_streak'] = df['is_low'].groupby((df['is_low'] != df['is_low'].shift()).cumsum()).cumsum()
        
        # ========== MULTI-CLASS TARGET ==========
        # Class 0: <1.5x, Class 1: 1.5x-3x, Class 2: >3x
        next_mult = df['multiplier'].shift(-1)
        df[self.target_column] = 0
        df.loc[next_mult >= 1.5, self.target_column] = 1
        df.loc[next_mult >= 3.0, self.target_column] = 2
        
        # Remove rows with NaN values
        df = df.dropna().reset_index(drop=True)
        
        # Store feature columns (exclude target and metadata)
        exclude_cols = ['timestamp', 'multiplier', self.target_column, 'is_high', 'is_low']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        return df
    
    def scale_features(self, df, fit=True):
        """Scale features using RobustScaler"""
        if len(self.feature_columns) == 0:
            return df
        
        feature_data = df[self.feature_columns].values
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        if fit:
            scaled_features = self.scaler.fit_transform(feature_data)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler not fitted. Call scale_features with fit=True first.")
            scaled_features = self.scaler.transform(feature_data)
        
        df_scaled = df.copy()
        df_scaled[self.feature_columns] = scaled_features
        return df_scaled
    
    def save_scaler(self, path):
        """Save scaler to file"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_scaler(self, path):
        """Load scaler from file"""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.is_fitted = True
            return True
        return False


def train_h2o_multiclass(data, config, progress_callback=None):
    """
    Train H2O AutoML MULTI-CLASS model.
    
    Args:
        data: DataFrame with 'timestamp' and 'multiplier' columns
        config: Config object with training parameters
        progress_callback: Optional callback function
    
    Returns:
        dict with model, feature_engineer, metrics, etc.
    """
    if not H2O_AVAILABLE:
        return None
    
    # Initialize H2O if not already initialized
    try:
        if not h2o.cluster().is_running():
            h2o.init()
            print("✅ H2O initialized")
    except:
        try:
            h2o.init()
            print("✅ H2O initialized")
        except Exception as e:
            print(f"❌ H2O initialization failed: {e}")
            return None
    
    print(f"🔄 Preparing features for H2O AutoML Multi-Class Classification...")
    
    # Initialize feature engineer
    feature_engineer = StrictFeatureEngineer(lookback=config.LOOKBACK)
    
    # Create features
    df_with_features = feature_engineer.create_features(data)
    
    if len(df_with_features) < config.MIN_SAMPLES_FOR_TRAINING:
        print(f"⚠️ Insufficient data: {len(df_with_features)} samples")
        return None
    
    print(f"📊 Features created: {len(feature_engineer.feature_columns)} features, {len(df_with_features)} samples")
    
    # Scale features
    df_scaled = feature_engineer.scale_features(df_with_features, fit=True)
    
    # Add split column for time-series aware splitting
    total_rows = len(df_scaled)
    train_size = int(total_rows * 0.8)
    df_scaled['_split'] = ['train'] * train_size + ['valid'] * (total_rows - train_size)
    
    # Convert to H2O Frame
    h2o_df = h2o.H2OFrame(df_scaled)
    
    # Define features and target
    features = feature_engineer.feature_columns
    target = feature_engineer.target_column
    
    # Convert target to factor (required for classification)
    h2o_df[target] = h2o_df[target].asfactor()
    
    # Split data (time-series aware)
    train = h2o_df[h2o_df['_split'] == 'train', :]
    valid = h2o_df[h2o_df['_split'] == 'valid', :]
    
    # Remove split column from features list
    features_clean = [f for f in features if f != '_split']
    
    print(f"📈 Training data: {h2o_df.nrows} rows, {len(features_clean)} features")
    print(f"📊 Train samples: {train.nrows}, Validation samples: {valid.nrows}")
    
    # Run AutoML - MULTI-CLASS MODE
    max_runtime_secs = getattr(config, 'H2O_MAX_RUNTIME_SECS', 300)
    max_models = getattr(config, 'H2O_MAX_MODELS', 20)
    
    aml = H2OAutoML(
        max_models=max_models,
        max_runtime_secs=max_runtime_secs,
        seed=42,
        balance_classes=True,
        stopping_metric="logloss",  # Multi-class metric
        sort_metric="logloss",
        nfolds=0,  # No cross-validation for time series
        verbosity="info"
    )
    
    print(f"🚀 Starting H2O AutoML Multi-Class training (max {max_runtime_secs}s, max {max_models} models)...")
    start_time = datetime.now()
    
    # Train
    aml.train(
        x=features_clean,
        y=target,
        training_frame=train,
        validation_frame=valid
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Get best model
    best_model = aml.leader
    leaderboard = aml.leaderboard.as_data_frame()
    
    # Evaluate on validation set
    perf = best_model.model_performance(valid)
    
    # Get predictions for detailed metrics
    predictions = best_model.predict(valid)
    pred_probs = predictions.as_data_frame()
    y_true = valid[target].as_data_frame().astype(int).values.flatten()
    
    # Calculate metrics
    logloss = perf.logloss()
    
    # Get class probabilities and predicted classes
    p0 = pred_probs['p0'].values
    p1 = pred_probs['p1'].values
    p2 = pred_probs['p2'].values
    y_pred = np.argmax([p0, p1, p2], axis=0)
    
    # Calculate accuracy manually (safer than using H2O's accuracy method)
    accuracy = np.mean(y_true == y_pred)
    
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        'logloss': round(logloss, 4),
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'training_time_sec': round(training_time, 2),
        'total_records': len(data),
        'train_samples': train.nrows,
        'test_samples': valid.nrows,
        'features_used': len(features_clean),
        'best_model': best_model.model_id
    }
    
    print(f"✅ H2O AutoML Multi-Class training complete!")
    print(f"   Best Model: {best_model.model_id}")
    print(f"   LogLoss: {logloss:.4f}")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"   Training time: {training_time:.2f} seconds")
    
    return {
        'model': best_model,
        'feature_engineer': feature_engineer,
        'metrics': metrics,
        'leaderboard': leaderboard,
        'input_dim': len(features_clean),
        'task_type': 'multiclass'
    }


def predict_h2o_multiclass(model_dict, data, lookback=30):
    """
    Predict multi-class probabilities using H2O AutoML model.
    
    Returns:
        dict with p0, p1, p2 probabilities
    """
    if model_dict is None or 'model' not in model_dict:
        return None
    
    model = model_dict['model']
    feature_engineer = model_dict['feature_engineer']
    
    # Create features for latest data
    df_features = feature_engineer.create_features(data)
    
    # Scale features
    df_scaled = feature_engineer.scale_features(df_features, fit=False)
    
    # Get the latest row for prediction
    if len(df_scaled) == 0:
        return None
    
    latest_features = df_scaled.iloc[-1:][feature_engineer.feature_columns]
    
    # Convert to H2O Frame
    h2o_features = h2o.H2OFrame(latest_features)
    
    # Make prediction (multi-class returns probabilities for each class)
    prediction = model.predict(h2o_features)
    pred_df = prediction.as_data_frame()
    
    return {
        'p0': float(pred_df['p0'].iloc[0]),  # <1.5x
        'p1': float(pred_df['p1'].iloc[0]),  # 1.5x-3x
        'p2': float(pred_df['p2'].iloc[0])    # >3x
    }


# ============================================================================
# LLM REASONING LAYER
# ============================================================================

def compute_context(recent_multipliers):
    """Compute context features for LLM: hit rate and volatility"""
    if len(recent_multipliers) == 0:
        return 0.0, "LOW"
    
    arr = np.array(recent_multipliers)
    hit_rate = np.mean(arr > 1.5)
    
    std = np.std(arr)
    if std < 0.7:
        volatility = "LOW"
    elif std < 1.5:
        volatility = "MEDIUM"
    else:
        volatility = "HIGH"
    
    return hit_rate, volatility


def expected_value(p1, p2, cashout_multiplier=1.5):
    """
    Calculate expected value for betting.
    
    Args:
        p1: Probability of 1.5x-3x
        p2: Probability of >3x
        cashout_multiplier: Target multiplier (e.g., 1.5x)
    
    Returns:
        Expected value
    """
    payout = cashout_multiplier - 1  # profit per $1
    loss = 1.0
    
    p_win = p1 + p2  # Probability of winning (>= 1.5x)
    ev = p_win * payout - (1 - p_win) * loss
    return ev


def llm_decision_engine(h2o_probs, hit_rate, volatility, drawdown, api_key=None):
    """
    LLM reasoning layer for betting decisions.
    
    Args:
        h2o_probs: dict with p0, p1, p2 probabilities
        hit_rate: Recent hit rate (>1.5x)
        volatility: "LOW", "MEDIUM", or "HIGH"
        drawdown: Current drawdown percentage
        api_key: OpenAI API key (optional, can use env var)
    
    Returns:
        dict with action, confidence, reason
    """
    if not LLM_AVAILABLE:
        # Fallback to rule-based decision if LLM not available
        p0, p1, p2 = h2o_probs.get('p0', 0), h2o_probs.get('p1', 0), h2o_probs.get('p2', 0)
        ev = expected_value(p1, p2)
        
        if ev <= 0:
            return {
                "action": "NO_BET",
                "confidence": 0.9,
                "reason": "Negative expected value"
            }
        elif ev > 0.1 and p2 > 0.2:
            return {
                "action": "BET",
                "confidence": min(0.9, ev * 2),
                "reason": f"Positive EV: {ev:.3f}, High probability of >3x: {p2:.2%}"
            }
        elif ev > 0.05:
            return {
                "action": "SMALL_BET",
                "confidence": min(0.7, ev * 3),
                "reason": f"Moderate positive EV: {ev:.3f}"
            }
        else:
            return {
                "action": "NO_BET",
                "confidence": 0.8,
                "reason": f"Low EV: {ev:.3f}"
            }
    
    try:
        # Initialize LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=api_key or os.getenv('OPENAI_API_KEY')
        )
        
        # Create prompt
        decision_prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are a professional risk manager for probabilistic betting systems.
You DO NOT predict outcomes.
You ONLY decide whether risk is justified given probabilities and recent behavior.
You are conservative after losses and aggressive only when edge is clear."""),
            
            ("human",
             """
H2O model probabilities for NEXT round:
- P(<1.5x): {p0}
- P(1.5x–3x): {p1}
- P(>3x): {p2}

Recent performance summary:
- Last 20 rounds hit-rate (>1.5x): {hit_rate}
- Recent volatility (LOW / MEDIUM / HIGH): {volatility}
- Current drawdown (%): {drawdown}

Rules:
1. Only recommend BET if expected value is positive
2. Avoid betting during HIGH volatility unless p2 is strong
3. Be conservative after drawdowns
4. NEVER force a bet

Respond ONLY in JSON:
{{
  "action": "NO_BET | SMALL_BET | BET",
  "confidence": float,
  "reason": string
}}
"""
            )
        ])
        
        p0, p1, p2 = h2o_probs.get('p0', 0), h2o_probs.get('p1', 0), h2o_probs.get('p2', 0)
        
        messages = decision_prompt.format_messages(
            p0=round(p0, 3),
            p1=round(p1, 3),
            p2=round(p2, 3),
            hit_rate=round(hit_rate, 2),
            volatility=volatility,
            drawdown=round(drawdown, 2)
        )
        
        response = llm.invoke(messages)
        result = json.loads(response.content)
        
        # Validate response
        if result.get('action') not in ['NO_BET', 'SMALL_BET', 'BET']:
            result['action'] = 'NO_BET'
        
        return result
        
    except Exception as e:
        print(f"⚠️ LLM decision error: {e}, using fallback")
        # Fallback to rule-based
        p0, p1, p2 = h2o_probs.get('p0', 0), h2o_probs.get('p1', 0), h2o_probs.get('p2', 0)
        ev = expected_value(p1, p2)
        
        if ev <= 0:
            return {
                "action": "NO_BET",
                "confidence": 0.9,
                "reason": "Negative expected value (fallback)"
            }
        elif ev > 0.1:
            return {
                "action": "BET",
                "confidence": min(0.9, ev * 2),
                "reason": f"Positive EV: {ev:.3f} (fallback)"
            }
        else:
            return {
                "action": "NO_BET",
                "confidence": 0.8,
                "reason": f"Low EV: {ev:.3f} (fallback)"
            }


# ============================================================================
# BAYESIAN SEQUENCE-BASED PREDICTION MODEL
# ============================================================================

def bayesian_seq_predict(
    recent_multipliers,
    threshold=2.0,
    window=10,
    alpha_prior=1.0,
    beta_prior=1.0,
    decay=0.95
):
    """
    Bayesian sequence-based probability estimator.
    
    Models P(next multiplier > threshold | recent sequence) using Beta-Bernoulli updating.
    
    Args:
        recent_multipliers: list or np.array of recent multipliers
        threshold: success threshold (e.g. 2.0x)
        window: how many recent samples to consider
        alpha_prior, beta_prior: Beta prior parameters (Beta(1,1) = uniform)
        decay: exponential decay for older observations (0.95 = recent matters more)
    
    Returns:
        dict with posterior probability, confidence, and parameters
    """
    if len(recent_multipliers) == 0:
        return {
            "probability": alpha_prior / (alpha_prior + beta_prior),
            "confidence": 0.0,
            "alpha": alpha_prior,
            "beta": beta_prior
        }
    
    # Use only recent window
    seq = np.array(recent_multipliers[-window:])
    
    # Convert to Bernoulli outcomes (1 if > threshold, 0 otherwise)
    outcomes = (seq > threshold).astype(int)
    
    # Apply decay weights (recent observations matter more)
    weights = decay ** np.arange(len(outcomes))[::-1]
    
    # Weighted counts
    weighted_success = np.sum(weights * outcomes)
    weighted_fail = np.sum(weights * (1 - outcomes))
    
    # Posterior parameters (Beta distribution)
    alpha_post = alpha_prior + weighted_success
    beta_post = beta_prior + weighted_fail
    
    # Posterior mean (expected probability)
    prob = alpha_post / (alpha_post + beta_post)
    
    # Confidence (inverse variance proxy - more observations = higher confidence)
    confidence = 1.0 - np.exp(-(alpha_post + beta_post) / window)
    
    return {
        "probability": round(float(prob), 4),
        "confidence": round(float(confidence), 4),
        "alpha": round(float(alpha_post), 3),
        "beta": round(float(beta_post), 3),
        "threshold": threshold,
        "window": len(seq)
    }


def bayesian_action(prob, confidence, prob_threshold_high=0.6, prob_threshold_low=0.55, min_confidence=0.5):
    """
    Convert Bayesian probability and confidence into betting action.
    
    Args:
        prob: Posterior probability from Bayesian model
        confidence: Confidence level (0-1)
        prob_threshold_high: Probability threshold for BET
        prob_threshold_low: Probability threshold for SMALL_BET
        min_confidence: Minimum confidence required to bet
    
    Returns:
        dict with action, risk_level, and reasoning
    """
    if confidence < min_confidence:
        return {
            "action": "NO_BET",
            "risk_level": "HIGH",
            "reason": f"Low confidence: {confidence:.2f} < {min_confidence}"
        }
    
    if prob > prob_threshold_high:
        return {
            "action": "BET",
            "risk_level": "LOW",
            "reason": f"High probability: {prob:.2%} with confidence {confidence:.2%}"
        }
    elif prob > prob_threshold_low:
        return {
            "action": "SMALL_BET",
            "risk_level": "MEDIUM",
            "reason": f"Moderate probability: {prob:.2%} with confidence {confidence:.2%}"
        }
    else:
        return {
            "action": "NO_BET",
            "risk_level": "HIGH",
            "reason": f"Low probability: {prob:.2%}"
        }


def multi_threshold_bayes(recent_multipliers, thresholds=[1.5, 2.0, 3.0, 5.0, 8.0], window=20):
    """
    Multi-threshold Bayesian prediction for different multiplier thresholds.
    
    Args:
        recent_multipliers: list of recent multipliers
        thresholds: list of thresholds to predict
        window: window size for each prediction
    
    Returns:
        dict with predictions for each threshold
    """
    results = {}
    for threshold in thresholds:
        results[f"gt_{threshold}"] = bayesian_seq_predict(
            recent_multipliers,
            threshold=threshold,
            window=window
        )
    return results


def predict_bayesian(data, threshold=2.0, window=20):
    """
    Make Bayesian prediction from full data DataFrame.
    
    Args:
        data: DataFrame with 'multiplier' column
        threshold: Success threshold
        window: Recent window size
    
    Returns:
        dict with prediction, action, and metadata
    """
    if data is None or len(data) < window:
        return None
    
    # Get recent multipliers
    recent_multipliers = data['multiplier'].tail(window).tolist()
    
    # Get Bayesian prediction
    bayes_result = bayesian_seq_predict(
        recent_multipliers=recent_multipliers,
        threshold=threshold,
        window=window,
        alpha_prior=1.5,  # Slightly optimistic prior
        beta_prior=1.5,
        decay=0.95
    )
    
    # Get betting action
    action_result = bayesian_action(
        prob=bayes_result['probability'],
        confidence=bayes_result['confidence'],
        prob_threshold_high=0.6,
        prob_threshold_low=0.55,
        min_confidence=0.5
    )
    
    # Get multi-threshold predictions for context
    multi_thresh = multi_threshold_bayes(recent_multipliers, thresholds=[1.5, 2.0, 3.0, 5.0, 8.0])
    
    return {
        'probability_gt_threshold': bayes_result['probability'],
        'confidence': bayes_result['confidence'] * 100,  # Convert to percentage
        'betting_action': action_result['action'],
        'risk_level': action_result['risk_level'],
        'reason': action_result['reason'],
        'bayesian_params': {
            'alpha': bayes_result['alpha'],
            'beta': bayes_result['beta'],
            'threshold': threshold
        },
        'multi_threshold': multi_thresh,
        'recent_sequence': recent_multipliers[-10:],  # Last 10 for display
        'timestamp': datetime.now().isoformat(),
        'last_actual': float(data['multiplier'].iloc[-1]) if len(data) > 0 else None,
        'model_type': 'bayesian_sequence'
    }
