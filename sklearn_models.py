"""
Sklearn Ensemble Models for Multiplier Prediction
Extracted from predictor_app.py for modular architecture
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost and LightGBM
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception):
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, Exception):
    LIGHTGBM_AVAILABLE = False


def handle_outliers(df, multiplier_col='multiplier', method='winsorize', percentile_low=1, percentile_high=99):
    """Handle outliers in multiplier data."""
    df = df.copy()
    multipliers = df[multiplier_col].values
    
    if method == 'winsorize':
        low_threshold = np.percentile(multipliers, percentile_low)
        high_threshold = np.percentile(multipliers, percentile_high)
        df[multiplier_col] = df[multiplier_col].clip(lower=low_threshold, upper=high_threshold)
    elif method == 'log_transform':
        df[multiplier_col] = np.log1p(df[multiplier_col])
    
    return df


def prepare_features_sklearn(df, lookback=30, use_all_data=True):
    """
    Prepare advanced features for sklearn models.
    Same as prepare_features from predictor_app.py
    """
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
    
    # Extract time components
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['second'] = df['timestamp'].dt.second
    df['time_of_day'] = df['hour'] * 3600 + df['minute'] * 60 + df['second']
    
    # Global statistics from entire dataset
    global_mean = df['multiplier'].mean()
    global_std = df['multiplier'].std()
    global_median = df['multiplier'].median()
    global_min = df['multiplier'].min()
    global_max = df['multiplier'].max()
    global_p25 = df['multiplier'].quantile(0.25)
    global_p75 = df['multiplier'].quantile(0.75)
    global_p90 = df['multiplier'].quantile(0.90)
    global_p95 = df['multiplier'].quantile(0.95)
    global_skew = df['multiplier'].skew()
    
    # Pattern analysis
    high_count_global = (df['multiplier'] > 5.0).sum()
    very_high_count_global = (df['multiplier'] > 10.0).sum()
    low_count_global = (df['multiplier'] < 1.5).sum()
    extreme_count_global = (df['multiplier'] > 50.0).sum()
    
    # Long-term trend
    if len(df) > 50:
        x_indices = np.arange(len(df))
        global_trend_coef = np.polyfit(x_indices, df['multiplier'].values, 2)
        global_trend_slope = global_trend_coef[1]
        global_trend_curve = global_trend_coef[0]
    else:
        global_trend_slope = 0
        global_trend_curve = 0
    
    # Autocorrelation
    if len(df) > 100:
        autocorr_lag5 = df['multiplier'].autocorr(lag=5) if len(df) > 5 else 0
        autocorr_lag10 = df['multiplier'].autocorr(lag=10) if len(df) > 10 else 0
        autocorr_lag20 = df['multiplier'].autocorr(lag=20) if len(df) > 20 else 0
    else:
        autocorr_lag5 = 0
        autocorr_lag10 = 0
        autocorr_lag20 = 0
    
    # Transition rate
    multiplier_values = df['multiplier'].values
    if len(multiplier_values) > 50:
        high_low_transitions = sum(1 for i in range(1, len(multiplier_values)) 
                                  if (multiplier_values[i-1] < 2.0 and multiplier_values[i] > 5.0) or
                                     (multiplier_values[i-1] > 5.0 and multiplier_values[i] < 2.0))
        transition_rate = high_low_transitions / len(multiplier_values)
    else:
        transition_rate = 0
    
    features = []
    targets = []
    
    # Use ALL data for training
    start_idx = lookback if use_all_data else max(lookback, len(df) - 100)
    
    # Pre-compute rolling statistics
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
        
        # Time features
        time_diff = (df['timestamp'].iloc[i] - df['timestamp'].iloc[i-1]).total_seconds() if i > 0 else 0
        hour = df['hour'].iloc[i]
        minute = df['minute'].iloc[i]
        time_of_day = df['time_of_day'].iloc[i]
        
        # Time since last high
        recent_highs = [j for j in range(max(0, i-10), i) if df['multiplier'].iloc[j] > 5.0]
        time_since_high = (i - recent_highs[-1]) if recent_highs else 20
        
        # Rolling statistics
        rolling_mean_3 = df['mult_rolling_mean_3'].iloc[i-1] if i > 0 else mean_val
        rolling_mean_5 = df['mult_rolling_mean_5'].iloc[i-1] if i > 0 else mean_val
        rolling_mean_10 = df['mult_rolling_mean_10'].iloc[i-1] if i > 0 else mean_val
        rolling_mean_20 = df['mult_rolling_mean_20'].iloc[i-1] if i > 0 else mean_val
        rolling_std_5 = df['mult_rolling_std_5'].iloc[i-1] if i > 0 else std_val
        rolling_std_10 = df['mult_rolling_std_10'].iloc[i-1] if i > 0 else std_val
        rolling_std_20 = df['mult_rolling_std_20'].iloc[i-1] if i > 0 else std_val
        rolling_min_5 = df['mult_rolling_min_5'].iloc[i-1] if i > 0 else min_val
        rolling_max_5 = df['mult_rolling_max_5'].iloc[i-1] if i > 0 else max_val
        
        # EMA
        ema_3 = df['mult_ema_3'].iloc[i-1] if i > 0 else mean_val
        ema_5 = df['mult_ema_5'].iloc[i-1] if i > 0 else mean_val
        ema_10 = df['mult_ema_10'].iloc[i-1] if i > 0 else mean_val
        ema_20 = df['mult_ema_20'].iloc[i-1] if i > 0 else mean_val
        
        # Sequence features
        last_1 = window[-1] if len(window) > 0 else mean_val
        last_2 = window[-2] if len(window) > 1 else last_1
        last_3 = window[-3] if len(window) > 2 else last_2
        last_5_mean = np.mean(window[-5:]) if len(window) >= 5 else mean_val
        last_10_mean = np.mean(window[-10:]) if len(window) >= 10 else mean_val
        
        # Comparison to global
        window_vs_global_mean = mean_val / global_mean if global_mean > 0 else 1
        window_vs_global_std = std_val / global_std if global_std > 0 else 1
        last_vs_global_mean = last_1 / global_mean if global_mean > 0 else 1
        last_vs_global_median = last_1 / global_median if global_median > 0 else 1
        percentile_in_global = (df['multiplier'].iloc[:i+1] < last_1).sum() / (i + 1) if i > 0 else 0.5
        dist_from_global_mean = last_1 - global_mean
        dist_from_global_median = last_1 - global_median
        z_score_global = (last_1 - global_mean) / global_std if global_std > 0 else 0
        
        # Trends
        recent_5 = window[-5:] if len(window) >= 5 else window
        recent_10 = window[-10:] if len(window) >= 10 else window
        trend_5 = np.polyfit(range(len(recent_5)), recent_5, 1)[0] if len(recent_5) > 1 else 0
        trend_10 = np.polyfit(range(len(recent_10)), recent_10, 1)[0] if len(recent_10) > 1 else 0
        
        # Volatility
        volatility_5 = np.std(window[-5:]) if len(window) >= 5 else std_val
        volatility_10 = np.std(window[-10:]) if len(window) >= 10 else std_val
        
        # Patterns
        high_ratio = len([x for x in window if x > 2.0]) / len(window)
        very_high_ratio = len([x for x in window if x > 10.0]) / len(window)
        low_ratio = len([x for x in window if x < 1.5]) / len(window)
        extreme_high_ratio = len([x for x in window if x > 50.0]) / len(window)
        
        # Momentum
        momentum_1 = last_1 - last_2 if len(window) > 1 else 0
        momentum_2 = last_2 - last_3 if len(window) > 2 else 0
        momentum_3 = last_3 - (window[-4] if len(window) > 3 else last_3)
        acceleration = momentum_1 - momentum_2
        
        # Rate of change
        roc_1 = (last_1 - last_2) / last_2 if len(window) > 1 and last_2 > 0 else 0
        roc_2 = (last_2 - last_3) / last_3 if len(window) > 2 and last_3 > 0 else 0
        roc_5 = (last_1 - window[-6]) / window[-6] if len(window) > 5 and window[-6] > 0 else 0
        
        # Autocorrelation
        if len(window) >= 5:
            autocorr_1 = np.corrcoef(window[:-1], window[1:])[0, 1] if len(window) > 1 and not np.isnan(np.corrcoef(window[:-1], window[1:])[0, 1]) else 0
        else:
            autocorr_1 = 0
        
        # Range features
        range_val = max_val - min_val
        range_ratio = range_val / mean_val if mean_val > 0 else 0
        cv = std_val / mean_val if mean_val > 0 else 0
        
        # Percentiles
        p10 = np.percentile(window, 10)
        p25 = np.percentile(window, 25)
        p50 = np.percentile(window, 50)
        p75 = np.percentile(window, 75)
        p90 = np.percentile(window, 90)
        iqr = p75 - p25
        
        # MA ratios
        ma_ratio_3_5 = rolling_mean_3 / rolling_mean_5 if rolling_mean_5 > 0 else 1
        ma_ratio_5_10 = rolling_mean_5 / rolling_mean_10 if rolling_mean_10 > 0 else 1
        ema_ratio_3_5 = ema_3 / ema_5 if ema_5 > 0 else 1
        
        # Distance from MA
        dist_from_ma3 = last_1 - rolling_mean_3
        dist_from_ma5 = last_1 - rolling_mean_5
        dist_from_ma10 = last_1 - rolling_mean_10
        dist_from_ema3 = last_1 - ema_3
        
        # Z-scores
        z_score_5 = (last_1 - rolling_mean_5) / rolling_std_5 if rolling_std_5 > 0 else 0
        z_score_10 = (last_1 - rolling_mean_10) / rolling_std_10 if rolling_std_10 > 0 else 0
        
        # Feature vector (70+ features)
        feature_row = [
            # Global features (20)
            global_mean, global_std, global_median, global_min, global_max,
            global_p25, global_p75, global_p90, global_p95, global_skew,
            high_count_global / len(df), very_high_count_global / len(df),
            low_count_global / len(df), extreme_count_global / len(df),
            global_trend_slope, global_trend_curve,
            autocorr_lag5, autocorr_lag10, autocorr_lag20, transition_rate,
            # Local features (50+)
            mean_val, std_val, min_val, max_val, median_val,
            last_1, last_2, last_3, last_5_mean, last_10_mean,
            rolling_mean_3, rolling_mean_5, rolling_mean_10, rolling_mean_20,
            rolling_std_5, rolling_std_10, rolling_std_20,
            rolling_min_5, rolling_max_5,
            ema_3, ema_5, ema_10, ema_20,
            trend_5, trend_10,
            volatility_5, volatility_10,
            high_ratio, very_high_ratio, low_ratio, extreme_high_ratio,
            momentum_1, momentum_2, momentum_3, acceleration,
            roc_1, roc_2, roc_5,
            range_val, range_ratio, cv,
            p10, p25, p50, p75, p90, iqr,
            ma_ratio_3_5, ma_ratio_5_10, ema_ratio_3_5,
            dist_from_ma3, dist_from_ma5, dist_from_ma10, dist_from_ema3,
            z_score_5, z_score_10,
            window_vs_global_mean, window_vs_global_std,
            last_vs_global_mean, last_vs_global_median,
            percentile_in_global,
            dist_from_global_mean, dist_from_global_median, z_score_global,
            time_diff, hour, minute, time_of_day, time_since_high,
            autocorr_1,
        ]
        
        features.append(feature_row)
        targets.append(current)
    
    return np.array(features), np.array(targets)


def train_sklearn_ensemble(data, config, progress_callback=None):
    """
    Train sklearn ensemble model.
    
    Args:
        data: DataFrame with 'timestamp' and 'multiplier' columns
        config: Config object with training parameters
        progress_callback: Optional callback function
    
    Returns:
        dict with model, scaler, feature_selector, metrics, etc.
    """
    print(f"🔄 Preparing features for sklearn ensemble...")
    
    features, targets = prepare_features_sklearn(data, lookback=config.LOOKBACK, use_all_data=True)
    
    if features is None or len(features) < config.MIN_SAMPLES_FOR_TRAINING:
        return None
    
    print(f"📊 Features shape: {features.shape}, Targets shape: {targets.shape}")
    
    # Winsorize targets
    targets_array = np.array(targets)
    target_q99 = np.percentile(targets_array, 99)
    target_q01 = np.percentile(targets_array, 1)
    targets_capped = np.clip(targets_array, target_q01, target_q99)
    
    print(f"📊 Target range: [{np.min(targets_capped):.2f}, {np.max(targets_capped):.2f}] (winsorized)")
    
    # Scale features
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Feature selection
    feature_selector = None
    if len(features_scaled) > 50 and features_scaled.shape[1] > 30:
        try:
            k_features = min(70, features_scaled.shape[1])
            mi_scores = mutual_info_regression(features_scaled, targets_capped, random_state=42, n_neighbors=3)
            top_indices = np.argsort(mi_scores)[-k_features:]
            
            class CustomFeatureSelector:
                def __init__(self, indices):
                    self.indices = indices
                def transform(self, X):
                    return X[:, self.indices]
            
            feature_selector = CustomFeatureSelector(top_indices)
            features_scaled = features_scaled[:, top_indices]
            print(f"✅ Selected {k_features} best features using Mutual Information")
        except Exception as e:
            k_features = min(70, features_scaled.shape[1])
            feature_selector = SelectKBest(f_regression, k=k_features)
            features_scaled = feature_selector.fit_transform(features_scaled, targets_capped)
            print(f"✅ Selected {features_scaled.shape[1]} features using F-regression (fallback)")
    else:
        print(f"📊 Using all {features_scaled.shape[1]} features")
    
    # Split data
    if len(features) > 50:
        split_idx = int(len(features) * 0.8)
        X_train, X_test = features_scaled[:split_idx], features_scaled[split_idx:]
        y_train, y_test = targets_capped[:split_idx], targets_capped[split_idx:]
    else:
        X_train, X_test = features_scaled, features_scaled[-5:] if len(features) > 5 else features_scaled
        y_train, y_test = targets_capped, targets_capped[-5:] if len(targets) > 5 else targets_capped
    
    # Create ensemble models
    models = []
    weights = []
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        try:
            if len(X_train) > 100:
                X_train_fit, X_val = X_train[:-50], X_train[-50:]
                y_train_fit, y_val = y_train[:-50], y_train[-50:]
            else:
                X_train_fit, X_val = X_train, X_train
                y_train_fit, y_val = y_train, y_train
            
            # Try new API first (XGBoost 2.0+)
            try:
                xgb_model = xgb.XGBRegressor(
                    n_estimators=1000,
                    max_depth=10,
                    learning_rate=0.01,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    colsample_bylevel=0.85,
                    min_child_weight=5,
                    gamma=0.2,
                    reg_alpha=0.2,
                    reg_lambda=2.0,
                    random_state=42,
                    n_jobs=-1,
                    tree_method='hist',
                    objective='reg:squarederror',
                    eval_metric='mae',
                    early_stopping_rounds=50  # New API: pass to constructor
                )
                xgb_model.fit(
                    X_train_fit, y_train_fit,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            except TypeError:
                # Fallback to old API (XGBoost < 2.0)
                xgb_model = xgb.XGBRegressor(
                    n_estimators=1000,
                    max_depth=10,
                    learning_rate=0.01,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    colsample_bylevel=0.85,
                    min_child_weight=5,
                    gamma=0.2,
                    reg_alpha=0.2,
                    reg_lambda=2.0,
                    random_state=42,
                    n_jobs=-1,
                    tree_method='hist',
                    objective='reg:squarederror',
                    eval_metric='mae'
                )
                xgb_model.fit(
                    X_train_fit, y_train_fit,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,  # Old API: pass to fit()
                    verbose=False
                )
            models.append(('xgb', xgb_model))
            weights.append(4)
            print("✅ XGBoost added to ensemble")
        except Exception as e:
            print(f"⚠️ XGBoost failed: {str(e)[:100]}")
    
    # LightGBM
    if LIGHTGBM_AVAILABLE:
        try:
            if len(X_train) > 100:
                X_train_fit, X_val = X_train[:-50], X_train[-50:]
                y_train_fit, y_val = y_train[:-50], y_train[-50:]
            else:
                X_train_fit, X_val = X_train, X_train
                y_train_fit, y_val = y_train, y_train
            
            lgb_model = lgb.LGBMRegressor(
                n_estimators=1000,
                max_depth=10,
                learning_rate=0.01,
                num_leaves=50,
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
            weights.append(4)
            print("✅ LightGBM added to ensemble")
        except Exception as e:
            print(f"⚠️ LightGBM failed: {str(e)[:100]}")
    
    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=800,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        oob_score=True,
        bootstrap=True
    )
    models.append(('rf', rf_model))
    weights.append(3 if len(models) == 1 else 2)
    
    # Gradient Boosting
    gb_model = GradientBoostingRegressor(
        n_estimators=600,
        max_depth=10,
        learning_rate=0.02,
        min_samples_split=4,
        min_samples_leaf=1,
        subsample=0.9,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=30,
        tol=1e-4
    )
    models.append(('gb', gb_model))
    weights.append(3 if len(models) == 2 else 2)
    
    # Extra Trees
    et_model = ExtraTreesRegressor(
        n_estimators=600,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        bootstrap=True
    )
    models.append(('et', et_model))
    weights.append(2)
    
    # Regularized models
    ridge_model = Ridge(alpha=0.3, random_state=42)
    models.append(('ridge', ridge_model))
    weights.append(1)
    
    elastic_model = ElasticNet(alpha=0.3, l1_ratio=0.5, random_state=42, max_iter=2000)
    models.append(('elastic', elastic_model))
    weights.append(1)
    
    # Create ensemble
    ensemble_model = VotingRegressor(models, weights=weights)
    
    model_names = [name for name, _ in models]
    print(f"🔄 Training sklearn ensemble: {', '.join(model_names)}...")
    print(f"   Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Train ensemble
    ensemble_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = ensemble_model.predict(X_test)
    y_pred = np.maximum(1.0, y_pred)
    y_test_clipped = np.maximum(1.0, y_test)
    
    mae = mean_absolute_error(y_test_clipped, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_clipped, y_pred))
    r2 = r2_score(y_test_clipped, y_pred)
    
    # Cross-validation
    if len(X_train) > 50:
        tscv = TimeSeriesSplit(n_splits=min(5, len(X_train)//20))
        cv_scores = cross_val_score(ensemble_model, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()
    else:
        cv_mae = mae
        cv_std = 0
    
    metrics = {
        'mae': round(mae, 3),
        'rmse': round(rmse, 3),
        'r2_score': round(r2, 3),
        'cv_mae': round(cv_mae, 3),
        'cv_std': round(cv_std, 3),
        'samples_trained': len(X_train),
        'samples_tested': len(X_test),
        'total_records': len(data)
    }
    
    print(f"📈 Sklearn Ensemble Performance:")
    print(f"   MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
    print(f"   CV MAE: {cv_mae:.3f} (+/- {cv_std:.3f})")
    
    return {
        'model': ensemble_model,
        'scaler': scaler,
        'feature_selector': feature_selector,
        'metrics': metrics,
        'input_dim': features_scaled.shape[1]
    }


def predict_sklearn_ensemble(model_dict, data, lookback=30):
    """
    Predict using sklearn ensemble model.
    
    Args:
        model_dict: Dict with 'model', 'scaler', 'feature_selector'
        data: DataFrame with 'timestamp' and 'multiplier' columns
        lookback: Lookback window size
    
    Returns:
        float: Predicted multiplier value
    """
    if model_dict is None or 'model' not in model_dict:
        return None
    
    model = model_dict['model']
    scaler = model_dict['scaler']
    feature_selector = model_dict.get('feature_selector')
    
    # Prepare features
    features, _ = prepare_features_sklearn(data, lookback=lookback, use_all_data=True)
    
    if features is None or len(features) == 0:
        return None
    
    # Use last window
    last_features = features[-1].reshape(1, -1)
    
    # Scale
    last_features_scaled = scaler.transform(last_features)
    
    # Feature selection
    if feature_selector is not None:
        if hasattr(feature_selector, 'indices'):
            last_features_scaled = last_features_scaled[:, feature_selector.indices]
        else:
            last_features_scaled = feature_selector.transform(last_features_scaled)
    
    # Predict
    prediction = model.predict(last_features_scaled)[0]
    prediction = max(1.0, prediction)
    
    return float(prediction)

