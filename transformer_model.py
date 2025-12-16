"""
Transformer Model Module - Modular training and evaluation
Supports both regression and binary classification tasks
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class TransformerPredictor(nn.Module):
    """Transformer-based model for time series prediction."""
    
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1, task_type='regression'):
        super(TransformerPredictor, self).__init__()
        
        self.task_type = task_type  # 'regression' or 'classification'
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim_feedforward, dim_feedforward // 2)
        
        # Output layer depends on task type
        if task_type == 'classification':
            self.fc3 = nn.Linear(dim_feedforward // 2, 1)  # Binary classification
            self.sigmoid = nn.Sigmoid()
        else:
            self.fc3 = nn.Linear(dim_feedforward // 2, 1)  # Regression
        
        self.activation = nn.ReLU()
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.input_projection(x)
        pos_enc = self.pos_encoder[:, :seq_len, :]
        x = x + pos_enc
        x = self.transformer(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        if self.task_type == 'classification':
            x = self.sigmoid(x)
        
        return x.squeeze(-1)


class MultiplierDataset(Dataset):
    """Dataset for multiplier prediction."""
    
    def __init__(self, features, targets, lookback=30):
        self.features = features
        self.targets = targets
        self.lookback = lookback
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        start_idx = max(0, idx - self.lookback + 1)
        end_idx = idx + 1
        feature_seq = self.features[start_idx:end_idx]
        
        if len(feature_seq) < self.lookback:
            padding = np.zeros((self.lookback - len(feature_seq), feature_seq.shape[1]))
            feature_seq = np.vstack([padding, feature_seq])
        
        target = self.targets[idx]
        return torch.FloatTensor(feature_seq), torch.FloatTensor([target])


def handle_outliers(df, multiplier_col='multiplier', percentile_low=1, percentile_high=99):
    """Handle outliers using winsorization."""
    df = df.copy()
    multipliers = df[multiplier_col].values
    low_threshold = np.percentile(multipliers, percentile_low)
    high_threshold = np.percentile(multipliers, percentile_high)
    df[multiplier_col] = df[multiplier_col].clip(lower=low_threshold, upper=high_threshold)
    return df


def prepare_features_transformer(df, lookback=30):
    """Prepare features for transformer model - based on predictor_app.py."""
    if len(df) < lookback + 1:
        return None, None
    
    df = df.copy()
    df['multiplier'] = pd.to_numeric(df['multiplier'], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = handle_outliers(df, percentile_low=1, percentile_high=99)
    
    if len(df) < lookback + 1:
        return None, None
    
    # Extract time components
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['second'] = df['timestamp'].dt.second
    df['time_of_day'] = df['hour'] * 3600 + df['minute'] * 60 + df['second']
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Global statistics
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
        autocorr_lag5 = autocorr_lag10 = autocorr_lag20 = 0
    
    # Transition rate
    multiplier_values = df['multiplier'].values
    if len(multiplier_values) > 50:
        high_low_transitions = sum(1 for i in range(1, len(multiplier_values)) 
                                  if (multiplier_values[i-1] < 2.0 and multiplier_values[i] > 5.0) or
                                     (multiplier_values[i-1] > 5.0 and multiplier_values[i] < 2.0))
        transition_rate = high_low_transitions / len(multiplier_values)
    else:
        transition_rate = 0
    
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
    
    # Additional features
    df['mult_diff_1'] = df['multiplier'].diff(1)
    df['mult_diff_2'] = df['multiplier'].diff(2)
    df['mult_pct_change'] = df['multiplier'].pct_change()
    
    features = []
    targets = []
    start_idx = lookback
    
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
        day_of_week = df['day_of_week'].iloc[i]
        is_weekend = df['is_weekend'].iloc[i]
        
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
        
        # Global comparisons
        window_vs_global_mean = mean_val / global_mean if global_mean > 0 else 1
        last_vs_global_mean = last_1 / global_mean if global_mean > 0 else 1
        last_vs_global_median = last_1 / global_median if global_median > 0 else 1
        percentile_in_global = (df['multiplier'].iloc[:i+1] < last_1).sum() / (i + 1) if i > 0 else 0.5
        dist_from_global_mean = last_1 - global_mean
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
        
        # Momentum
        momentum_1 = last_1 - last_2 if len(window) > 1 else 0
        momentum_2 = last_2 - last_3 if len(window) > 2 else 0
        acceleration = momentum_1 - momentum_2
        
        # Rate of change
        roc_1 = (last_1 - last_2) / last_2 if len(window) > 1 and last_2 > 0 else 0
        roc_2 = (last_2 - last_3) / last_3 if len(window) > 2 and last_3 > 0 else 0
        
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
        
        # Z-scores
        z_score_5 = (last_1 - rolling_mean_5) / rolling_std_5 if rolling_std_5 > 0 else 0
        z_score_10 = (last_1 - rolling_mean_10) / rolling_std_10 if rolling_std_10 > 0 else 0
        
        # Additional features
        diff_1 = df['mult_diff_1'].iloc[i] if not pd.isna(df['mult_diff_1'].iloc[i]) else 0
        diff_2 = df['mult_diff_2'].iloc[i] if not pd.isna(df['mult_diff_2'].iloc[i]) else 0
        pct_change = df['mult_pct_change'].iloc[i] if not pd.isna(df['mult_pct_change'].iloc[i]) else 0
        
        # Autocorrelation
        autocorr_1 = np.corrcoef(window[:-1], window[1:])[0, 1] if len(window) > 1 and not np.isnan(np.corrcoef(window[:-1], window[1:])[0, 1]) else 0
        
        # Feature vector (80+ features)
        # NOTE: We reduce weight of last_1, last_2, last_3 to prevent naive baseline
        # Instead, we emphasize patterns, trends, and relative features
        feature_row = [
            global_mean, global_std, global_median, global_min, global_max,
            global_p25, global_p75, global_p90, global_p95, global_skew,
            high_count_global / len(df), very_high_count_global / len(df),
            low_count_global / len(df), extreme_count_global / len(df),
            global_trend_slope, global_trend_curve,
            autocorr_lag5, autocorr_lag10, autocorr_lag20, transition_rate,
            mean_val, std_val, min_val, max_val, median_val,
            # Reduced weight: multiply recent values by 0.3 to prevent naive baseline
            last_1 * 0.3, last_2 * 0.3, last_3 * 0.3, last_5_mean, last_10_mean,
            rolling_mean_3, rolling_mean_5, rolling_mean_10, rolling_mean_20,
            rolling_std_5, rolling_std_10, rolling_std_20,
            rolling_min_5, rolling_max_5,
            ema_3, ema_5, ema_10, ema_20,
            trend_5, trend_10,
            volatility_5, volatility_10,
            high_ratio, very_high_ratio, low_ratio,
            momentum_1, momentum_2, acceleration,
            roc_1, roc_2,
            range_val, range_ratio, cv,
            p10, p25, p50, p75, p90, iqr,
            ma_ratio_3_5, ma_ratio_5_10, ema_ratio_3_5,
            dist_from_ma3, dist_from_ma5, dist_from_ma10,
            z_score_5, z_score_10,
            window_vs_global_mean,
            last_vs_global_mean, last_vs_global_median,
            percentile_in_global,
            dist_from_global_mean, z_score_global,
            time_diff, hour, minute, time_of_day, day_of_week, is_weekend,
            diff_1, diff_2, pct_change,
            autocorr_1,
        ]
        
        features.append(feature_row)
        targets.append(current)
    
    return np.array(features), np.array(targets)


def train_transformer_model(data, config, task_type='regression', progress_callback=None, predict_delta=True):
    """
    Train transformer model for regression or classification.
    
    Args:
        data: DataFrame with 'timestamp' and 'multiplier' columns
        config: Config object with training parameters
        task_type: 'regression' or 'classification'
        progress_callback: Optional callback function(epoch, total, progress)
        predict_delta: If True, predict change from last value instead of absolute value (prevents naive baseline)
    
    Returns:
        dict with model, scalers, metrics, etc.
    """
    print(f"🔄 Preparing features for {task_type} task...")
    
    # For classification, convert targets to binary (>20 or not)
    if task_type == 'classification':
        data = data.copy()
        data['target'] = (data['multiplier'] > 20.0).astype(int)
        features, targets = prepare_features_transformer(data, lookback=config.LOOKBACK)
        if features is None:
            return None
        # Use binary targets
        targets = data['target'].iloc[config.LOOKBACK:].values
    else:
        features, targets = prepare_features_transformer(data, lookback=config.LOOKBACK)
    
    if features is None or len(features) < config.MIN_SAMPLES_FOR_TRAINING:
        return None
    
    print(f"📊 Features shape: {features.shape}, Targets shape: {targets.shape}")
    
    # For regression: Predict delta/change instead of absolute value to avoid naive baseline
    if task_type == 'regression' and predict_delta:
        # Get last values for each target
        last_values = data['multiplier'].iloc[config.LOOKBACK-1:len(data)-1].values
        if len(last_values) != len(targets):
            # Adjust if mismatch
            min_len = min(len(last_values), len(targets))
            last_values = last_values[:min_len]
            targets = targets[:min_len]
            features = features[:min_len]
        
        # Predict delta (change) instead of absolute value
        deltas = targets - last_values
        print(f"🔄 Using delta prediction (change from last value) to avoid naive baseline")
        print(f"   Delta stats: mean={np.mean(deltas):.4f}, std={np.std(deltas):.4f}, min={np.min(deltas):.4f}, max={np.max(deltas):.4f}")
        targets = deltas
        # Store last_values for reconstruction during prediction
        last_values_train = last_values
    else:
        last_values_train = None
    
    # Split data (80% train, 20% test)
    split_idx = int(len(features) * 0.8)
    train_features = features[:split_idx]
    train_targets = targets[:split_idx]
    test_features = features[split_idx:]
    test_targets = targets[split_idx:]
    
    if last_values_train is not None:
        train_last_values = last_values_train[:split_idx]
        test_last_values = last_values_train[split_idx:]
    else:
        train_last_values = None
        test_last_values = None
    
    # Scale features
    scaler = RobustScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    # Scale targets (for regression only)
    target_scaler = None
    if task_type == 'regression':
        target_scaler = RobustScaler()
        train_targets_scaled = target_scaler.fit_transform(train_targets.reshape(-1, 1)).flatten()
        test_targets_scaled = target_scaler.transform(test_targets.reshape(-1, 1)).flatten()
    else:
        # Classification targets are already 0/1
        train_targets_scaled = train_targets.astype(float)
        test_targets_scaled = test_targets.astype(float)
    
    # Create datasets
    train_dataset = MultiplierDataset(train_features_scaled, train_targets_scaled, lookback=config.LOOKBACK)
    test_dataset = MultiplierDataset(test_features_scaled, test_targets_scaled, lookback=config.LOOKBACK)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Create model
    input_dim = features.shape[1]
    model = TransformerPredictor(
        input_dim=input_dim,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        task_type=task_type
    ).to(config.DEVICE)
    
    # Training setup
    if task_type == 'classification':
        criterion = nn.BCELoss()
    else:
        # Use Huber loss for regression - less sensitive to outliers and encourages learning patterns
        criterion = nn.HuberLoss(delta=1.0)
        # Alternative: Use MSE with regularization term to penalize predictions too close to last value
        # We'll add this as a custom loss component
    
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.05)  # Lower LR, higher weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
    
    num_epochs = config.TRAINING_EPOCHS if hasattr(config, 'TRAINING_EPOCHS') else 50
    print(f"🚀 Training transformer model ({num_epochs} epochs) for {task_type}...")
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(config.DEVICE)
            batch_targets = batch_targets.to(config.DEVICE)
            
            optimizer.zero_grad()
            predictions = model(batch_features)
            
            # Base loss
            loss = criterion(predictions, batch_targets.squeeze())
            
            # Note: We're predicting deltas, so the model must learn patterns of change
            # rather than just copying the last value. The delta approach naturally prevents naive baseline.
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(config.DEVICE)
                batch_targets = batch_targets.to(config.DEVICE)
                predictions = model(batch_features)
                loss = criterion(predictions, batch_targets.squeeze())
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if progress_callback:
            progress = int((epoch + 1) / num_epochs * 100)
            progress_callback(epoch + 1, num_epochs, progress)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Evaluate on test set
    print("📊 Evaluating model...")
    model.eval()
    all_predictions = []
    all_targets = []
    batch_idx = 0
    
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_features = batch_features.to(config.DEVICE)
            predictions = model(batch_features)
            
            if task_type == 'regression':
                predictions_delta = target_scaler.inverse_transform(predictions.cpu().numpy().reshape(-1, 1)).flatten()
                targets_delta = target_scaler.inverse_transform(batch_targets.cpu().numpy().reshape(-1, 1)).flatten()
                
                # Reconstruct absolute predictions from deltas
                if predict_delta and test_last_values is not None:
                    # Get corresponding last values for this batch
                    batch_size = len(predictions_delta)
                    batch_start = batch_idx * config.BATCH_SIZE
                    batch_end = min(batch_start + batch_size, len(test_last_values))
                    batch_last_values = test_last_values[batch_start:batch_end]
                    
                    # Ensure same length
                    min_len = min(len(batch_last_values), len(predictions_delta), len(targets_delta))
                    batch_last_values = batch_last_values[:min_len]
                    predictions_delta = predictions_delta[:min_len]
                    targets_delta = targets_delta[:min_len]
                    
                    # Reconstruct: prediction = last_value + delta
                    predictions_original = batch_last_values + predictions_delta
                    targets_original = batch_last_values + targets_delta
                else:
                    predictions_original = predictions_delta
                    targets_original = targets_delta
            else:
                predictions_original = predictions.cpu().numpy()
                targets_original = batch_targets.cpu().numpy().flatten()
            
            all_predictions.extend(predictions_original)
            all_targets.extend(targets_original)
            batch_idx += 1
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Calculate metrics based on task type
    if task_type == 'classification':
        # Convert probabilities to binary predictions
        pred_binary = (all_predictions >= 0.5).astype(int)
        all_targets_binary = all_targets.astype(int)
        
        accuracy = accuracy_score(all_targets_binary, pred_binary)
        precision = precision_score(all_targets_binary, pred_binary, zero_division=0)
        recall = recall_score(all_targets_binary, pred_binary, zero_division=0)
        f1 = f1_score(all_targets_binary, pred_binary, zero_division=0)
        try:
            auc = roc_auc_score(all_targets_binary, all_predictions)
        except:
            auc = 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        print(f"✅ Training complete!")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        print(f"   AUC: {auc:.4f}")
    else:
        # Regression metrics
        all_predictions = np.maximum(1.0, all_predictions)
        all_targets = np.maximum(1.0, all_targets)
        
        mae = mean_absolute_error(all_targets, all_predictions)
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        r2 = r2_score(all_targets, all_predictions)
        mape = np.mean(np.abs((all_targets - all_predictions) / all_targets)) * 100
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        print(f"✅ Training complete!")
        print(f"   MAE: {mae:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   R²: {r2:.4f}")
        print(f"   MAPE: {mape:.2f}%")
    
    return {
        'model': model,
        'scaler': scaler,
        'target_scaler': target_scaler,
        'input_dim': input_dim,
        'task_type': task_type,
        'metrics': metrics,
        'predict_delta': predict_delta if task_type == 'regression' else False
    }


def evaluate_transformer_model(model, scaler, target_scaler, data, config, task_type='regression'):
    """Evaluate transformer model and return metrics."""
    if task_type == 'classification':
        data = data.copy()
        data['target'] = (data['multiplier'] > 20.0).astype(int)
        features, _ = prepare_features_transformer(data, lookback=config.LOOKBACK)
        if features is None:
            return None
        targets = data['target'].iloc[config.LOOKBACK:].values
    else:
        features, targets = prepare_features_transformer(data, lookback=config.LOOKBACK)
    
    if features is None or len(features) == 0:
        return None
    
    # Use last 20% for testing
    split_idx = int(len(features) * 0.8)
    test_features = features[split_idx:]
    test_targets = targets[split_idx:]
    
    test_features_scaled = scaler.transform(test_features)
    
    if task_type == 'regression':
        test_targets_scaled = target_scaler.transform(test_targets.reshape(-1, 1)).flatten()
    else:
        test_targets_scaled = test_targets.astype(float)
    
    test_dataset = MultiplierDataset(test_features_scaled, test_targets_scaled, lookback=config.LOOKBACK)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_features = batch_features.to(config.DEVICE)
            predictions = model(batch_features)
            
            if task_type == 'regression':
                predictions_original = target_scaler.inverse_transform(predictions.cpu().numpy().reshape(-1, 1)).flatten()
                targets_original = target_scaler.inverse_transform(batch_targets.cpu().numpy().reshape(-1, 1)).flatten()
            else:
                predictions_original = predictions.cpu().numpy()
                targets_original = batch_targets.cpu().numpy().flatten()
            
            all_predictions.extend(predictions_original)
            all_targets.extend(targets_original)
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    if task_type == 'classification':
        pred_binary = (all_predictions >= 0.5).astype(int)
        all_targets_binary = all_targets.astype(int)
        
        accuracy = accuracy_score(all_targets_binary, pred_binary)
        precision = precision_score(all_targets_binary, pred_binary, zero_division=0)
        recall = recall_score(all_targets_binary, pred_binary, zero_division=0)
        f1 = f1_score(all_targets_binary, pred_binary, zero_division=0)
        try:
            auc = roc_auc_score(all_targets_binary, all_predictions)
        except:
            auc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'predictions': all_predictions,
            'targets': all_targets
        }
    else:
        all_predictions = np.maximum(1.0, all_predictions)
        all_targets = np.maximum(1.0, all_targets)
        
        mae = mean_absolute_error(all_targets, all_predictions)
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        r2 = r2_score(all_targets, all_predictions)
        mape = np.mean(np.abs((all_targets - all_predictions) / all_targets)) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'predictions': all_predictions,
            'targets': all_targets
        }
