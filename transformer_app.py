"""
H2O AutoML-Based Aviator Prediction Flask Application
Uses ModelIntegration layer for managing H2O AutoML model
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import time
import json
from datetime import datetime, timedelta
import os
import glob
import csv
import pandas as pd
import numpy as np
from model_integration import ModelIntegration, ModelConfig
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
data_folder = "data"
global_csv_filename = os.path.join(data_folder, "aviator_payouts_global.csv")

# ============================================================================
# FLASK APPLICATION
# ============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'transformer-aviator-secret-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize Model Integration
integration = ModelIntegration()

# Global system state
system_state = {
    'is_running': False,
    'is_trained': False,
    'current_prediction': None,
    'current_bayesian_prediction': None,
    'prediction_history': [],
    'last_processed_row': 0,
    'retraining': False,
    'training_lock': threading.Lock(),  # Lock to prevent simultaneous training
    'betting_stats': {
        'bankroll': 50000.0,
        'total_bets': 0,
        'wins': 0,
        'losses': 0,
        'total_profit': 0.0
    },
    'betting_stats_bayesian': {
        'bankroll': 50000.0,
        'total_bets': 0,
        'wins': 0,
        'losses': 0,
        'total_profit': 0.0
    },
    'betting_history': [],
    'simulation_results': {
        'h2o': None,
        'bayesian': None
    }
}

# Bet log files
BET_LOG_PATH = os.path.join(integration.config.MODEL_DIR, "h2o_bet_history.csv")
BET_LOG_BAYESIAN_PATH = os.path.join(integration.config.MODEL_DIR, "bayesian_bet_history.csv")
SIMULATION_RESULTS_PATH = os.path.join(integration.config.MODEL_DIR, "simulation_results.json")

def load_current_csv():
    """Load the current day CSV file."""
    today = datetime.now().strftime('%Y%m%d')
    current_csv = os.path.join(data_folder, f"aviator_payouts_{today}.csv")
    
    if not os.path.exists(current_csv):
        return None, None
    
    try:
        df = pd.read_csv(current_csv)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['multiplier'] = pd.to_numeric(df['multiplier'], errors='coerce')
        df = df[df['multiplier'] >= 1.0]
        df = df.dropna().reset_index(drop=True)
        return df, current_csv
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None, None

def init_system():
    """Initialize the system - loads existing model or trains if needed."""
    os.makedirs(integration.config.MODEL_DIR, exist_ok=True)
    os.makedirs(integration.config.DATA_DIR, exist_ok=True)
    
    # Load current CSV
    df, filename = load_current_csv()
    if df is not None:
        system_state['last_processed_row'] = len(df)
        print(f"✅ Found CSV: {filename} ({len(df)} records)")
    
    # Check and load H2O model
    print("\n📊 Checking H2O AutoML model...")
    h2o_loaded = integration.load_h2o_model()
    
    if h2o_loaded:
        # Check if model is still relevant
        is_relevant, reason = integration.is_model_relevant(
            integration.config.H2O_METADATA,
            max_age_hours=24,  # Models are valid for 24 hours
            min_data_samples=integration.config.MIN_SAMPLES_FOR_TRAINING
        )
        
        if is_relevant:
            system_state['is_trained'] = True
            print(f"✅ H2O AutoML model loaded and relevant")
            print(f"   {reason}")
            if integration.h2o_metadata:
                metrics = integration.h2o_metadata.get('metrics', {})
                print(f"   AUC: {metrics.get('auc', 'N/A'):.4f}")
                print(f"   LogLoss: {metrics.get('logloss', 'N/A'):.4f}")
                print(f"   Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            
            # Make initial predictions to show on UI
            prediction = predict_next()
            if prediction:
                system_state['current_prediction'] = prediction
                socketio.emit('new_prediction', prediction)
            
            # Make initial Bayesian prediction
            bayesian_prediction = predict_next_bayesian(threshold=2.0)
            if bayesian_prediction:
                system_state['current_bayesian_prediction'] = bayesian_prediction
                socketio.emit('new_bayesian_prediction', bayesian_prediction)
        else:
            print(f"⚠️ H2O AutoML model exists but is not relevant: {reason}")
            print("🔄 Retraining H2O AutoML model...")
            result = train_h2o_model()
            if result.get('success'):
                system_state['is_trained'] = True
                print("✅ H2O AutoML model retrained and ready!")
                
                # Make initial prediction
                prediction = predict_next()
                if prediction:
                    system_state['current_prediction'] = prediction
                    socketio.emit('new_prediction', prediction)
            else:
                print(f"⚠️ H2O AutoML training failed: {result.get('error')}")
                system_state['is_trained'] = False
    else:
        print("🔄 No existing model found. Training new H2O AutoML model...")
        result = train_h2o_model()
        if result.get('success'):
            system_state['is_trained'] = True
            print("✅ H2O AutoML model ready!")
            
            # Make initial prediction
            prediction = predict_next()
            if prediction:
                system_state['current_prediction'] = prediction
                socketio.emit('new_prediction', prediction)
        else:
            print(f"⚠️ H2O AutoML training failed: {result.get('error')}")
            system_state['is_trained'] = False
    
    print("\n✅ System initialized")

def train_h2o_model():
    """Train H2O AutoML model - with lock to prevent simultaneous training."""
    if not system_state['training_lock'].acquire(blocking=False):
        print("⚠️ Another model is training. H2O AutoML training skipped.")
        return {"error": "Another model is currently training"}
    
    try:
        def progress_callback(epoch, total, progress):
            socketio.emit('training_status', {
                'status': 'training',
                'message': f'Training H2O AutoML: Progress {progress}%',
                'progress': progress
            })
        
        result = integration.train_h2o_model(progress_callback=progress_callback)
        
        if result.get('success'):
            system_state['is_trained'] = True
            metrics = result['metrics']
            socketio.emit('training_status', {
                'status': 'complete',
                'message': f'H2O AutoML training complete! MAE: {metrics.get("mae", 0):.4f}',
                'progress': 100
            })
            socketio.emit('model_metrics', {
                'model': 'h2o',
                'metrics': metrics,
                'last_training_time': result['metadata']['last_training_time']
            })
            return {"success": True, **metrics}
        else:
            return {"error": result.get('error', 'Training failed')}
    finally:
        system_state['training_lock'].release()

# Backward compatibility functions
def train_model1():
    """Model 1 is disabled"""
    return {"error": "Model 1 training is disabled"}

def train_combined_ensemble():
    """Train H2O AutoML model (backward compatibility)"""
    return train_h2o_model()

def train_model2():
    """Train H2O AutoML model (backward compatibility)"""
    return train_h2o_model()

def log_bet_to_csv(bet_record, file_path=None):
    """Save bet record to CSV."""
    if file_path is None:
        file_path = BET_LOG_PATH
    
    try:
        os.makedirs(integration.config.MODEL_DIR, exist_ok=True)
        
        # Determine headers based on model type
        if bet_record.get('model_type') == 'bayesian_sequence':
            headers = ['timestamp', 'probability_gt_threshold', 'confidence', 'betting_action', 'risk_level',
                      'target', 'bet_amount', 'status', 'result', 'actual', 'profit_loss',
                      'bankroll_before', 'bankroll_after', 'reason', 'model_type']
        else:
            headers = ['timestamp', 'probability_gt_2', 'betting_action', 'estimated_min', 'estimated_max', 
                      'target', 'bet_amount', 'status', 'result', 'actual', 'profit_loss',
                      'bankroll_before', 'bankroll_after', 'input_sequence', 'confidence', 'model_type']
        
        write_headers = not os.path.exists(file_path)
        with open(file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if write_headers:
                writer.writeheader()
            row = {h: bet_record.get(h, '') for h in headers}
            writer.writerow(row)
    except Exception as e:
        print(f"⚠ Could not log bet to CSV: {e}")

def predict_next():
    """Make prediction using H2O AutoML model."""
    if not system_state['is_trained']:
        return None
    
    df, _ = load_current_csv()
    if df is None or len(df) < integration.config.LOOKBACK + 10:
        return None
    
    try:
        # Use H2O AutoML prediction
        prediction = integration.predict(data=df)
        return prediction
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict_next_bayesian(threshold=2.0):
    """Make prediction using Bayesian sequence model."""
    df, _ = load_current_csv()
    if df is None or len(df) < 20:  # Need at least 20 samples for Bayesian
        return None
    
    try:
        # Use Bayesian prediction
        prediction = integration.predict_bayesian(data=df, threshold=threshold, window=20)
        return prediction
    except Exception as e:
        print(f"Bayesian prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# BETTING SIMULATION FUNCTIONS
# ============================================================================

def simulate_h2o_strategy(df, initial_balance=50000.0, prob_threshold=0.7, target_multiplier=8.0):
    """
    Simulate betting strategy for H2O Binary Classification approach.
    
    Args:
        df: DataFrame with multiplier data
        initial_balance: Starting bankroll
        prob_threshold: Probability threshold for betting (prob >= threshold)
        target_multiplier: Target multiplier to cashout at
    
    Returns:
        dict with simulation results
    """
    balance = initial_balance
    balances = [balance]
    bets = []
    outcomes = []
    bet_amount = 100.0
    
    # We need to predict BEFORE seeing the next multiplier
    for i in range(len(df) - 1):
        # Use data up to index i to predict i+1
        history = df.iloc[:i+1].copy()
        if len(history) < integration.config.LOOKBACK + 10:  # warm-up
            balances.append(balance)
            continue
        
        # Get prediction
        try:
            prediction = integration.predict(data=history)
            if prediction is None:
                balances.append(balance)
                continue
            
            prob_gt_2 = prediction.get('probability_gt_2', 0.0)
            
            # Bet if probability >= threshold
            if prob_gt_2 >= prob_threshold and balance >= bet_amount:
                # Place bet
                balance -= bet_amount
                
                # Check result
                next_mult = df['multiplier'].iloc[i+1]
                win = next_mult >= target_multiplier
                
                if win:
                    profit = (target_multiplier - 1) * bet_amount
                    balance += profit
                    outcomes.append(1)
                else:
                    outcomes.append(0)
                
                bets.append(bet_amount)
            else:
                bets.append(0.0)
                outcomes.append(0)
            
            balances.append(balance)
            
            if balance <= 0.01:
                print(f"⚠️ Ruin at step {i}")
                break
        except Exception as e:
            balances.append(balance)
            bets.append(0.0)
            outcomes.append(0)
            continue
    
    # Calculate metrics
    total_return = (balance / initial_balance - 1) * 100
    total_bets = sum(1 for b in bets if b > 0)
    wins = sum(outcomes)
    losses = total_bets - wins
    
    if len(balances) > 1:
        returns = np.diff(balances)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) if np.std(returns) > 0 else 0
        max_drawdown = ((np.maximum.accumulate(balances) - balances).max() / 
                       np.maximum.accumulate(balances).max() * 100) if len(balances) > 0 else 0
    else:
        sharpe = 0
        max_drawdown = 0
    
    return {
        'final_balance': balance,
        'total_return_pct': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_drawdown,
        'total_bets': total_bets,
        'wins': wins,
        'losses': losses,
        'win_rate': (wins / total_bets * 100) if total_bets > 0 else 0,
        'balances': balances,
        'bets': bets,
        'outcomes': outcomes
    }


def simulate_bayesian_strategy(df, initial_balance=50000.0, threshold=2.0, window=20):
    """
    Simulate betting strategy for Bayesian sequence-based approach.
    
    Args:
        df: DataFrame with multiplier data
        initial_balance: Starting bankroll
        threshold: Success threshold for Bayesian prediction
        window: Window size for Bayesian model
    
    Returns:
        dict with simulation results
    """
    balance = initial_balance
    balances = [balance]
    bets = []
    outcomes = []
    
    # We need to predict BEFORE seeing the next multiplier
    for i in range(len(df) - 1):
        # Use data up to index i to predict i+1
        history = df.iloc[:i+1].copy()
        if len(history) < window:  # warm-up
            balances.append(balance)
            continue
        
        # Get Bayesian prediction
        try:
            prediction = integration.predict_bayesian(
                data=history,
                threshold=threshold,
                window=window
            )
            
            if prediction is None:
                balances.append(balance)
                continue
            
            prob = prediction.get('probability_gt_threshold', 0.0)
            confidence = prediction.get('confidence', 0.0) / 100.0  # Convert back to 0-1
            action = prediction.get('betting_action', 'NO_BET')
            
            # Determine bet amount based on action
            if action == 'BET' and confidence >= 0.5 and balance >= 100.0:
                bet_amount = 100.0
            elif action == 'SMALL_BET' and confidence >= 0.5 and balance >= 100.0:
                bet_amount = 100.0
            else:
                bet_amount = 0.0
            
            if bet_amount > 0:
                # Place bet
                balance -= bet_amount
                
                # Check result (use threshold as target)
                next_mult = df['multiplier'].iloc[i+1]
                win = next_mult >= threshold
                
                if win:
                    profit = (threshold - 1) * bet_amount
                    balance += profit
                    outcomes.append(1)
                else:
                    outcomes.append(0)
                
                bets.append(bet_amount)
            else:
                bets.append(0.0)
                outcomes.append(0)
            
            balances.append(balance)
            
            if balance <= 0.01:
                print(f"⚠️ Ruin at step {i}")
                break
        except Exception as e:
            balances.append(balance)
            bets.append(0.0)
            outcomes.append(0)
            continue
    
    # Calculate metrics
    total_return = (balance / initial_balance - 1) * 100
    total_bets = sum(1 for b in bets if b > 0)
    wins = sum(outcomes)
    losses = total_bets - wins
    
    if len(balances) > 1:
        returns = np.diff(balances)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) if np.std(returns) > 0 else 0
        max_drawdown = ((np.maximum.accumulate(balances) - balances).max() / 
                       np.maximum.accumulate(balances).max() * 100) if len(balances) > 0 else 0
    else:
        sharpe = 0
        max_drawdown = 0
    
    return {
        'final_balance': balance,
        'total_return_pct': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_drawdown,
        'total_bets': total_bets,
        'wins': wins,
        'losses': losses,
        'win_rate': (wins / total_bets * 100) if total_bets > 0 else 0,
        'balances': balances,
        'bets': bets,
        'outcomes': outcomes
    }


def run_betting_simulations():
    """Run betting simulations for both H2O and Bayesian approaches."""
    df, _ = load_current_csv()
    if df is None or len(df) < 1000:
        return {"error": "Insufficient data for simulation (need at least 1000 samples)"}
    
    print("🔄 Running betting simulations...")
    
    # Run H2O simulation
    print("   Running H2O Binary Classification simulation...")
    h2o_results = simulate_h2o_strategy(df, prob_threshold=0.7, target_multiplier=8.0)
    
    # Run Bayesian simulation
    print("   Running Bayesian Sequence simulation...")
    bayesian_results = simulate_bayesian_strategy(df, threshold=2.0, window=20)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'h2o_binary': h2o_results,
        'bayesian': bayesian_results
    }
    
    with open(SIMULATION_RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    
    system_state['simulation_results'] = results
    
    print("✅ Simulations complete!")
    print(f"   H2O: Return {h2o_results['total_return_pct']:.2f}%, Sharpe {h2o_results['sharpe_ratio']:.3f}")
    print(f"   Bayesian: Return {bayesian_results['total_return_pct']:.2f}%, Sharpe {bayesian_results['sharpe_ratio']:.3f}")
    
    return results

def retrain_models_background():
    """Background retraining function - trains H2O AutoML model."""
    if system_state['retraining']:
        return
    
    try:
        system_state['retraining'] = True
        
        # Retrain H2O model if needed (every 30 min)
        if integration.should_retrain_h2o():
            print(f"\n🔄 Background retraining H2O AutoML model...")
            result = train_h2o_model()  # Uses lock internally
            if result.get('success'):
                system_state['is_trained'] = True
                print("✅ H2O AutoML retraining complete!")
                # Make predictions immediately after retraining
                prediction = predict_next()
                if prediction:
                    system_state['current_prediction'] = prediction
                    socketio.emit('new_prediction', prediction)
                
                bayesian_prediction = predict_next_bayesian(threshold=2.0)
                if bayesian_prediction:
                    system_state['current_bayesian_prediction'] = bayesian_prediction
                    socketio.emit('new_bayesian_prediction', bayesian_prediction)
            else:
                print(f"⚠️ H2O AutoML retraining failed: {result.get('error')}")
        
    except Exception as e:
        print(f"❌ Background retraining error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        system_state['retraining'] = False

def csv_monitor_loop():
    """Monitor CSV file for updates and make predictions with betting."""
    print("📊 Starting CSV monitoring loop...")
    last_count = system_state['last_processed_row']
    last_prediction_time = None
    pending_bet = None
    pending_bet_bayesian = None
    
    while system_state['is_running']:
        try:
            df, filename = load_current_csv()
            if df is None:
                time.sleep(1)
                continue
            
            current_count = len(df)
            
            # Check for new data
            if current_count > last_count:
                new_rows = current_count - last_count
                print(f"📈 New data detected: {new_rows} new rows (total: {current_count})")
                
                # Process pending bet if we have new actual multiplier
                if pending_bet is not None and current_count > last_count:
                    actual_multiplier = float(df['multiplier'].iloc[-1])
                    target = pending_bet['target']
                    bet_amount = pending_bet['bet_amount']
                    bankroll_before = pending_bet['bankroll_before']
                    
                    # Check if bet won
                    if actual_multiplier >= target:
                        profit = (target - 1) * bet_amount
                        bankroll_after = bankroll_before + profit
                        result = 'win'
                    else:
                        profit = -bet_amount
                        bankroll_after = bankroll_before + profit
                        result = 'loss'
                    
                    # Update betting stats
                    system_state['betting_stats']['bankroll'] = bankroll_after
                    system_state['betting_stats']['total_bets'] += 1
                    if result == 'win':
                        system_state['betting_stats']['wins'] += 1
                    else:
                        system_state['betting_stats']['losses'] += 1
                    system_state['betting_stats']['total_profit'] += profit
                    
                    # Update bet record
                    pending_bet['status'] = 'completed'
                    pending_bet['result'] = result
                    pending_bet['actual'] = actual_multiplier
                    pending_bet['profit_loss'] = profit
                    pending_bet['bankroll_after'] = bankroll_after
                    pending_bet['model_type'] = 'h2o_binary'
                    
                    # Log bet
                    log_bet_to_csv(pending_bet, BET_LOG_PATH)
                    
                    # Emit bet result
                    socketio.emit('bet_result', {
                        'result': result,
                        'actual': actual_multiplier,
                        'target': target,
                        'profit': profit,
                        'bankroll': bankroll_after
                    })
                    
                    pending_bet = None
                
                # Make predictions for both approaches
                prediction = predict_next()
                bayesian_prediction = predict_next_bayesian(threshold=2.0)
                
                if prediction:
                    system_state['current_prediction'] = prediction
                    socketio.emit('new_prediction', prediction)
                
                if bayesian_prediction:
                    system_state['current_bayesian_prediction'] = bayesian_prediction
                    socketio.emit('new_bayesian_prediction', bayesian_prediction)
                
                last_prediction_time = datetime.now()
                
                # ========== H2O BINARY CLASSIFICATION BETTING ==========
                if prediction:
                    prob_gt_2 = prediction.get('probability_gt_2', 0.0)
                    betting_action = prediction.get('betting_action', 'NO_BET')
                    
                    # Betting logic based on probability thresholds
                    if betting_action == 'BET' and prob_gt_2 >= 0.7:
                        # High confidence bet
                        target = 8  # Bet on multiplier > 2.0
                        bet_amount = 100.0
                        bankroll = system_state['betting_stats']['bankroll']
                        
                        if bankroll >= bet_amount:
                            bankroll_before = bankroll
                            bankroll_after = bankroll - bet_amount
                            
                            bet_record = {
                                'timestamp': prediction['timestamp'],
                                'probability_gt_2': prob_gt_2,
                                'betting_action': betting_action,
                                'estimated_min': prediction.get('estimated_min', 2.0),
                                'estimated_max': prediction.get('estimated_max', 3.0),
                                'target': target,
                                'bet_amount': bet_amount,
                                'status': 'pending',
                                'result': None,
                                'actual': None,
                                'profit_loss': None,
                                'bankroll_before': bankroll_before,
                                'bankroll_after': bankroll_after,
                                'input_sequence': str(prediction['input_sequence']),
                                'confidence': prediction.get('confidence', 0),
                                'model_type': 'h2o_binary'
                            }
                            
                            pending_bet = bet_record
                            system_state['betting_stats']['bankroll'] = bankroll_after
                            
                            socketio.emit('bet_placed', bet_record)
                    elif betting_action == 'SMALL_BET' and prob_gt_2 >= 0.55:
                        # Medium confidence - smaller bet
                        target = 5
                        bet_amount = 100  # Smaller bet amount
                        bankroll = system_state['betting_stats']['bankroll']
                        
                        if bankroll >= bet_amount:
                            bankroll_before = bankroll
                            bankroll_after = bankroll - bet_amount
                            
                            bet_record = {
                                'timestamp': prediction['timestamp'],
                                'probability_gt_2': prob_gt_2,
                                'betting_action': betting_action,
                                'estimated_min': prediction.get('estimated_min', 1.5),
                                'estimated_max': prediction.get('estimated_max', 2.5),
                                'target': target,
                                'bet_amount': bet_amount,
                                'status': 'pending',
                                'result': None,
                                'actual': None,
                                'profit_loss': None,
                                'bankroll_before': bankroll_before,
                                'bankroll_after': bankroll_after,
                                'input_sequence': str(prediction['input_sequence']),
                                'confidence': prediction.get('confidence', 0),
                                'model_type': 'h2o_binary'
                            }
                            
                            pending_bet = bet_record
                            system_state['betting_stats']['bankroll'] = bankroll_after
                            
                            socketio.emit('bet_placed', bet_record)
                
                # ========== BAYESIAN SEQUENCE BETTING ==========
                if bayesian_prediction:
                    bayes_prob = bayesian_prediction.get('probability_gt_threshold', 0.0)
                    bayes_confidence = bayesian_prediction.get('confidence', 0.0) / 100.0  # Convert to 0-1
                    bayes_action = bayesian_prediction.get('betting_action', 'NO_BET')
                    
                    # Betting logic based on Bayesian action and confidence
                    if bayes_action == 'BET' and bayes_confidence >= 0.5:
                        target = 2.0  # Bet on multiplier >= 2.0
                        bet_amount = 100.0
                        bankroll = system_state['betting_stats_bayesian']['bankroll']
                        
                        if bankroll >= bet_amount:
                            bankroll_before = bankroll
                            bankroll_after = bankroll - bet_amount
                            
                            bet_record = {
                                'timestamp': bayesian_prediction['timestamp'],
                                'probability_gt_threshold': bayes_prob,
                                'confidence': bayesian_prediction.get('confidence', 0),
                                'betting_action': bayes_action,
                                'risk_level': bayesian_prediction.get('risk_level', 'HIGH'),
                                'target': target,
                                'bet_amount': bet_amount,
                                'status': 'pending',
                                'result': None,
                                'actual': None,
                                'profit_loss': None,
                                'bankroll_before': bankroll_before,
                                'bankroll_after': bankroll_after,
                                'reason': bayesian_prediction.get('reason', ''),
                                'model_type': 'bayesian_sequence'
                            }
                            
                            pending_bet_bayesian = bet_record
                            system_state['betting_stats_bayesian']['bankroll'] = bankroll_after
                            
                            socketio.emit('bet_placed_bayesian', bet_record)
                    elif bayes_action == 'SMALL_BET' and bayes_confidence >= 0.5:
                        target = 2.0
                        bet_amount = 100.0
                        bankroll = system_state['betting_stats_bayesian']['bankroll']
                        
                        if bankroll >= bet_amount:
                            bankroll_before = bankroll
                            bankroll_after = bankroll - bet_amount
                            
                            bet_record = {
                                'timestamp': bayesian_prediction['timestamp'],
                                'probability_gt_threshold': bayes_prob,
                                'confidence': bayesian_prediction.get('confidence', 0),
                                'betting_action': bayes_action,
                                'risk_level': bayesian_prediction.get('risk_level', 'MEDIUM'),
                                'target': target,
                                'bet_amount': bet_amount,
                                'status': 'pending',
                                'result': None,
                                'actual': None,
                                'profit_loss': None,
                                'bankroll_before': bankroll_before,
                                'bankroll_after': bankroll_after,
                                'reason': bayesian_prediction.get('reason', ''),
                                'model_type': 'bayesian_sequence'
                            }
                            
                            pending_bet_bayesian = bet_record
                            system_state['betting_stats_bayesian']['bankroll'] = bankroll_after
                            
                            socketio.emit('bet_placed_bayesian', bet_record)
                
                # Process pending Bayesian bet if we have new actual multiplier
                if pending_bet_bayesian is not None and current_count > last_count:
                    actual_multiplier = float(df['multiplier'].iloc[-1])
                    target = pending_bet_bayesian['target']
                    bet_amount = pending_bet_bayesian['bet_amount']
                    bankroll_before = pending_bet_bayesian['bankroll_before']
                    
                    # Check if bet won
                    if actual_multiplier >= target:
                        profit = (target - 1) * bet_amount
                        bankroll_after = bankroll_before + profit
                        result = 'win'
                    else:
                        profit = -bet_amount
                        bankroll_after = bankroll_before + profit
                        result = 'loss'
                    
                    # Update betting stats
                    system_state['betting_stats_bayesian']['bankroll'] = bankroll_after
                    system_state['betting_stats_bayesian']['total_bets'] += 1
                    if result == 'win':
                        system_state['betting_stats_bayesian']['wins'] += 1
                    else:
                        system_state['betting_stats_bayesian']['losses'] += 1
                    system_state['betting_stats_bayesian']['total_profit'] += profit
                    
                    # Update bet record
                    pending_bet_bayesian['status'] = 'completed'
                    pending_bet_bayesian['result'] = result
                    pending_bet_bayesian['actual'] = actual_multiplier
                    pending_bet_bayesian['profit_loss'] = profit
                    pending_bet_bayesian['bankroll_after'] = bankroll_after
                    
                    # Log bet
                    log_bet_to_csv(pending_bet_bayesian, BET_LOG_BAYESIAN_PATH)
                    
                    # Emit bet result
                    socketio.emit('bet_result_bayesian', {
                        'result': result,
                        'actual': actual_multiplier,
                        'target': target,
                        'profit': profit,
                        'bankroll': bankroll_after
                    })
                    
                    pending_bet_bayesian = None
                
                last_count = current_count
                
                # Check if retraining is needed (only if not already training)
                if not system_state['retraining'] and integration.should_retrain_h2o():
                    threading.Thread(target=retrain_models_background, daemon=True).start()
            
            time.sleep(0.5)
            
        except Exception as e:
            print(f"⚠️ Monitoring error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(2)

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Main page."""
    return render_template('transformer_index.html')

@app.route('/api/status')
def api_status():
    """Get system status."""
    status = integration.get_model_status()
    return jsonify({
        'is_running': system_state['is_running'],
        'is_trained': system_state['is_trained'],
        'h2o_model': {
            'loaded': status['model2']['loaded'],
            'should_retrain': status['model2']['should_retrain'],
            'last_training': status['model2']['metadata'].get('last_training_time') if status['model2']['metadata'] else None,
            'metrics': status['model2']['metadata'].get('metrics', {}) if status['model2']['metadata'] else {}
        },
        'bayesian_model': {
            'loaded': True,  # Bayesian is always available (no training needed)
            'type': 'sequence_based'
        },
        'betting_stats': system_state['betting_stats'],
        'betting_stats_bayesian': system_state['betting_stats_bayesian']
    })

@app.route('/api/train', methods=['POST'])
def api_train():
    """Train H2O AutoML model."""
    data = request.json or {}
    model = data.get('model', 'h2o')  # 'h2o', 'combined', 'model2' (all map to H2O)
    
    # Handle aliases for backward compatibility
    if model in ['combined', 'both', 'model2', 'transformer', 'sklearn']:
        model = 'h2o'
    elif model == 'model1':
        return jsonify({'success': False, 'error': 'Model 1 (Classification) training is disabled'}), 400
    
    if model == 'h2o':
        result = train_h2o_model()
        if result.get('success'):
            return jsonify({'success': True, 'message': 'H2O AutoML training completed', 'metrics': result.get('metrics', {})})
        else:
            return jsonify({'success': False, 'error': result.get('error', 'Training failed')}), 500
    else:
        return jsonify({'success': False, 'error': f'Invalid model type: {model}. Valid types: h2o, combined, model2'}), 400

@app.route('/api/start', methods=['POST'])
def api_start():
    """Start monitoring."""
    if not system_state['is_trained']:
        return jsonify({'error': 'Model not trained. Please train first.'}), 400
    
    if system_state['is_running']:
        return jsonify({'error': 'Already running'}), 400
    
    system_state['is_running'] = True
    threading.Thread(target=csv_monitor_loop, daemon=True).start()
    return jsonify({'success': True, 'message': 'Monitoring started'})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Stop monitoring."""
    system_state['is_running'] = False
    return jsonify({'success': True, 'message': 'Monitoring stopped'})

@app.route('/api/predict', methods=['GET'])
def api_predict():
    """Get current prediction (H2O Binary Classification)."""
    prediction = predict_next()
    if prediction:
        return jsonify(prediction)
    else:
        return jsonify({'error': 'Prediction not available'}), 404

@app.route('/api/predict_bayesian', methods=['GET'])
def api_predict_bayesian():
    """Get current prediction (Bayesian Sequence Model)."""
    threshold = request.args.get('threshold', 2.0, type=float)
    prediction = predict_next_bayesian(threshold=threshold)
    if prediction:
        return jsonify(prediction)
    else:
        return jsonify({'error': 'Bayesian prediction not available'}), 404

@app.route('/api/simulate', methods=['POST'])
def api_simulate():
    """Run betting simulations for both approaches."""
    try:
        results = run_betting_simulations()
        if 'error' in results:
            return jsonify({'success': False, 'error': results['error']}), 400
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/simulation_results', methods=['GET'])
def api_simulation_results():
    """Get simulation results."""
    if system_state['simulation_results']['h2o'] is None:
        return jsonify({'error': 'No simulation results available. Run simulation first.'}), 404
    return jsonify(system_state['simulation_results'])

@app.route('/api/history')
def api_history():
    """Get betting history."""
    return jsonify(system_state['betting_history'])

@app.route('/api/betting_stats')
def api_betting_stats():
    """Get betting statistics."""
    return jsonify(system_state['betting_stats'])

# ============================================================================
# WEBSOCKET EVENTS
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print("Client connected")
    emit('connected', {'status': 'connected'})
    
    # Send current status
    status = integration.get_model_status()
    emit('model_metrics', {
        'model': 'h2o',
        'metrics': status['model2']['metadata'].get('metrics', {}) if status['model2']['metadata'] else {},
        'last_training_time': status['model2']['metadata'].get('last_training_time') if status['model2']['metadata'] else None
    })
    
    if system_state['current_prediction']:
        emit('new_prediction', system_state['current_prediction'])
    
    if system_state['current_bayesian_prediction']:
        emit('new_bayesian_prediction', system_state['current_bayesian_prediction'])

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print("Client disconnected")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    init_system()
    print("\n🚀 Starting Flask server...")
    print("📊 Access the UI at: http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5002, debug=False)
