"""
Model Integration Layer - Manages H2O AutoML model with unified interface
Uses H2O AutoML for time-efficient and accurate predictions
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from h2o_automl_model import (
    train_h2o_automl, predict_h2o_automl, H2O_AVAILABLE,
    predict_bayesian, bayesian_seq_predict, bayesian_action
)
import warnings
warnings.filterwarnings('ignore')

# H2O will be initialized when needed
if H2O_AVAILABLE:
    try:
        import h2o
    except ImportError:
        h2o = None


class ModelConfig:
    """Configuration for models"""
    MODEL_DIR = "models"
    DATA_DIR = "data"
    LOOKBACK = 30
    MIN_SAMPLES_FOR_TRAINING = 100
    
    # H2O AutoML parameters
    H2O_MAX_MODELS = 30
    H2O_MAX_RUNTIME_SECS = 300  # 5 minutes max training time
    
    # H2O Model paths
    H2O_MODEL_PATH = os.path.join(MODEL_DIR, "h2o_automl_model")
    H2O_METADATA = os.path.join(MODEL_DIR, "h2o_metadata.json")
    H2O_SCALER_PATH = os.path.join(MODEL_DIR, "h2o_scaler.pkl")
    CURRENT_CSV = None  # Will be set dynamically based on date


class ModelIntegration:
    """Integration layer for managing H2O AutoML model"""
    
    def __init__(self, config=None):
        self.config = config or ModelConfig()
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)
        os.makedirs(self.config.DATA_DIR, exist_ok=True)
        
        # H2O AutoML Model
        self.h2o_model = None
        self.h2o_feature_engineer = None
        self.h2o_metadata = None
        self.h2o_loaded = False
        
        # Update current CSV path
        today = datetime.now().strftime('%Y%m%d')
        self.config.CURRENT_CSV = os.path.join(self.config.DATA_DIR, f"aviator_payouts_global.csv")
    
    def load_h2o_model(self):
        """Load H2O AutoML model"""
        if not H2O_AVAILABLE:
            return False
        
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
                print(f"⚠️ H2O initialization failed: {e}")
                return False
        
        try:
            # Find latest model directory
            if os.path.exists(self.config.H2O_MODEL_PATH):
                model_files = []
                for root, dirs, files in os.walk(self.config.H2O_MODEL_PATH):
                    for file in files:
                        if file.endswith('.zip'):
                            model_files.append(os.path.join(root, file))
                
                if not model_files:
                    return False
                
                # Load most recent model
                latest_model = max(model_files, key=os.path.getctime)
                self.h2o_model = h2o.load_model(latest_model)
                
                # Load feature engineer (will be recreated during prediction)
                from h2o_automl_model import MinimalFeatureEngineer
                self.h2o_feature_engineer = MinimalFeatureEngineer(
                    lookback=self.config.LOOKBACK,
                    prediction_threshold=2.0
                )
                
                # Try to load scaler
                if os.path.exists(self.config.H2O_SCALER_PATH):
                    self.h2o_feature_engineer.load_scaler(self.config.H2O_SCALER_PATH)
                
                # Load metadata
                if os.path.exists(self.config.H2O_METADATA):
                    with open(self.config.H2O_METADATA, 'r') as f:
                        self.h2o_metadata = json.load(f)
                
                self.h2o_loaded = True
                print(f"✅ Loaded H2O AutoML model: {self.h2o_model.model_id}")
                return True
        except Exception as e:
            print(f"⚠️ Could not load H2O model: {e}")
            return False
    
    def train_h2o_model(self, progress_callback=None):
        """
        Train H2O AutoML model on current data.
        Returns: dict with success, metrics, training_time
        """
        if not H2O_AVAILABLE:
            return {"success": False, "error": "H2O AutoML not available. Install with: pip install h2o"}
        
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
                return {"success": False, "error": f"H2O initialization failed: {e}"}
        
        if not os.path.exists(self.config.CURRENT_CSV):
            return {"success": False, "error": "Current CSV file not found"}
        
        try:
            data = pd.read_csv(self.config.CURRENT_CSV)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.sort_values('timestamp').reset_index(drop=True)
            data['multiplier'] = pd.to_numeric(data['multiplier'], errors='coerce')
            data = data.dropna().reset_index(drop=True)
            
            if len(data) < self.config.MIN_SAMPLES_FOR_TRAINING:
                return {"success": False, "error": f"Insufficient data: {len(data)} samples"}
            
            print(f"🔄 Training H2O AutoML model on {len(data)} samples...")
            start_time = datetime.now()
            
            # Create config object for H2O training
            class TempConfig:
                LOOKBACK = self.config.LOOKBACK
                MIN_SAMPLES_FOR_TRAINING = self.config.MIN_SAMPLES_FOR_TRAINING
                H2O_MAX_MODELS = self.config.H2O_MAX_MODELS
                H2O_MAX_RUNTIME_SECS = self.config.H2O_MAX_RUNTIME_SECS
            
            temp_config = TempConfig()
            
            result = train_h2o_automl(
                data=data,
                config=temp_config,
                progress_callback=progress_callback
            )
            
            if result is None:
                return {"success": False, "error": "Training failed"}
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Save model
            os.makedirs(self.config.H2O_MODEL_PATH, exist_ok=True)
            model_path = h2o.save_model(
                model=result['model'],
                path=self.config.H2O_MODEL_PATH,
                force=True
            )
            
            # Save scaler
            result['feature_engineer'].save_scaler(self.config.H2O_SCALER_PATH)
            
            # Save metadata
            metadata = {
                'last_training_time': datetime.now().isoformat(),
                'training_duration_seconds': training_time,
                'training_samples': len(data),
                'metrics': result['metrics'],
                'model_path': model_path,
                'model_id': result['model'].model_id
            }
            
            with open(self.config.H2O_METADATA, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.h2o_model = result['model']
            self.h2o_feature_engineer = result['feature_engineer']
            self.h2o_metadata = metadata
            self.h2o_loaded = True
            
            print(f"✅ H2O AutoML training complete in {training_time:.2f} seconds")
            
            return {
                "success": True,
                "metrics": result['metrics'],
                "training_time": training_time,
                "metadata": metadata
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def predict(self, data):
        """
        Make predictions using H2O AutoML CLASSIFICATION model.
        Returns probability that next multiplier > 2.0
        """
        if not self.h2o_loaded or self.h2o_model is None:
            return None
        
        try:
            # Get probability prediction from H2O (classification)
            prob_gt_2 = predict_h2o_automl(
                {
                    'model': self.h2o_model,
                    'feature_engineer': self.h2o_feature_engineer
                },
                data,
                lookback=self.config.LOOKBACK
            )
            
            if prob_gt_2 is None:
                return None
            
            # Get model metrics for confidence calculation
            auc = self.h2o_metadata.get('metrics', {}).get('auc', 0.5) if self.h2o_metadata else 0.5
            
            # Calculate confidence based on probability and model AUC
            # Higher AUC = more reliable predictions
            confidence = min(100, max(0, (prob_gt_2 * 100) * (auc / 0.6)))  # Normalize by expected AUC
            
            # Determine betting decision based on probability thresholds
            if prob_gt_2 >= 0.65:
                betting_action = "BET"
                risk_level = "LOW"
            elif prob_gt_2 >= 0.55:
                betting_action = "SMALL_BET"
                risk_level = "MEDIUM"
            else:
                betting_action = "NO_BET"
                risk_level = "HIGH"
            
            # Estimate expected multiplier range (for display purposes only)
            # Based on probability, estimate likely range
            if prob_gt_2 > 0.6:
                estimated_min = 2.0
                estimated_max = 3.5
            elif prob_gt_2 > 0.5:
                estimated_min = 1.5
                estimated_max = 2.5
            else:
                estimated_min = 1.0
                estimated_max = 2.0
            
            # Get input sequence
            lookback = self.config.LOOKBACK
            input_sequence = data['multiplier'].tail(lookback).values.tolist() if len(data) >= lookback else data['multiplier'].values.tolist()
            recent_actuals = [float(x) for x in data['multiplier'].tail(20).tolist()]
            
            return {
                'probability_gt_2': float(round(prob_gt_2, 4)),
                'confidence': float(round(confidence, 1)),
                'betting_action': betting_action,
                'risk_level': risk_level,
                'estimated_min': float(round(estimated_min, 2)),
                'estimated_max': float(round(estimated_max, 2)),
                'timestamp': datetime.now().isoformat(),
                'last_actual': float(data['multiplier'].iloc[-1]) if len(data) > 0 else None,
                'input_sequence': [float(x) for x in input_sequence],
                'recent_actuals': recent_actuals,
                'model_type': 'h2o_automl_classification',
                'model_auc': float(round(auc, 4)),
                # Backward compatibility fields
                'predicted_value': float(round(estimated_min, 2)),  # For UI compatibility
                'min_range': float(round(estimated_min, 2)),
                'max_range': float(round(estimated_max, 2))
            }
        except Exception as e:
            print(f"⚠️ H2O prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def is_model_relevant(self, metadata_path, max_age_hours=24, min_data_samples=None):
        """
        Check if a model is still relevant based on:
        1. Model age (default: 24 hours)
        2. Data freshness (if new data available)
        3. Minimum samples used for training
        
        Args:
            metadata_path: Path to model metadata JSON
            max_age_hours: Maximum age in hours before considering model stale
            min_data_samples: Minimum samples that should have been used for training
        
        Returns:
            tuple: (is_relevant: bool, reason: str)
        """
        if not os.path.exists(metadata_path):
            return False, "Model metadata not found"
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check 1: Model age
            last_training_str = metadata.get('last_training_time')
            if not last_training_str:
                return False, "No training timestamp in metadata"
            
            last_training = datetime.fromisoformat(last_training_str)
            age_hours = (datetime.now() - last_training).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                return False, f"Model is {age_hours:.1f} hours old (max: {max_age_hours}h)"
            
            # Check 2: Training samples
            training_samples = metadata.get('training_samples', 0)
            if min_data_samples and training_samples < min_data_samples:
                return False, f"Insufficient training samples: {training_samples} < {min_data_samples}"
            
            # Check 3: Data freshness (compare with current data)
            if os.path.exists(self.config.CURRENT_CSV):
                try:
                    current_data = pd.read_csv(self.config.CURRENT_CSV)
                    current_samples = len(current_data)
                    
                    # If current data has significantly more samples, model might be stale
                    if training_samples > 0 and current_samples > training_samples * 1.5:
                        return False, f"Data has grown significantly: {current_samples} vs {training_samples} training samples"
                except:
                    pass  # If we can't read current data, assume model is still relevant
            
            return True, f"Model is relevant (age: {age_hours:.1f}h, samples: {training_samples})"
            
        except Exception as e:
            return False, f"Error checking model relevance: {str(e)}"
    
    def should_retrain_h2o(self):
        """Check if H2O model should be retrained (every 30 min for live updates)"""
        is_relevant, reason = self.is_model_relevant(
            self.config.H2O_METADATA,
            max_age_hours=0.5,  # 30 minutes for live retraining
            min_data_samples=self.config.MIN_SAMPLES_FOR_TRAINING
        )
        return not is_relevant
    
    def get_model_status(self):
        """Get status of H2O model (maintains compatibility with old interface)"""
        status = {
            'model1': {
                'loaded': False,  # Disabled
                'metadata': None,
                'should_retrain': False
            },
            'model2': {
                'loaded': self.h2o_loaded,  # H2O model status
                'metadata': self.h2o_metadata,
                'should_retrain': self.should_retrain_h2o()
            }
        }
        return status

    # Backward compatibility methods (disabled)
    def load_model1(self):
        """Model 1 is disabled"""
        return False
    
    def load_model2(self):
        """Load H2O model (backward compatibility)"""
        return self.load_h2o_model()
    
    def train_model1(self, progress_callback=None):
        """Model 1 is disabled"""
        return {"success": False, "error": "Model 1 training is disabled"}
    
    def train_model2(self, progress_callback=None):
        """Train H2O model (backward compatibility)"""
        return self.train_h2o_model(progress_callback)
    
    def should_retrain_model1(self):
        """Model 1 is disabled"""
        return False
    
    def should_retrain_model2(self):
        """Check if H2O should retrain (backward compatibility)"""
        return self.should_retrain_h2o()
    
    def load_combined_ensemble(self):
        """Load H2O model (backward compatibility)"""
        return self.load_h2o_model()
    
    def train_combined_ensemble(self, progress_callback=None):
        """Train H2O model (backward compatibility)"""
        return self.train_h2o_model(progress_callback)
    
    def train_sklearn_model(self, progress_callback=None):
        """Sklearn is disabled - use H2O instead"""
        return {"success": False, "error": "Sklearn model is disabled. Use H2O AutoML instead."}
    
    # ============================================================================
    # BAYESIAN SEQUENCE MODEL METHODS
    # ============================================================================
    
    def predict_bayesian(self, data, threshold=2.0, window=20):
        """
        Make predictions using Bayesian sequence-based model.
        
        Args:
            data: DataFrame with multiplier data
            threshold: Success threshold for prediction
            window: Recent window size
        
        Returns:
            dict with Bayesian prediction, action, and metadata
        """
        try:
            prediction = predict_bayesian(
                data=data,
                threshold=threshold,
                window=window
            )
            return prediction
        except Exception as e:
            print(f"⚠️ Bayesian prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None
