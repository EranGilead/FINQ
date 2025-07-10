"""
Model inference module for FINQ Stock Predictor.
Handles model inference and prediction serving.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
from datetime import datetime, timedelta
import joblib
import os
import sys
import glob

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_config import MODELS_DIR, PREDICTION_HORIZON_DAYS, MARKET_TIMEZONE
from data.fetcher import DataFetcher
from features.engineer import FeatureEngineer




class ModelInference:
    """
    Handles model inference for stock prediction.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize ModelInference.
        
        Args:
            model_path: Path to saved model file
        """
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.model_name = None
        self.data_fetcher = DataFetcher()
        self.feature_engineer = FeatureEngineer()
        
        if model_path:
            self.load_model(model_path)
        
        logger.info("ModelInference initialized")
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to saved model file
        """
        logger.info("Loading model from {}", model_path)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.scaler = model_data.get('scaler')
        self.feature_columns = model_data['feature_columns']
        self.model_name = model_data['model_name']
        
        logger.info("Loaded {} model with {} features", self.model_name, len(self.feature_columns))
    
    def predict_single_stock(
        self, 
        ticker: str, 
        prediction_date: datetime,
        lookback_days: int = 252
    ) -> Dict[str, Any]:
        """
        Make prediction for a single stock.
        
        Args:
            ticker: Stock ticker symbol
            prediction_date: Date for which to make prediction
            lookback_days: Number of days to look back for features
            
        Returns:
            Dictionary containing prediction results
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        logger.info("Making prediction for {} on {}", ticker, prediction_date)
        
        # Ensure prediction_date is timezone-aware for consistent comparisons
        prediction_date = prediction_date.astimezone(MARKET_TIMEZONE)
        
        # Calculate date range for data fetch
        start_date = prediction_date - timedelta(days=lookback_days + 30)  # Extra buffer
        end_date = prediction_date
        
        try:
            # Fetch stock data
            stock_data = self.data_fetcher.fetch_stock_data(ticker, start_date, end_date)
            
            # Get data up to prediction date
            stock_data = stock_data[stock_data.index <= prediction_date]
            
            if len(stock_data) < 50:  # Minimum data requirement
                raise ValueError(f"Insufficient data for {ticker}: {len(stock_data)} days")
            
            # Fetch benchmark data for relative features (same approach as training)
            try:
                benchmark_data = self.data_fetcher.fetch_stock_data("^GSPC", start_date, end_date)
                benchmark_data = benchmark_data[benchmark_data.index <= prediction_date]
                
                # Use the same feature engineering approach as training
                features = self.feature_engineer.engineer_features_with_benchmark(stock_data, benchmark_data)
                logger.debug("Generated {} features using benchmark approach", len(features.columns))
                
            except Exception as benchmark_error:
                logger.warning("Failed to fetch benchmark data, falling back to standard features: {}", str(benchmark_error))
                # Fallback to standard features if benchmark fails
                features = self.feature_engineer.engineer_features(stock_data)
                logger.debug("Generated {} features using standard approach", len(features.columns))
            
            # Get the most recent feature vector
            latest_features = features.iloc[-1:].copy()
            
            # Prepare features for prediction
            prediction_features = self._prepare_features_for_prediction(latest_features)
            
            # Make prediction
            prediction_proba = self.model.predict_proba(prediction_features)[0]
            prediction = self.model.predict(prediction_features)[0]
            
            # Get confidence score
            confidence = max(prediction_proba)
            outperform_probability = prediction_proba[1]  # Probability of outperformance
            
            result = {
                'ticker': ticker,
                'prediction_date': prediction_date,
                'prediction': bool(prediction),
                'outperform_probability': float(outperform_probability),
                'confidence': float(confidence),
                'model_name': self.model_name,
                'features_used': len(self.feature_columns),
                'last_price': float(stock_data['Close'].iloc[-1]),
                'prediction_horizon_days': PREDICTION_HORIZON_DAYS
            }
            
            logger.info("Prediction for {}: {} (confidence: {:.2%})", 
                       ticker, prediction, confidence)
            
            return result
            
        except Exception as e:
            logger.error("Failed to make prediction for {}: {}", ticker, str(e))
            raise
    
    def predict_multiple_stocks(
        self, 
        tickers: List[str], 
        prediction_date: datetime,
        lookback_days: int = 252
    ) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple stocks.
        
        Args:
            tickers: List of stock ticker symbols
            prediction_date: Date for which to make predictions
            lookback_days: Number of days to look back for features
            
        Returns:
            List of prediction results
        """
        logger.info("Making predictions for {} stocks on {}", len(tickers), prediction_date)
        
        results = []
        failed_tickers = []
        
        for ticker in tickers:
            try:
                result = self.predict_single_stock(ticker, prediction_date, lookback_days)
                results.append(result)
            except Exception as e:
                logger.warning("Failed to predict for {}: {}", ticker, str(e))
                failed_tickers.append(ticker)
        
        logger.info("Completed predictions for {} out of {} stocks", 
                   len(results), len(tickers))
        
        if failed_tickers:
            logger.warning("Failed predictions for: {}", failed_tickers)
        
        return results
    
    def get_top_predictions(
        self, 
        tickers: List[str], 
        prediction_date: datetime,
        top_n: int = 10,
        min_confidence: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Get top N predictions sorted by confidence.
        
        Args:
            tickers: List of stock ticker symbols
            prediction_date: Date for which to make predictions
            top_n: Number of top predictions to return
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of top predictions
        """
        logger.info("Getting top {} predictions with min_confidence={:.2%}", top_n, min_confidence)
        
        # Get all predictions
        all_predictions = self.predict_multiple_stocks(tickers, prediction_date)
        
        # Filter by confidence and positive predictions
        filtered_predictions = [
            pred for pred in all_predictions 
            if pred['prediction'] and pred['confidence'] >= min_confidence
        ]
        
        # Sort by outperform probability
        sorted_predictions = sorted(
            filtered_predictions, 
            key=lambda x: x['outperform_probability'], 
            reverse=True
        )
        
        # Return top N
        top_predictions = sorted_predictions[:top_n]
        
        logger.info("Found {} qualifying predictions, returning top {}", 
                   len(sorted_predictions), len(top_predictions))
        
        return top_predictions
    
    def _prepare_features_for_prediction(self, features: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for model prediction using the same approach as training.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Prepared feature array
        """
        # Check which features are available vs required
        available_features = set(features.columns)
        required_features = set(self.feature_columns)
        missing_features = required_features - available_features
        extra_features = available_features - required_features
        
        if missing_features:
            logger.warning("Missing {} features for prediction: {}", 
                         len(missing_features), list(missing_features)[:5])
        
        if extra_features:
            logger.debug("Extra {} features available (will be ignored): {}", 
                        len(extra_features), list(extra_features)[:5])
        
        # Select only the features used in training, in the same order
        try:
            selected_features = features[self.feature_columns].copy()
        except KeyError as e:
            # If some features are missing, create them with neutral values
            logger.warning("Creating missing features with neutral values: {}", str(e))
            selected_features = pd.DataFrame(index=features.index, columns=self.feature_columns)
            
            # Fill available features
            for feature in self.feature_columns:
                if feature in available_features:
                    selected_features[feature] = features[feature]
                else:
                    # Fill missing features with 0 (neutral value)
                    selected_features[feature] = 0.0
        
        # Handle missing values using the same approach as training
        # First try median of available data, then fall back to 0
        for col in selected_features.columns:
            if selected_features[col].isnull().any():
                median_val = selected_features[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                selected_features[col] = selected_features[col].fillna(median_val)
        
        # Handle infinite values (same as training)
        selected_features = selected_features.replace([np.inf, -np.inf], np.nan)
        selected_features = selected_features.fillna(0)
        
        # Ensure all values are numeric
        selected_features = selected_features.astype(float)
        
        # Scale features if scaler is available (same as training)
        if self.scaler is not None:
            features_array = self.scaler.transform(selected_features)
        else:
            features_array = selected_features.values
        
        logger.debug("Prepared {} features for prediction", features_array.shape[1])
        return features_array
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from the loaded model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("No model loaded")
        
        if not hasattr(self.model, 'feature_importances_') and not hasattr(self.model, 'coef_'):
            raise ValueError("Model does not support feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            importances = np.abs(self.model.coef_[0])
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def validate_prediction(
        self, 
        ticker: str,
        prediction_date: datetime,
        actual_outcome: bool = None
    ) -> Dict[str, Any]:
        """
        Validate a prediction against actual outcome.
        
        Args:
            ticker: Stock ticker symbol
            prediction_date: Date of prediction
            actual_outcome: Actual outcome (True if outperformed)
            
        Returns:
            Validation results
        """
        logger.info("Validating prediction for {} on {}", ticker, prediction_date)
        
        # Get the original prediction
        original_prediction = self.predict_single_stock(ticker, prediction_date)
        
        # If actual outcome is provided, calculate accuracy
        if actual_outcome is not None:
            correct = original_prediction['prediction'] == actual_outcome
            
            return {
                'ticker': ticker,
                'prediction_date': prediction_date,
                'predicted': original_prediction['prediction'],
                'actual': actual_outcome,
                'correct': correct,
                'confidence': original_prediction['confidence'],
                'outperform_probability': original_prediction['outperform_probability']
            }
        
        return original_prediction
    
    def list_saved_models(self) -> pd.DataFrame:
        """
        List all saved models and their performance metrics.
        
        Returns:
            DataFrame with model information and metrics
        """
        if not os.path.exists(MODELS_DIR):
            logger.warning("Models directory does not exist: {}", MODELS_DIR)
            return pd.DataFrame()
        
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
        
        if not model_files:
            logger.info("No saved models found in {}", MODELS_DIR)
            return pd.DataFrame()
        
        models_info = []
        
        for model_file in model_files:
            try:
                model_path = os.path.join(MODELS_DIR, model_file)
                model_data = joblib.load(model_path)
                
                # Extract basic info
                info = {
                    'file_name': model_file,
                    'model_name': model_data.get('model_name', 'unknown'),
                    'timestamp': model_data.get('timestamp', 'unknown'),
                    'feature_count': len(model_data.get('feature_columns', [])),
                    'file_size_mb': round(os.path.getsize(model_path) / (1024 * 1024), 2)
                }
                
                # Extract performance metrics
                metadata = model_data.get('metadata', {})
                
                # Check for model_scores (validation metrics) or test_scores
                scores = metadata.get('model_scores') or metadata.get('test_scores', {})
                
                if scores:
                    info.update({
                        'auc': scores.get('auc', 0.0),
                        'accuracy': scores.get('accuracy', 0.0),
                        'precision': scores.get('precision', 0.0),
                        'recall': scores.get('recall', 0.0),
                        'f1_score': scores.get('f1_score', 0.0)
                    })
                else:
                    # No scores available
                    info.update({
                        'auc': 0.0,
                        'accuracy': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1_score': 0.0
                    })
                
                # Additional metadata
                info.update({
                    'training_samples': metadata.get('training_samples', 0),
                    'training_date': metadata.get('training_date', 'unknown'),
                    'hyperparameter_tuning': metadata.get('hyperparameter_tuning', False),
                    'is_best_model': metadata.get('is_best_model', False)
                })
                
                models_info.append(info)
                
            except Exception as e:
                logger.warning("Failed to load model {}: {}", model_file, str(e))
                continue
        
        if not models_info:
            logger.info("No valid models found")
            return pd.DataFrame()
        
        # Create DataFrame and sort by AUC (best first)
        df = pd.DataFrame(models_info)
        df = df.sort_values('auc', ascending=False)
        
        logger.info("Found {} valid models", len(df))
        return df
    
    def test_feature_columns_consistency(self) -> Dict[str, Any]:
        """
        Test that the loaded feature columns are consistent and valid.
        This method can be used to verify model integrity.
        
        Returns:
            Dictionary with test results
        """
        results = {
            'feature_columns_present': False,
            'feature_columns_valid': False,
            'feature_count': 0,
            'feature_columns_type': None,
            'sample_features': [],
            'issues': []
        }
        
        # Check if feature columns exist
        if self.feature_columns is None:
            results['issues'].append("Feature columns is None")
            return results
        
        results['feature_columns_present'] = True
        results['feature_columns_type'] = type(self.feature_columns).__name__
        
        # Check if it's a valid list
        if not isinstance(self.feature_columns, list):
            results['issues'].append(f"Feature columns should be a list, got {type(self.feature_columns)}")
            return results
        
        # Check if it's not empty
        if len(self.feature_columns) == 0:
            results['issues'].append("Feature columns list is empty")
            return results
        
        results['feature_count'] = len(self.feature_columns)
        results['sample_features'] = self.feature_columns[:5]  # First 5 features as sample
        
        # Check for duplicate features
        if len(set(self.feature_columns)) != len(self.feature_columns):
            duplicates = [item for item in set(self.feature_columns) 
                         if self.feature_columns.count(item) > 1]
            results['issues'].append(f"Duplicate features found: {duplicates}")
        
        # Check for invalid feature names
        invalid_features = []
        for feature in self.feature_columns:
            if not isinstance(feature, str):
                invalid_features.append(f"{feature} (type: {type(feature)})")
            elif feature.strip() == '':
                invalid_features.append("empty string feature")
        
        if invalid_features:
            results['issues'].append(f"Invalid feature names: {invalid_features}")
        
        # Check if feature columns seem reasonable (basic heuristics)
        suspicious_features = [f for f in self.feature_columns if f in ['ticker', 'date', 'target', 'label']]
        if suspicious_features:
            results['issues'].append(f"Suspicious feature names found (should be excluded): {suspicious_features}")
        
        # If no issues found, mark as valid
        if not results['issues']:
            results['feature_columns_valid'] = True
        
        return results


def get_latest_model_path() -> str:
    """
    Get the path to the most recently saved model.
    
    Returns:
        Path to the latest model file
        
    Raises:
        FileNotFoundError: If no model files are found
    """
    import glob
    
    # Look for model files in the models directory
    model_patterns = [
        os.path.join(MODELS_DIR, "*.pkl"),
        os.path.join(MODELS_DIR, "saved", "*.pkl")
    ]
    
    model_files = []
    for pattern in model_patterns:
        model_files.extend(glob.glob(pattern))
    
    if not model_files:
        raise FileNotFoundError(f"No model files found in {MODELS_DIR}")
    
    # Return the most recently modified model
    latest_model = max(model_files, key=os.path.getmtime)
    logger.info("Found latest model: {}", latest_model)
    
    return latest_model


if __name__ == "__main__":
    # Test the model inference
    logger.info("Testing ModelInference...")
    
    # Note: This test requires a trained model file
    try:
        # Try to load the latest model
        inference = ModelInference()
        inference.list_saved_models()
        
        # Test prediction
        prediction_date = datetime.now() - timedelta(days=1)
        result = inference.predict_single_stock("AAPL", prediction_date)
        
        logger.info("Test prediction result: {}", result)
        
        # Test feature importance
        importance = inference.get_feature_importance()
        logger.info("Top 5 features: {}", importance.head())
        
        # Test feature columns consistency
        consistency = inference.test_feature_columns_consistency()
        logger.info("Feature columns consistency: {}", consistency)
        
    except FileNotFoundError as e:
        logger.warning("Model inference test skipped: {}", str(e))
    except Exception as e:
        logger.error("Model inference test failed: {}", str(e))
    
    logger.info("Model inference test completed")
