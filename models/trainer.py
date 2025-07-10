"""
Model training module for FINQ Stock Predictor.
Handles model training, validation, and evaluation.
"""
import joblib
from datetime import datetime
import pandas as pd
import numpy as np
import asyncio
import concurrent.futures
import os
from functools import partial
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any, Optional
from loguru import logger
from main_config import MODELS_DIR

# Try to import optional dependencies
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not available. Install with: pip install catboost")

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import main configurations
from main_config import MODEL_PARAMS

# Import specialized model configurations
from config.models import MODEL_SELECTION, MODEL_CONFIGURATIONS, MODEL_STRATEGIES


class ModelTrainer:
    """
    Handles model training and evaluation for stock prediction.
    """
    
    def __init__(self, random_state: int = MODEL_PARAMS['random_state']):
        """
        Initialize ModelTrainer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        logger.info("ModelTrainer initialized with random_state={}", self.random_state)
    
    def _prepare_features(
        self, 
        X: pd.DataFrame, 
        exclude_columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Prepare features for training.
        
        Args:
            X: Feature DataFrame
            exclude_columns: Columns to exclude from features
            
        Returns:
            Prepared feature DataFrame
        """
        if exclude_columns is None:
            exclude_columns = ['ticker', 'date']
        
        # Exclude non-feature columns
        feature_columns = [col for col in X.columns if col not in exclude_columns]
        X_prepared = X[feature_columns].copy()
        
        # Handle infinite values
        X_prepared = X_prepared.replace([np.inf, -np.inf], np.nan)
        
        # Check for NaN values
        nan_count = X_prepared.isnull().sum().sum()
        if nan_count > 0:
            logger.warning("Found {} NaN values in features, using median imputation", nan_count)
            # Use median imputation for any remaining NaNs
            X_prepared = X_prepared.fillna(X_prepared.median())
        
        # Store feature columns for later use
        self.feature_columns = feature_columns
        
        logger.debug("Prepared {} features for training", len(feature_columns))
        return X_prepared
    
    def _get_model_configurations(
        self, 
        async_mode: bool = False, 
        strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get model configurations for training.
        
        Args:
            async_mode: Whether models will be trained asynchronously
            strategy: Strategy name from MODEL_STRATEGIES (e.g., 'quick_test', 'balanced', 'comprehensive', 'full')
                     If None, uses MODEL_SELECTION configuration
            
        Returns:
            Dictionary of model configurations
        """
        # Determine which models to use
        if strategy is not None:
            if strategy not in MODEL_STRATEGIES:
                raise ValueError(f"Unknown strategy: {strategy}. Available strategies: {list(MODEL_STRATEGIES.keys())}")
            model_selection = MODEL_STRATEGIES[strategy]
            logger.info(f"Using strategy '{strategy}' for model selection")
        else:
            model_selection = MODEL_SELECTION
            logger.info("Using MODEL_SELECTION configuration")
        
        # Build model configurations
        models = {}
        
        # Ridge Classifier
        if model_selection.get('ridge', False):
            models['ridge'] = RidgeClassifier(**MODEL_CONFIGURATIONS['ridge'])
        
        # ElasticNet (using SGDClassifier with elasticnet penalty)
        if model_selection.get('elasticnet', False):
            models['elasticnet'] = SGDClassifier(**MODEL_CONFIGURATIONS['elasticnet'])
        
        # Logistic Regression
        if model_selection.get('logistic_regression', False):
            models['logistic_regression'] = LogisticRegression(**MODEL_CONFIGURATIONS['logistic_regression'])
        
        # Random Forest
        if model_selection.get('random_forest', False):
            config = MODEL_CONFIGURATIONS['random_forest'].copy()
            config['n_jobs'] = 1 if async_mode else -1  # Use single job per model for async
            models['random_forest'] = RandomForestClassifier(**config)
        
        # Extra Trees
        if model_selection.get('extra_trees', False):
            config = MODEL_CONFIGURATIONS['extra_trees'].copy()
            config['n_jobs'] = 1 if async_mode else -1  # Use single job per model for async
            models['extra_trees'] = ExtraTreesClassifier(**config)
        
        # Gradient Boosting
        if model_selection.get('gradient_boosting', False):
            models['gradient_boosting'] = GradientBoostingClassifier(**MODEL_CONFIGURATIONS['gradient_boosting'])
        
        # LightGBM
        if model_selection.get('lightgbm', False):
            if LIGHTGBM_AVAILABLE:
                models['lightgbm'] = lgb.LGBMClassifier(**MODEL_CONFIGURATIONS['lightgbm'])
            else:
                logger.warning("LightGBM requested but not available. Skipping.")
        
        # XGBoost
        if model_selection.get('xgboost', False):
            if XGBOOST_AVAILABLE:
                models['xgboost'] = xgb.XGBClassifier(**MODEL_CONFIGURATIONS['xgboost'])
            else:
                logger.warning("XGBoost requested but not available. Skipping.")
        
        # CatBoost
        if model_selection.get('catboost', False):
            if CATBOOST_AVAILABLE:
                models['catboost'] = cb.CatBoostClassifier(**MODEL_CONFIGURATIONS['catboost'])
            else:
                logger.warning("CatBoost requested but not available. Skipping.")
        
        # SVM
        if model_selection.get('svm', False):
            models['svm'] = SVC(**MODEL_CONFIGURATIONS['svm'])
        
        # Neural Network
        if model_selection.get('neural_network', False):
            models['neural_network'] = MLPClassifier(**MODEL_CONFIGURATIONS['neural_network'])
        
        logger.info(f"Selected {len(models)} models for training: {list(models.keys())}")
        return models

    def _prepare_training_data(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
        """
        Prepare training data for model training.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Tuple of (prepared_features, scaled_features, target)
        """
        # Prepare features
        X_train_prepared = self._prepare_features(X_train)
        y_train_target = y_train['outperforms']
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_prepared)
        self.scalers['main'] = scaler
        
        return X_train_prepared, X_train_scaled, y_train_target
    
    def train_models(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.DataFrame,
        X_val: pd.DataFrame = None,
        y_val: pd.DataFrame = None,
        strategy: Optional[str] = None
    ):
        """
        Train multiple models sequentially.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            strategy: Strategy name from MODEL_STRATEGIES (e.g., 'quick_test', 'balanced', 'comprehensive', 'full')
                     If None, uses MODEL_SELECTION configuration
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Training models with {} training samples", len(X_train))
        
        # Prepare training data using unified method
        X_train_prepared, X_train_scaled, y_train_target = self._prepare_training_data(
            X_train, y_train
        )
        
        # Get model configurations
        models_to_train = self._get_model_configurations(async_mode=False, strategy=strategy)
        
        # Train each model sequentially
        trained_models = {}
        for name, model in models_to_train.items():
            result = self._train_single_model(
                name, model, X_train_prepared, X_train_scaled, y_train_target, X_val, y_val
            )
            
            model_name, trained_model, val_scores = result
            if trained_model is not None:
                trained_models[model_name] = trained_model
        
        self.models = trained_models
        logger.info("Trained {} models successfully", len(trained_models))
        
    def evaluate_model(
        self, 
        model: Any, 
        X_test: pd.DataFrame, 
        y_test: pd.DataFrame,
        model_name: str = "model"
    ) -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model for logging
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating {} model", model_name)
        
        # Prepare features
        X_test_prepared = self._prepare_features(X_test)
        y_test_target = y_test['outperforms']
        
        # Scale features if needed
        if model_name in ['logistic_regression', 'svm', 'ridge', 'elasticnet']:
            X_test_scaled = self.scalers['main'].transform(X_test_prepared)
            y_pred = model.predict(X_test_scaled)
            # Check if model has predict_proba method
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                # For models without predict_proba, use decision_function or predict
                if hasattr(model, 'decision_function'):
                    y_pred_proba = model.decision_function(X_test_scaled)
                else:
                    y_pred_proba = y_pred.astype(float)
        else:
            y_pred = model.predict(X_test_prepared)
            # Check if model has predict_proba method
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_prepared)[:, 1]
            else:
                # For models without predict_proba, use decision_function or predict
                if hasattr(model, 'decision_function'):
                    y_pred_proba = model.decision_function(X_test_prepared)
                else:
                    y_pred_proba = y_pred.astype(float)
        
        # Calculate metrics
        auc = roc_auc_score(y_test_target, y_pred_proba)
        accuracy = (y_pred == y_test_target).mean()
        
        # Calculate precision and recall
        precision, recall, _ = precision_recall_curve(y_test_target, y_pred_proba)
        avg_precision = np.mean(precision)
        
        # Log results
        logger.info("{} Evaluation Results:", model_name)
        logger.info("  AUC: {:.4f}", auc)
        logger.info("  Accuracy: {:.4f}", accuracy)
        logger.info("  Avg Precision: {:.4f}", avg_precision)
        
        # Print classification report
        report = classification_report(y_test_target, y_pred, output_dict=True)
        logger.info("  Precision (Class 1): {:.4f}", report['1']['precision'])
        logger.info("  Recall (Class 1): {:.4f}", report['1']['recall'])
        logger.info("  F1-Score (Class 1): {:.4f}", report['1']['f1-score'])
        
        return {
            'auc': auc,
            'accuracy': accuracy,
            'avg_precision': avg_precision,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score']
        }
    
    def select_best_model(
        self, 
        X_val: pd.DataFrame, 
        y_val: pd.DataFrame,
        metric: str = 'auc'
    ) -> Tuple[str, Any]:
        """
        Select the best model based on validation performance.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            metric: Metric to use for selection
            
        Returns:
            Tuple of (best_model_name, best_model)
        """
        if not self.models:
            raise ValueError("No models have been trained yet")
        
        logger.info("Selecting best model based on {} metric", metric)
        
        best_score = -np.inf
        best_model_name = None
        best_model = None
        
        for model_name, model in self.models.items():
            try:
                scores = self.evaluate_model(model, X_val, y_val, model_name)
                score = scores[metric]
                
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
                    best_model = model
                    
            except Exception as e:
                logger.warning("Failed to evaluate {}: {}", model_name, str(e))
                continue
        
        logger.info("Best model: {} with {} = {:.4f}", best_model_name, metric, best_score)
        return best_model_name, best_model
    
    def tune_hyperparameters(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.DataFrame,
        model_type: str = 'random_forest'
    ) -> Any:
        """
        Tune hyperparameters for a specific model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_type: Type of model to tune
            
        Returns:
            Best model with tuned hyperparameters
        """
        logger.info("Tuning hyperparameters for {} model", model_type)
        
        # Prepare features
        X_train_prepared = self._prepare_features(X_train)
        y_train_target = y_train['outperforms']
        
        # Define parameter grids
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        if model_type not in param_grids:
            raise ValueError(f"Hyperparameter tuning not supported for {model_type}")
        
        # Create base model
        base_models = {
            'random_forest': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            'gradient_boosting': GradientBoostingClassifier(random_state=self.random_state),
            'logistic_regression': LogisticRegression(random_state=self.random_state, max_iter=1000)
        }
        
        base_model = base_models[model_type]
        param_grid = param_grids[model_type]
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        # Use scaled features for linear models
        if model_type == 'logistic_regression':
            X_train_scaled = self.scalers['main'].transform(X_train_prepared)
            grid_search.fit(X_train_scaled, y_train_target)
        else:
            grid_search.fit(X_train_prepared, y_train_target)
        
        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_
        
        logger.info("Best hyperparameters for {}: {}", model_type, grid_search.best_params_)
        logger.info("Best cross-validation score: {:.4f}", best_score)
        
        return best_model
    
    def get_feature_importance(
        self, 
        model: Any, 
        model_name: str,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importance from a trained model.
        
        Args:
            model: Trained model
            model_name: Name of the model
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        logger.info("Getting feature importance for {} model", model_name)
        
        if not hasattr(model, 'feature_importances_') and not hasattr(model, 'coef_'):
            logger.warning("Model {} does not support feature importance", model_name)
            return pd.DataFrame()
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
        else:
            # Linear models
            importances = np.abs(model.coef_[0])
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Return top N features
        top_features = importance_df.head(top_n)
        
        logger.info("Top {} features for {}:", top_n, model_name)
        for _, row in top_features.iterrows():
            logger.info("  {}: {:.4f}", row['feature'], row['importance'])
        
        return top_features

    def _train_single_model(
        self, 
        name: str, 
        model, 
        X_train_prepared: np.ndarray,
        X_train_scaled: np.ndarray,
        y_train_target: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.DataFrame = None
    ) -> Tuple[str, Any, Dict]:
        """
        Train a single model.
        
        Args:
            name: Model name
            model: Model instance
            X_train_prepared: Prepared training features
            X_train_scaled: Scaled training features
            y_train_target: Training target
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Tuple of (model_name, trained_model, validation_scores)
        """
        logger.info("Training {} model", name)
        try:
            if name in ['logistic_regression', 'svm', 'ridge', 'elasticnet']:
                # Use scaled features for linear models
                model.fit(X_train_scaled, y_train_target)
            else:
                # Use original features for tree-based models
                model.fit(X_train_prepared, y_train_target)
            
            logger.info("Successfully trained {} model", name)
            
            # Evaluate on validation set if provided
            val_scores = {}
            if X_val is not None and y_val is not None:
                val_scores = self.evaluate_model(model, X_val, y_val, model_name=name)
                logger.info("{} validation AUC: {:.4f}", name, val_scores['auc'])
            
            return name, model, val_scores
            
        except Exception as e:
            logger.error("Failed to train {} model: {}", name, str(e))
            return name, None, {}

    async def train_models_async(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.DataFrame,
        X_val: pd.DataFrame = None,
        y_val: pd.DataFrame = None,
        max_workers: int = None,
        strategy: Optional[str] = None
    ):
        """
        Train multiple models asynchronously in parallel.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            max_workers: Maximum number of worker threads (default: number of CPU cores)
            strategy: Strategy name from MODEL_STRATEGIES (e.g., 'quick_test', 'balanced', 'comprehensive', 'full')
                     If None, uses MODEL_SELECTION configuration
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Training models asynchronously with {} training samples", len(X_train))
        
        # Prepare training data using unified method
        X_train_prepared, X_train_scaled, y_train_target = self._prepare_training_data(
            X_train, y_train
        )
        
        # Get model configurations for async training
        models_to_train = self._get_model_configurations(async_mode=True, strategy=strategy)
        
        # Create tasks for async training
        loop = asyncio.get_event_loop()
        
        # Use ThreadPoolExecutor for CPU-bound tasks
        max_workers = max_workers or min(len(models_to_train), os.cpu_count() or 4)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create tasks for each model
            tasks = []
            for name, model in models_to_train.items():
                # Create a partial function with all arguments
                train_func = partial(
                    self._train_single_model,
                    name,
                    model,
                    X_train_prepared,
                    X_train_scaled,
                    y_train_target,
                    X_val,
                    y_val
                )
                
                # Submit to thread pool
                task = loop.run_in_executor(executor, train_func)
                tasks.append(task)
            
            logger.info("Starting parallel training of {} models using {} workers", 
                       len(models_to_train), max_workers)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results using the same logic as sync training
        trained_models = {}
        
        for result in results:
            if isinstance(result, Exception):
                logger.error("Model training task failed: {}", str(result))
                continue
            
            model_name, trained_model, val_scores = result
            if trained_model is not None:
                trained_models[model_name] = trained_model
            else:
                logger.warning("Model {} training failed", model_name)
        
        self.models = trained_models
        logger.info("Successfully trained {} out of {} models in parallel", 
                   len(trained_models), len(models_to_train))
        
    def save_model(
        self, 
        model: Any, 
        model_name: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model
            model_name: Name of the model
            metadata: Additional metadata to save
            
        Returns:
            Path to saved model
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Include stocks_count in filename if available in metadata
        stocks_info = ""
        if metadata and 'stocks_count' in metadata:
            stocks_info = f"_{metadata['stocks_count']}stocks"
        
        model_filename = f"{model_name}{stocks_info}_{timestamp}.pkl"
        model_path = os.path.join(MODELS_DIR, model_filename)
        
        # Prepare model data
        model_data = {
            'model': model,
            'scaler': self.scalers.get('main'),
            'feature_columns': self.feature_columns,
            'model_name': model_name,
            'timestamp': timestamp,
            'metadata': metadata or {}
        }
        
        # Save model
        joblib.dump(model_data, model_path)
        logger.info("Saved {} model to {}", model_name, model_path)
        
        return model_path
    
    def save_all_models(
        self,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
        base_metadata: Dict[str, Any] = None
    ) -> List[str]:
        """
        Save all trained models with their individual performance metrics.
        
        Args:
            X_val: Validation features for evaluation
            y_val: Validation labels for evaluation
            base_metadata: Base metadata to include in all models
            
        Returns:
            List of paths to saved models
        """
        if not self.models:
            raise ValueError("No models have been trained yet")
        
        logger.info("Saving all {} trained models", len(self.models))
        
        saved_paths = []
        
        for model_name, model in self.models.items():
            try:
                # Evaluate this specific model
                logger.info("Evaluating {} model for saving", model_name)
                model_scores = self.evaluate_model(model, X_val, y_val, model_name)
                
                # Create metadata for this model
                model_metadata = (base_metadata or {}).copy()
                model_metadata['model_scores'] = model_scores
                model_metadata['model_type'] = model_name
                
                # Save the model
                model_path = self.save_model(model, model_name, model_metadata)
                saved_paths.append(model_path)
                
                logger.info("Saved {} model with AUC: {:.4f}", model_name, model_scores['auc'])
                
            except Exception as e:
                logger.error("Failed to save {} model: {}", model_name, str(e))
                continue
        
        logger.info("Successfully saved {}/{} models", len(saved_paths), len(self.models))
        return saved_paths
    
    def list_available_strategies(self) -> Dict[str, Dict[str, bool]]:
        """
        List all available training strategies.
        
        Returns:
            Dictionary of strategy names and their model configurations
        """
        return MODEL_STRATEGIES.copy()
    
    def get_current_model_selection(self) -> Dict[str, bool]:
        """
        Get the current model selection configuration.
        
        Returns:
            Dictionary of model names and their selection status
        """
        return MODEL_SELECTION.copy()
    
    def get_strategy_models(self, strategy: str) -> List[str]:
        """
        Get the list of models that will be trained for a given strategy.
        
        Args:
            strategy: Strategy name
            
        Returns:
            List of model names
        """
        if strategy not in MODEL_STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Available strategies: {list(MODEL_STRATEGIES.keys())}")
        
        strategy_config = MODEL_STRATEGIES[strategy]
        return [model for model, enabled in strategy_config.items() if enabled]
    
    def compare_strategies(self) -> pd.DataFrame:
        """
        Compare different strategies and show which models they use.
        
        Returns:
            DataFrame comparing strategies
        """
        strategies_df = pd.DataFrame(MODEL_STRATEGIES).T
        strategies_df = strategies_df.fillna(False)
        
        # Add model counts
        strategies_df['total_models'] = strategies_df.sum(axis=1)
        
        return strategies_df
