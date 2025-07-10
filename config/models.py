"""
Model configuration for FINQ Stock Predictor.
Contains all model selection and parameter settings.
"""

# Model Selection Configuration
MODEL_SELECTION = {
    # Fast & Lightweight Models (Recommended for quick testing)
    "ridge": True,           # Ridge Classifier - Very fast, good baseline
    "elasticnet": True,      # ElasticNet - Fast with feature selection
    "extra_trees": True,     # Extra Trees - Faster than Random Forest
    
    # Tree-based Models (Good performance)
    "random_forest": True,   # Random Forest - Current default, solid performance
    "lightgbm": True,        # LightGBM - Fast gradient boosting, excellent performance
    "xgboost": True,         # XGBoost - Industry standard, great performance
    "catboost": False,       # CatBoost - Good with categorical features (slower)
    
    # Linear Models (Fast)
    "logistic_regression": True,  # Logistic Regression - Current default, interpretable
    
    # Heavy Models (Slow - use for final comparison only)
    "gradient_boosting": False,   # Gradient Boosting - Slower than LightGBM/XGBoost
    "svm": False,                 # SVM - Very slow, use only for small datasets
    "neural_network": False,      # Neural Network - Requires more tuning
}

# Model Parameters Configuration
MODEL_CONFIGURATIONS = {
    # Fast Linear Models
    "ridge": {
        "alpha": 1.0,
        "solver": "auto",
        "random_state": 42
    },
    "elasticnet": {
        "alpha": 0.0001,
        "l1_ratio": 0.5,
        "loss": "log_loss",
        "penalty": "elasticnet",
        "random_state": 42,
        "max_iter": 1000
    },
    "logistic_regression": {
        "random_state": 42,
        "max_iter": 1000,
        "solver": "lbfgs"
    },
    
    # Tree-based Models
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42
    },
    "extra_trees": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42
    },
    
    # Gradient Boosting Models
    "lightgbm": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1
    },
    "xgboost": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "logloss"
    },
    "catboost": {
        "iterations": 100,
        "learning_rate": 0.1,
        "depth": 6,
        "random_state": 42,
        "verbose": False
    },
    
    # Heavy Models (for final comparison)
    "gradient_boosting": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42
    },
    "svm": {
        "kernel": "rbf",
        "probability": True,
        "random_state": 42,
        "C": 1.0,
        "gamma": "scale"
    },
    "neural_network": {
        "hidden_layer_sizes": (100, 50),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,
        "learning_rate": "constant",
        "max_iter": 200,
        "random_state": 42
    }
}

# Model Training Strategies
MODEL_STRATEGIES = {
    # Quick Testing Strategy (2-3 minutes)
    "quick_test": {
        "ridge": True,
        "elasticnet": True,
        "extra_trees": True,
        "logistic_regression": True,
        "random_forest": False,
        "lightgbm": False,
        "xgboost": False,
        "catboost": False,
        "gradient_boosting": False,
        "svm": False,
        "neural_network": False
    },
    
    # Balanced Strategy (5-10 minutes)
    "balanced": {
        "ridge": True,
        "elasticnet": True,
        "extra_trees": True,
        "logistic_regression": True,
        "random_forest": True,
        "lightgbm": True,
        "xgboost": False,
        "catboost": False,
        "gradient_boosting": False,
        "svm": False,
        "neural_network": False
    },
    
    # Comprehensive Strategy (15-30 minutes)
    "comprehensive": {
        "ridge": True,
        "elasticnet": True,
        "extra_trees": True,
        "logistic_regression": True,
        "random_forest": True,
        "lightgbm": True,
        "xgboost": True,
        "catboost": True,
        "gradient_boosting": False,
        "svm": False,
        "neural_network": False
    },
    
    # Full Strategy (30+ minutes)
    "full": {
        "ridge": True,
        "elasticnet": True,
        "extra_trees": True,
        "logistic_regression": True,
        "random_forest": True,
        "lightgbm": True,
        "xgboost": True,
        "catboost": True,
        "gradient_boosting": True,
        "svm": True,
        "neural_network": True
    }
}
