"""
Main training script for FINQ Stock Predictor.
Orchestrates the entire ML pipeline from data fetching to model training.
"""

import argparse
import os
import sys
import pandas as pd
import asyncio
import numpy as np
import random
from datetime import datetime
from loguru import logger

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import main configurations from main_config.py
from main_config import LOG_CONFIG, STOCKS_TICKERS, ASYNC_CONFIG

# Import specialized configurations
from config.features import FEATURE_SELECTION, FEATURE_PARAMS
from config.models import MODEL_SELECTION, MODEL_CONFIGURATIONS, MODEL_STRATEGIES

# Import other modules
from data.fetcher import DataFetcher, get_sp500_data, get_sp500_data_async
from data.processor import DataProcessor
from features.engineer import FeatureEngineer
from models.trainer import ModelTrainer


def setup_logging():
    """Setup logging configuration."""
    logger.remove()
    logger.add(
        sys.stdout,
        level=LOG_CONFIG['level'],
        format=LOG_CONFIG['format'],
        serialize=LOG_CONFIG['serialize']
    )
    logger.add(
        "logs/training_{time}.log",
        level=LOG_CONFIG['level'],
        format=LOG_CONFIG['format'],
        serialize=LOG_CONFIG['serialize'],
        rotation="10 MB"
    )


async def fetch_stock_data(args):
    """
    Data Fetching - Handle stock data fetching.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of (stock_data, benchmark_data)
    """
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None
    
    # Use async data fetching by default (unless --sync-data-fetch is specified)
    use_async_fetch = not args.sync_data_fetch
    max_fetch_workers = ASYNC_CONFIG['max_fetch_workers']
    
    if use_async_fetch:
        logger.info("Using async data fetching with max_workers={}", max_fetch_workers)
        stock_data, benchmark_data = await get_sp500_data_async(
            start_date=start_date,
            end_date=end_date,
            max_stocks=args.max_stocks,
            max_workers=max_fetch_workers
        )
    else:
        logger.info("Using synchronous data fetching")
        stock_data, benchmark_data = get_sp500_data(
            start_date=start_date,
            end_date=end_date,
            max_stocks=args.max_stocks
        )
    
    logger.info("Fetched data for {} stocks", len(stock_data))
    return stock_data, benchmark_data


def process_and_engineer_features(stock_data, benchmark_data):
    """
    Data Processing and Feature Engineering.
    
    Args:
        stock_data: Stock data dictionary
        benchmark_data: Benchmark data DataFrame
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Data Processing (basic processing and labeling)
    logger.info("Processing data and creating labels")
    processor = DataProcessor()
    basic_features, labels = processor.get_or_create_processed_data(stock_data, benchmark_data)
    
    logger.info("Basic features: {} features, {} samples", 
               len(basic_features.columns), len(basic_features))
    
    # Feature Engineering (advanced feature creation)
    logger.info("Engineering advanced features")
    engineer = FeatureEngineer()
    enhanced_features = engineer.engineer_features_multiple_stocks_with_benchmark(stock_data, benchmark_data)
    
    # Combine enhanced features with basic features
    all_features = []
    for ticker, ticker_features in enhanced_features.items():
        ticker_features['ticker'] = ticker
        ticker_features['date'] = ticker_features.index
        all_features.append(ticker_features)
    
    combined_features = pd.concat(all_features, ignore_index=True)
    
    # Align enhanced features with labels
    merged_data = pd.merge(combined_features, labels, on=['ticker', 'date'], how='inner')
    features = merged_data.drop(['outperforms', 'stock_return', 'benchmark_return', 'excess_return'], axis=1)
    labels = merged_data[['outperforms', 'stock_return', 'benchmark_return', 'excess_return', 'ticker', 'date']]
    
    logger.info("Enhanced features: {} features, {} samples", 
               len(features.columns), len(features))
    
    # Train/Validation/Test Split
    logger.info("Creating time-series splits")
    X_train, X_val, X_test, y_train, y_val, y_test = processor.create_time_series_splits(
        features, labels
    )
    
    logger.info("Data splits - Train: {}, Val: {}, Test: {}", 
               len(X_train), len(X_val), len(X_test))
    
    return X_train, X_val, X_test, y_train, y_val, y_test




def evaluate_and_select_best_model(trainer, X_val, y_val, X_test, y_test):
    """
    Model Selection and Evaluation.
    
    Args:
        trainer: ModelTrainer instance
        X_val, y_val: Validation data
        X_test, y_test: Test data
        
    Returns:
        Tuple of (best_model_name, best_model, test_scores)
    """
    best_model_name, best_model = trainer.select_best_model(X_val, y_val)
    
    # Final evaluation on test set
    logger.info("Final evaluation on test set")
    test_scores = trainer.evaluate_model(best_model, X_test, y_test, best_model_name)
    
    return best_model_name, best_model, test_scores


def analyze_feature_importance(trainer, best_model, best_model_name):
    """
    Feature Importance Analysis.
    
    Args:
        trainer: ModelTrainer instance
        best_model: Best trained model
        best_model_name: Name of the best model
        
    Returns:
        Feature importance DataFrame
    """
    importance = trainer.get_feature_importance(best_model, best_model_name)
    logger.info("Top 10 most important features:")
    for _, row in importance.head(10).iterrows():
        logger.info("  {}: {:.4f}", row['feature'], row['importance'])
    
    return importance


def tune_hyperparameters_and_reevaluate(trainer, X_train, y_train, X_val, y_val, X_test, y_test, args):
    """
    Hyperparameter Tuning and Re-evaluation.
    
    Args:
        trainer: ModelTrainer instance
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        args: Command line arguments
        
    Returns:
        Tuple of (best_model_name, best_model, test_scores) or None if no tuning
    """
    if not args.tune_hyperparameters:
        return None
    
    logger.info("Performing hyperparameter tuning for {}", args.model_type)
    
    # Tune hyperparameters
    best_model = trainer.tune_hyperparameters(X_train, y_train, args.model_type)
    trainer.models[args.model_type] = best_model
    
    # Re-evaluate with tuned model
    best_model_name = args.model_type
    test_scores = trainer.evaluate_model(best_model, X_test, y_test, best_model_name)
    
    logger.info("Tuned model performance - AUC: {:.4f}", test_scores['auc'])
    
    return best_model_name, best_model, test_scores


def save_trained_models(trainer, X_val, y_val, best_model, best_model_name, test_scores, stock_data, args, scale_info=None):
    """
    Save Models.
    
    Args:
        trainer: ModelTrainer instance
        X_val, y_val: Validation data
        best_model: Best trained model
        best_model_name: Name of the best model
        test_scores: Test evaluation scores
        stock_data: Stock data dictionary
        args: Command line arguments
        
    Returns:
        List of saved model paths
    """
    if not (args.save_model or args.save_all_models):
        return []
    
    logger.info("Saving model(s)")
    
    # Base metadata that applies to all models
    base_metadata = {
        'training_samples': len(X_val) * 4,  # Approximate from validation size - validation is 20% of total
        'validation_samples': len(X_val),
        'test_samples': len(X_val),  # Approximate
        'features_count': len(trainer.feature_columns),
        'stocks_count': len(stock_data),
        'training_date': datetime.now().isoformat(),
        'hyperparameter_tuning': args.tune_hyperparameters,
    }
    if scale_info:
        base_metadata.update(scale_info)
    
    saved_paths = []
    
    if args.save_all_models:
        # Save all models with their individual metrics
        logger.info("Saving all trained models")
        saved_paths = trainer.save_all_models(X_val, y_val, base_metadata)
        logger.info("Saved {} models:", len(saved_paths))
        for path in saved_paths:
            logger.info("  - {}", path)
    else:
        # Save only the best model
        logger.info("Saving best model only")
        best_model_metadata = base_metadata.copy()
        best_model_metadata['test_scores'] = test_scores
        best_model_metadata['model_type'] = best_model_name
        best_model_metadata['is_best_model'] = True
        
        model_path = trainer.save_model(best_model, best_model_name, best_model_metadata)
        logger.info("Best model saved to: {}", model_path)
        saved_paths.append(model_path)
    
    return saved_paths


def log_training_summary(best_model_name, test_scores, trainer, stock_data, X_train, X_val, X_test):
    """
    Log Training Summary.
    
    Args:
        best_model_name: Name of the best model
        test_scores: Test evaluation scores
        trainer: ModelTrainer instance
        stock_data: Stock data dictionary
        X_train, X_val, X_test: Data splits
    """
    logger.info("Training Pipeline Summary")
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info("Best model: {} (AUC: {:.4f})", best_model_name, test_scores['auc'])
    logger.info("Test accuracy: {:.4f}", test_scores['accuracy'])
    logger.info("Test precision: {:.4f}", test_scores['precision'])
    logger.info("Test recall: {:.4f}", test_scores['recall'])
    logger.info("Training samples: {}", len(X_train))
    logger.info("Validation samples: {}", len(X_val))
    logger.info("Test samples: {}", len(X_test))
    logger.info("Total features: {}", len(trainer.feature_columns))
    logger.info("Total stocks processed: {}", len(stock_data))
    logger.info("=" * 60)


def handle_post_training_steps(trainer, X_train, y_train, X_val, y_val, X_test, y_test, stock_data, args, scale_info=None):
    """
    Handle Post-training evaluation and processing.
    
    Args:
        trainer: ModelTrainer instance
        X_train, y_train: Training data
        X_val, y_val: Validation data  
        X_test, y_test: Test data
        stock_data: Stock data dictionary
        args: Command line arguments
        scale_info: Optional dict with scale information for multi-scale training
        
    Returns:
        Dict with results including best_model_name, test_scores, etc.
    """
    # Step 1: Best Model Selection and Evaluation
    best_model_name, best_model, test_scores = evaluate_and_select_best_model(
        trainer, X_val, y_val, X_test, y_test
    )
    
    # Step 2: Feature Importance Analysis
    importance = analyze_feature_importance(trainer, best_model, best_model_name)
    
    # Step 3: Hyperparameter Tuning and Re-evaluation
    tuned_results = tune_hyperparameters_and_reevaluate(
        trainer, X_train, y_train, X_val, y_val, X_test, y_test, args
    )
    
    # Use tuned results if available
    if tuned_results is not None:
        best_model_name, best_model, test_scores = tuned_results
    
    # Step 4: Save Models (only for single-scale or if requested for multi-scale)
    saved_paths = []
    if not scale_info or (args.save_model or args.save_all_models):
        saved_paths = save_trained_models(
            trainer, X_val, y_val, best_model, best_model_name, test_scores, stock_data, args, scale_info
        )
    
    # Step 5: Log Training Summary (only for single-scale)
    if not scale_info:
        log_training_summary(best_model_name, test_scores, trainer, stock_data, X_train, X_val, X_test)
    
    # Return results
    results = {
        'best_model_name': best_model_name,
        'best_model': best_model,
        'test_scores': test_scores,
        'importance': importance,
        'saved_paths': saved_paths,
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test)
    }
    
    # Add scale-specific information if provided
    if scale_info:
        results.update(scale_info)
    
    return results


async def main():
    """Main training pipeline with structured process."""
    parser = argparse.ArgumentParser(description="Train FINQ Stock Predictor models")
    parser.add_argument("--max-stocks", type=int, default=None, help="Maximum number of stocks to process (for testing)")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--model-type", type=str, default="random_forest", 
                       choices=["random_forest", "gradient_boosting", "logistic_regression", "svm"],
                       help="Model type to train")
    parser.add_argument("--strategy", type=str, default=None,
                       choices=["quick_test", "balanced", "comprehensive", "full"],
                       help="Training strategy (overrides MODEL_SELECTION if provided)")
    parser.add_argument("--tune-hyperparameters", action="store_true", help="Perform hyperparameter tuning")
    parser.add_argument("--save-model", action="store_true", help="Save the best trained model")
    parser.add_argument("--save-all-models", action="store_true", help="Save all trained models (not just the best one)")
    parser.add_argument("--async-training", action="store_true", help="Train models asynchronously in parallel")
    parser.add_argument("--sync-data-fetch", action="store_true", help="Use synchronous data fetching (default is async)")
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum number of worker threads for async training")
    parser.add_argument("--scale-step", type=int, default=None, help="Step size for multi-scale training (enables multi-scale mode when provided)")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for stock sampling (default: 42)")
    
    args = parser.parse_args()
    
    # Set global random seed for reproducibility
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    
    #Setup logging
    setup_logging()
    logger.info("Starting FINQ Stock Predictor training pipeline")
    logger.debug(f"Setting global random seed to {args.random_seed}")
    
    # Log strategy information
    if args.strategy:
        logger.info(f"Using training strategy: {args.strategy}")
    else:
        logger.info("Using MODEL_SELECTION configuration (no strategy specified)")
    
    try:
        # Step 1: Data Fetching
        stock_data, benchmark_data = await fetch_stock_data(args)
        
        # Step 2: Data Processing and Feature Engineering
        X_train, X_val, X_test, y_train, y_val, y_test = process_and_engineer_features(stock_data, benchmark_data)
        
        # Step 3: Model Training - Setup trainer and handle single/multi-scale training
        logger.info("Setting up trainer and model training")
        trainer = ModelTrainer()
        
        if args.scale_step is None:
            # Single-scale training using prepared data
            logger.info("Single-scale training with all {} stocks", len(stock_data))
            
            if args.async_training:
                logger.info("Using asynchronous parallel training")
                if args.max_workers:
                    logger.info("Using {} worker threads", args.max_workers)
                await trainer.train_models_async(
                    X_train, y_train, X_val, y_val, max_workers=args.max_workers, strategy=args.strategy
                )
            else:
                logger.info("Using synchronous sequential training")
                trainer.train_models(X_train, y_train, X_val, y_val, strategy=args.strategy)
            
            # Handle post-training steps (evaluation, saving, logging)
            handle_post_training_steps(trainer, X_train, y_train, X_val, y_val, X_test, y_test, stock_data, args)
            
        else:
            # Multi-scale training using prepared data with different stock counts
            logger.info("Multi-scale training enabled (scale-step: {})", args.scale_step)
            
            # Generate stock counts for training
            min_stocks = min(args.scale_step, args.max_stocks)
            stock_counts = list(range(min_stocks, args.max_stocks + 1, args.scale_step))
            if args.max_stocks not in stock_counts:
                stock_counts.append(args.max_stocks)
            
            logger.info("Training with stock counts: {}", stock_counts)
            
            # Get all available tickers from the prepared data
            all_tickers = list(X_train['ticker'].unique())
            
            results = {}
            
            for n_stocks in stock_counts:
                logger.info("=" * 60)
                logger.info("Training models with {} stocks", n_stocks)
                logger.info("=" * 60)
                
                try:
                    # Sample n_stocks from all available tickers
                    n_stocks_actual = min(n_stocks, len(all_tickers))
                    sampled_tickers = random.sample(all_tickers, n_stocks_actual)
                    logger.debug("Sampled {} tickers: {}", n_stocks_actual, sampled_tickers)
                    
                    # Filter prepared training data to only include sampled stocks
                    X_train_filtered = X_train[X_train['ticker'].isin(sampled_tickers)].copy()
                    y_train_filtered = y_train[y_train['ticker'].isin(sampled_tickers)].copy()
                    
                    logger.info("Filtered to {} training samples from {} stocks", 
                               len(X_train_filtered), n_stocks_actual)
                    
                    # Train models with filtered data
                    if args.async_training:
                        await trainer.train_models_async(
                            X_train_filtered, y_train_filtered, X_val, y_val, 
                            max_workers=args.max_workers, strategy=args.strategy
                        )
                    else:
                        trainer.train_models(
                            X_train_filtered, y_train_filtered, X_val, y_val, strategy=args.strategy
                        )
                    
                    # Handle post-training steps (evaluation, saving, logging)
                    scale_info = {
                        'stocks_count': n_stocks,
                        'sampled_tickers': sampled_tickers
                    }
                    scale_results = handle_post_training_steps(
                        trainer, X_train_filtered, y_train_filtered, X_val, y_val, X_test, y_test, 
                        stock_data, args, scale_info
                    )
                    
                    # Store results for this scale
                    results[n_stocks] = scale_results
                    
                    logger.info("Completed {}-stock training: {} (AUC: {:.4f})", 
                               n_stocks, scale_results['best_model_name'], scale_results['test_scores']['auc'])
                    
                except Exception as e:
                    logger.error("Failed training with {} stocks: {}", n_stocks, str(e))
                    results[n_stocks] = {'error': str(e)}
                    continue
            
            # Multi-scale training summary
            logger.info("=" * 60)
            logger.info("MULTI-SCALE TRAINING SUMMARY")
            logger.info("=" * 60)
            
            for n_stocks, result in results.items():
                if 'error' in result:
                    logger.error("{} stocks: FAILED - {}", n_stocks, result['error'])
                else:
                    logger.info("{} stocks: {} (AUC: {:.4f})", 
                               n_stocks, result['best_model_name'], result['test_scores']['auc'])
            
            logger.info("Multi-scale training completed successfully!")
        
    except Exception as e:
        logger.error("Training pipeline failed: {}", str(e))
        raise


if __name__ == "__main__":
    asyncio.run(main())
