#!/usr/bin/env python3
"""
Script to compare all saved models and their performance metrics.
"""

import pandas as pd
import os
from loguru import logger
from models.inference import ModelInference
from main_config import MODELS_DIR

def compare_all_models():
    """Compare all saved models and display their performance metrics."""
    
    logger.info("=== Comparing All Saved Models ===")
    
    inference = ModelInference()
    models_df = inference.list_saved_models()
    
    if models_df.empty:
        logger.info("No saved models found")
        return
    
    # Display model comparison
    logger.info("Found {} saved models", len(models_df))
    
    # Display summary table
    display_columns = [
        'model_name', 'timestamp', 'auc', 'accuracy', 'precision', 
        'recall', 'f1_score', 'training_samples', 'file_size_mb'
    ]
    
    print("\n" + "="*120)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*120)
    
    # Format the DataFrame for display
    display_df = models_df[display_columns].copy()
    display_df['auc'] = display_df['auc'].apply(lambda x: f"{x:.4f}")
    display_df['accuracy'] = display_df['accuracy'].apply(lambda x: f"{x:.4f}")
    display_df['precision'] = display_df['precision'].apply(lambda x: f"{x:.4f}")
    display_df['recall'] = display_df['recall'].apply(lambda x: f"{x:.4f}")
    display_df['f1_score'] = display_df['f1_score'].apply(lambda x: f"{x:.4f}")
    
    # Display the table
    print(display_df.to_string(index=False))
    
    # Show best model
    best_model = models_df.iloc[0]
    print(f"\n🏆 BEST MODEL: {best_model['model_name']} (AUC: {best_model['auc']:.4f})")
    print(f"   File: {best_model['file_name']}")
    print(f"   Trained: {best_model['training_date']}")
    
    # Show model breakdown by type
    print(f"\n📊 MODEL TYPE BREAKDOWN:")
    model_counts = models_df['model_name'].value_counts()
    for model_type, count in model_counts.items():
        avg_auc = models_df[models_df['model_name'] == model_type]['auc'].mean()
        print(f"   {model_type}: {count} models (avg AUC: {avg_auc:.4f})")
    
    return models_df

def test_specific_model(model_name: str = None):
    """Test a specific model on new data."""
    
    if not model_name:
        logger.info("No model name specified, using best model")
        inference = ModelInference()
        model_path = inference.list_saved_models()[0]
        inference.load_model(model_path)
    else:
        # Find model by name
        inference = ModelInference()
        models_df = inference.list_saved_models()
        
        matching_models = models_df[models_df['model_name'] == model_name]
        if matching_models.empty:
            logger.error("No models found with name: {}", model_name)
            return
        
        # Use the best (first) matching model
        best_match = matching_models.iloc[0]
        model_path = os.path.join(MODELS_DIR, best_match['file_name'])
        
        inference = ModelInference()
        inference.load_model(model_path)
        
        logger.info("Loaded {} model from {}", model_name, best_match['file_name'])
    
    # Test on sample tickers
    test_tickers = ["AAPL", "GOOGL", "MSFT"]
    from datetime import datetime
    
    logger.info("Testing model on sample tickers: {}", test_tickers)
    
    for ticker in test_tickers:
        try:
            result = inference.predict_single_stock(ticker, datetime(2025, 7, 7))
            logger.info("{}: Prediction={}, Confidence={:.4f}", 
                       ticker, result.get('prediction', 'N/A'), result.get('confidence', 0.0))
        except Exception as e:
            logger.warning("Failed to predict for {}: {}", ticker, str(e))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare saved models")
    parser.add_argument("--test-model", type=str, help="Test specific model by name")
    
    args = parser.parse_args()
    
    # Compare all models
    models_df = compare_all_models()
    
    # Test specific model if requested
    if args.test_model:
        test_specific_model(args.test_model)
    elif not models_df.empty:
        # Test the best model
        best_model_name = models_df.iloc[0]['model_name']
        test_specific_model(best_model_name)
