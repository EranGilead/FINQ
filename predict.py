#!/usr/bin/env python3
"""
Simple one-liner to get a stock prediction.
Usage: python predict.py TICKER
Example: python predict.py AAPL
"""

import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(__file__))

def quick_predict(ticker):
    """Get a quick prediction for a ticker."""
    try:
        from models.inference import ModelInference, get_best_model_path
        
        print(f"🔍 Loading best model...")
        model_path = get_best_model_path("auc")
        inference = ModelInference(model_path)
        
        print(f"📈 Predicting for {ticker.upper()}...")
        result = inference.predict_single_stock(ticker.upper(), datetime.now())
        
        print(f"\n🎯 PREDICTION RESULT:")
        print(f"   Ticker: {result['ticker']}")
        print(f"   Prediction: {'✅ OUTPERFORM' if result['prediction'] == 1 else '❌ UNDERPERFORM'} S&P 500")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Model: {result.get('model_name', 'Unknown')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Try running setup first:")
        print("   python setup.py")
        print("   python train.py --save-model --max-stocks 20")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py TICKER")
        print("Example: python predict.py AAPL")
        sys.exit(1)
    
    ticker = sys.argv[1]
    quick_predict(ticker)
