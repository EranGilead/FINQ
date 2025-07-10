"""
FastAPI application for FINQ Stock Predictor.
Provides REST API endpoints for stock prediction.
"""

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, date
import uvicorn
import os
import sys
from loguru import logger

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_config import API_CONFIG, STOCKS_TICKERS
from models.inference import ModelInference, get_best_model_path


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol", example="AAPL")
    prediction_date: Optional[date] = Field(None, description="Date for prediction (YYYY-MM-DD)", example="2025-07-08")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "prediction_date": "2025-07-08"
            }
        }


class PredictionResponse(BaseModel):
    ticker: str
    prediction_date: date
    prediction: bool = Field(..., description="Whether stock will outperform S&P 500")
    outperform_probability: float = Field(..., description="Probability of outperformance (0-1)")
    confidence: float = Field(..., description="Model confidence (0-1)")
    model_name: str
    features_used: int
    last_price: float
    prediction_horizon_days: int
    
    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "prediction_date": "2025-07-08",
                "prediction": True,
                "outperform_probability": 0.72,
                "confidence": 0.72,
                "model_name": "random_forest",
                "features_used": 145,
                "last_price": 150.25,
                "prediction_horizon_days": 5
            }
        }


class BatchPredictionRequest(BaseModel):
    tickers: List[str] = Field(..., description="List of stock ticker symbols", example=["AAPL", "MSFT", "GOOGL"])
    prediction_date: Optional[date] = Field(None, description="Date for prediction (YYYY-MM-DD)", example="2025-07-08")
    
    class Config:
        json_schema_extra = {
            "example": {
                "tickers": ["AAPL", "MSFT", "GOOGL"],
                "prediction_date": "2025-07-08"
            }
        }


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    success_count: int
    failed_tickers: List[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "ticker": "AAPL",
                        "prediction_date": "2025-07-08",
                        "prediction": True,
                        "outperform_probability": 0.72,
                        "confidence": 0.72,
                        "model_name": "random_forest",
                        "features_used": 145,
                        "last_price": 150.25,
                        "prediction_horizon_days": 5
                    }
                ],
                "success_count": 1,
                "failed_tickers": []
            }
        }


class TopPredictionsResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_analyzed: int
    qualifying_predictions: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "ticker": "AAPL",
                        "prediction_date": "2025-07-08",
                        "prediction": True,
                        "outperform_probability": 0.85,
                        "confidence": 0.85,
                        "model_name": "random_forest",
                        "features_used": 145,
                        "last_price": 150.25,
                        "prediction_horizon_days": 5
                    }
                ],
                "total_analyzed": 30,
                "qualifying_predictions": 8
            }
        }


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str]
    features_count: Optional[int]
    
    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_name": "random_forest",
                "features_count": 145
            }
        }


# Initialize FastAPI app
app = FastAPI(
    title="FINQ Stock Predictor API",
    description="API for predicting stock outperformance using machine learning",
    version="1.0.0",
    contact={
        "name": "FINQ AI Team",
        "email": "ai@finq.com"
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model inference instance
model_inference = None


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global model_inference

    try:
        # Try to load the best performing model (by AUC score)
        model_path = get_best_model_path(metric="auc")
        model_inference = ModelInference(model_path)
        logger.info(
            "Best performing model loaded successfully on startup: {}", model_path
        )
    except Exception as e:
        logger.error("Failed to load best model on startup: {}", str(e))
        logger.info("Attempting to load latest model as fallback...")
        try:
            from models.inference import get_latest_model_path

            model_path = get_latest_model_path()
            model_inference = ModelInference(model_path)
            logger.info("Latest model loaded as fallback: {}", model_path)
        except Exception as fallback_error:
            logger.error("Failed to load any model: {}", str(fallback_error))
            model_inference = None


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "FINQ Stock Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Single stock prediction",
            "/predict/batch": "Batch predictions",
            "/predict/top": "Top predictions",
            "/tickers": "Available tickers",
            "/docs": "API documentation"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if model_inference is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_name=None,
            features_count=None
        )
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name=model_inference.model_name,
        features_count=len(model_inference.feature_columns) if model_inference.feature_columns else None
    )


@app.get("/tickers", response_model=List[str])
async def get_available_tickers():
    """Get list of available S&P 500 tickers."""
    return STOCKS_TICKERS


@app.post("/predict", response_model=PredictionResponse)
async def predict_stock(request: PredictionRequest):
    """
    Predict whether a stock will outperform S&P 500.
    
    Args:
        request: Prediction request with ticker and optional date
        
    Returns:
        Prediction response with results
    """
    if model_inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate ticker
    if request.ticker not in STOCKS_TICKERS:
        raise HTTPException(
            status_code=400, 
            detail=f"Ticker {request.ticker} not supported. Use /tickers endpoint for available tickers."
        )
    
    # Use current date if not provided
    prediction_date = request.prediction_date or date.today()
    prediction_datetime = datetime.combine(prediction_date, datetime.min.time())
    
    try:
        # Make prediction
        result = model_inference.predict_single_stock(request.ticker, prediction_datetime)
        
        # Convert to response format
        response = PredictionResponse(
            ticker=result['ticker'],
            prediction_date=result['prediction_date'].date(),
            prediction=result['prediction'],
            outperform_probability=result['outperform_probability'],
            confidence=result['confidence'],
            model_name=result['model_name'],
            features_used=result['features_used'],
            last_price=result['last_price'],
            prediction_horizon_days=result['prediction_horizon_days']
        )
        
        return response
        
    except Exception as e:
        logger.error("Prediction failed for {}: {}", request.ticker, str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict multiple stocks in batch.
    
    Args:
        request: Batch prediction request with tickers and optional date
        
    Returns:
        Batch prediction response
    """
    if model_inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate tickers
    invalid_tickers = [ticker for ticker in request.tickers if ticker not in STOCKS_TICKERS]
    if invalid_tickers:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid tickers: {invalid_tickers}. Use /tickers endpoint for available tickers."
        )
    
    # Use current date if not provided
    prediction_date = request.prediction_date or date.today()
    prediction_datetime = datetime.combine(prediction_date, datetime.min.time())
    
    try:
        # Make predictions
        results = model_inference.predict_multiple_stocks(request.tickers, prediction_datetime)
        
        # Convert to response format
        predictions = []
        success_tickers = set()
        
        for result in results:
            predictions.append(PredictionResponse(
                ticker=result['ticker'],
                prediction_date=result['prediction_date'].date(),
                prediction=result['prediction'],
                outperform_probability=result['outperform_probability'],
                confidence=result['confidence'],
                model_name=result['model_name'],
                features_used=result['features_used'],
                last_price=result['last_price'],
                prediction_horizon_days=result['prediction_horizon_days']
            ))
            success_tickers.add(result['ticker'])
        
        failed_tickers = [ticker for ticker in request.tickers if ticker not in success_tickers]
        
        return BatchPredictionResponse(
            predictions=predictions,
            success_count=len(predictions),
            failed_tickers=failed_tickers
        )
        
    except Exception as e:
        logger.error("Batch prediction failed: {}", str(e))
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/predict/top", response_model=TopPredictionsResponse)
async def get_top_predictions(
    top_n: int = Query(10, ge=1, le=50, description="Number of top predictions to return"),
    min_confidence: float = Query(0.6, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    prediction_date: Optional[date] = Query(None, description="Date for prediction (YYYY-MM-DD)")
):
    """
    Get top N predictions with highest confidence.
    
    Args:
        top_n: Number of top predictions to return
        min_confidence: Minimum confidence threshold
        prediction_date: Date for prediction (defaults to today)
        
    Returns:
        Top predictions response
    """
    if model_inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Use current date if not provided
    prediction_date = prediction_date or date.today()
    prediction_datetime = datetime.combine(prediction_date, datetime.min.time())
    
    try:
        # Get top predictions
        results = model_inference.get_top_predictions(
            STOCKS_TICKERS, 
            prediction_datetime, 
            top_n=top_n, 
            min_confidence=min_confidence
        )
        
        # Convert to response format
        predictions = []
        for result in results:
            predictions.append(PredictionResponse(
                ticker=result['ticker'],
                prediction_date=result['prediction_date'].date(),
                prediction=result['prediction'],
                outperform_probability=result['outperform_probability'],
                confidence=result['confidence'],
                model_name=result['model_name'],
                features_used=result['features_used'],
                last_price=result['last_price'],
                prediction_horizon_days=result['prediction_horizon_days']
            ))
        
        return TopPredictionsResponse(
            predictions=predictions,
            total_analyzed=len(STOCKS_TICKERS),
            qualifying_predictions=len(predictions)
        )
        
    except Exception as e:
        logger.error("Top predictions failed: {}", str(e))
        raise HTTPException(status_code=500, detail=f"Top predictions failed: {str(e)}")


@app.get("/model/info", response_model=dict)
async def get_model_info():
    """Get information about the loaded model."""
    if model_inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        feature_importance = model_inference.get_feature_importance(top_n=10)
        
        return {
            "model_name": model_inference.model_name,
            "feature_count": len(model_inference.feature_columns),
            "top_features": feature_importance.to_dict('records'),
            "prediction_horizon_days": 5
        }
        
    except Exception as e:
        logger.error("Failed to get model info: {}", str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "main:app",
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        reload=API_CONFIG['reload'],
        log_level=API_CONFIG['log_level']
    )
