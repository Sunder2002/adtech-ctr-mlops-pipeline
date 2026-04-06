import time
import yaml
import mlflow
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Production Configuration Loader
def load_config():
    try:
        with open("config/config.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return {"mlflow": {"tracking_uri": "sqlite:///mlflow.db", "experiment_name": "MiQ_Ad_CTR_Prediction"}}

config = load_config()

app = FastAPI(
    title="MiQ Programmatic Bidding Engine",
    description="Scalable CTR inference engine supporting OpenRTB standards.",
    version="2.4.0"
)

# 1. Pydantic V2 Schemas
class BidRequest(BaseModel):
    auction_id: str = Field(..., min_length=1)
    user_id: int
    site_domain: str
    user_agent: str
    device_type: str = "mobile"
    floor_price: float = Field(default=0.01, ge=0)

    @field_validator('floor_price')
    @classmethod
    def ensure_positive_floor(cls, v: float) -> float:
        if v < 0: raise ValueError("Floor price must be >= 0")
        return v

class BidResponse(BaseModel):
    id: str
    bid: bool
    price: float
    currency: str = "USD"
    ctr_prediction: int
    latency_ms: float

# 2. Predictive Engine (Thread-Safe Registry)
class PredictiveEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PredictiveEngine, cls).__new__(cls)
            cls._instance.model = None
            cls._instance._load()
        return cls._instance

    def _load(self):
        logger.info("Syncing with MLflow Model Registry...")
        try:
            mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
            client = mlflow.tracking.MlflowClient()
            exp = client.get_experiment_by_name(config["mlflow"]["experiment_name"])
            if exp:
                # We pull the best run based on the most recent SUCCESSFUL training
                runs = client.search_runs(
                    exp.experiment_id, 
                    filter_string="attributes.status = 'FINISHED'",
                    order_by=["attributes.start_time DESC"], 
                    max_results=1
                )
                if runs:
                    self.model = mlflow.pyfunc.load_model(f"runs:/{runs[0].info.run_id}/xgboost-ctr-model")
                    logger.info(f"Engine Online: Loaded Run {runs[0].info.run_id}")
        except Exception as e:
            logger.error(f"REGISTRY_FAILURE: {e}")

engine = PredictiveEngine()

# 3. Request Handlers
@app.get("/health")
def health():
    return {
        "status": "ready", 
        "model_loaded": engine.model is not None,
        "runtime": "Python 3.12"
    }

@app.post("/bid", response_model=BidResponse)
async def process_bid(request: BidRequest):
    """
    Handles Real-Time Bidding (RTB) logic.
    Calculates CTR prediction and decides bid price in <100ms.
    """
    start_time = time.perf_counter()
    
    if engine.model is None:
        raise HTTPException(status_code=503, detail="Model Engine Initializing")

    try:
        # Pydantic V2 model_dump()
        payload = request.model_dump()
        
        # Real-time feature derivation
        hour = datetime.now().hour
        features = {
            "ad_spend_cpm": payload["floor_price"],
            "hour_of_day": hour,
            "is_peak_hour": 1 if 17 <= hour <= 21 else 0,
            "is_mobile": 1 if "mobi" in payload["user_agent"].lower() else 0,
            "is_shopping_site": 1 if "shop" in payload["site_domain"].lower() else 0
        }
        
        # Scoring
        prediction = int(engine.model.predict(pd.DataFrame([features]))[0])
        
        # Bidding Strategy: 15% Premium on predicted clicks
        bid_price = payload["floor_price"] * 1.15 if prediction == 1 else 0.0
        
        latency = (time.perf_counter() - start_time) * 1000

        return BidResponse(
            id=payload["auction_id"],
            bid=bid_price > 0,
            price=round(bid_price, 4),
            ctr_prediction=prediction,
            latency_ms=round(latency, 2)
        )
    except Exception as e:
        logger.error(f"INFERENCE_ERROR: {e}")
        raise HTTPException(status_code=500, detail="Internal Logic Error")