from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import mlflow
import yaml
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="MiQ Ad-Tech Bidding Engine v2.5.0")

class BidRequest(BaseModel):
    auction_id: str
    user_id: int
    site_domain: str
    user_agent: str
    floor_price: float

model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("MiQ_Ad_CTR_Prediction")
        if exp:
            runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"])
            if runs:
                model = mlflow.pyfunc.load_model(f"runs:/{runs[0].info.run_id}/xgboost-ctr-model")
                logger.info("API is LIVE: Latest model registered.")
    except Exception as e:
        logger.error(f"Startup Model Load Error: {e}")

@app.get("/", response_class=HTMLResponse)
def root():
    """Returns a professional landing page for the service."""
    return """
    <html>
        <body style="font-family: sans-serif; padding: 50px;">
            <h1>🚀 MiQ Bidding Engine v2.5.0</h1>
            <p>Status: <span style="color: green;">Online</span></p>
            <hr>
            <h3>Service Links:</h3>
            <ul>
                <li><a href="/docs">Interactive API Documentation (Swagger)</a></li>
                <li><a href="/health">System Health Check</a></li>
            </ul>
        </body>
    </html>
    """

@app.get("/health")
def health():
    return {"status": "ready", "model_loaded": model is not None}

@app.post("/bid")
def bid(request: BidRequest):
    if model is None: raise HTTPException(status_code=503, detail="Model Loading")
    
    try:
        hour = datetime.now().hour
        # STRICT TYPE CASTING: Fixed for MLflow compatibility
        features = {
            "ad_spend_cpm": np.float64(request.floor_price),
            "hour_of_day": np.int64(hour),
            "is_peak_hour": np.int32(1 if 17 <= hour <= 21 else 0),
            "is_mobile": np.int32(1 if "mobi" in request.user_agent.lower() else 0),
            "is_shopping_site": np.int32(1 if "shop" in request.site_domain.lower() else 0)
        }
        
        df_input = pd.DataFrame([features])
        df_input = df_input[['ad_spend_cpm', 'hour_of_day', 'is_peak_hour', 'is_mobile', 'is_shopping_site']]
        
        prediction = int(model.predict(df_input)[0])
        bid_price = request.floor_price * 1.15 if prediction == 1 else 0.0
        
        return {"id": request.auction_id, "bid": bid_price > 0, "price": round(bid_price, 4)}
    except Exception as e:
        logger.error(f"Inference crash: {e}")
        raise HTTPException(status_code=500, detail=str(e))