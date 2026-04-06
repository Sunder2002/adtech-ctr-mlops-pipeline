from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
import yaml
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Load config safely
try:
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    logger.warning("Config file not found. This is normal during CI/CD testing.")
    config = {"mlflow": {"tracking_uri": "sqlite:///mlflow.db", "experiment_name": "test"}}

app = FastAPI(title="AdTech CTR Prediction API", version="1.0.0")

# Define Data Validation Schema (Strict typing for production)
class AdRequest(BaseModel):
    ad_spend_cpm: float
    hour_of_day: int
    is_peak_hour: int
    is_mobile: int
    is_shopping_site: int

# Attempt to load model at startup
try:
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(config["mlflow"]["experiment_name"])
    if experiment:
        runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
        if runs:
            latest_run_id = runs[0].info.run_id
            model_uri = f"runs:/{latest_run_id}/xgboost-ctr-model"
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info("Successfully loaded latest MLflow model.")
        else:
            model = None
    else:
        model = None
except Exception as e:
    logger.warning(f"Model could not be loaded at startup (Normal if not trained yet): {e}")
    model = None

@app.get("/health")
def health_check():
    """Production health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict_ctr(request: AdRequest):
    """Generates CTR prediction based on ad request features."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable. Train the model first.")
    
    # Convert validated Pydantic model to DataFrame for MLflow (Using V1 .dict() method)
    data = pd.DataFrame([request.dict()])
    
    try:
        prediction = model.predict(data)
        return {"click_prediction": int(prediction[0]), "status": "success"}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal inference error")