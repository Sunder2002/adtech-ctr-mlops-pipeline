import time
import psutil
import hashlib
import random
import numpy as np
import pandas as pd
import mlflow.sklearn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger(__name__)

model = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("MiQ_Ad_CTR_Prediction")
        if exp:
            runs = client.search_runs(exp.experiment_id, filter_string="attributes.status = 'FINISHED'", order_by=["start_time DESC"], max_results=1)
            if runs:
                model = mlflow.sklearn.load_model(f"runs:/{runs[0].info.run_id}/xgboost-ctr-model")
                logger.info("Predictive Engine Online.")
    except Exception as e: logger.error(f"Registry Sync Failed: {e}")
    yield

app = FastAPI(title="MiQ Enterprise Bidding Platform", lifespan=lifespan)
templates = Jinja2Templates(directory="src/api/templates")

class FeatureStore:
    def __init__(self):
        self._profiles = {} 
        self.history =[]
        self.metrics = {"total": 0, "bids": 0}

    def get_profile(self, uid):
        if uid not in self._profiles:
            self._profiles[uid] = {"electronics": 0.005, "automotive": 0.005, "general": 0.005}
        return self._profiles[uid]

    def update_intent(self, uid, query):
        profile = self.get_profile(uid)
        q = query.lower()
        if any(x in q for x in ["phone", "laptop", "macbook"]): profile["electronics"] += 0.20
        elif any(x in q for x in["car", "tesla", "tires"]): profile["automotive"] += 0.20
        else: profile["general"] += 0.01
        for k in profile: profile[k] = min(profile[k], 1.0)

    def log(self, uid, cat, bid, proba):
        self.metrics["total"] += 1
        if bid: self.metrics["bids"] += 1
        self.history.insert(0, {"t": datetime.now().strftime("%H:%M:%S"), "u": uid[:8], "c": cat, "r": "WON ✅" if bid else "PASS ❌", "p": f"{proba*100:.1f}%"})
        self.history = self.history[:15]

feature_store = FeatureStore()

class BidRequest(BaseModel):
    email: str
    query: str

@app.post("/bid")
async def bid(req: BidRequest):
    start = time.perf_counter()
    if not model: raise HTTPException(status_code=503, detail="Model not loaded")
    
    uid = hashlib.sha256(req.email.encode()).hexdigest()
    feature_store.update_intent(uid, req.query)
    profile = feature_store.get_profile(uid)
    category = max(profile, key=profile.get)
    
    # EXACT MATCH TO TRAINING FEATURES
    features = {
        "ad_spend_cpm": np.float64(2.50),
        "historical_user_ctr": np.float64(profile[category]),
        "is_peak_hour": np.int32(1 if 17 <= datetime.now().hour <= 21 else 0),
        "is_mobile": np.int32(1), # Simulating mobile device
        "intent_electronics": np.int32(1 if category == "electronics" else 0),
        "intent_automotive": np.int32(1 if category == "automotive" else 0),
        "is_shopping_site": np.int32(1 if category != "general" else 0)
    }
    
    df = pd.DataFrame([features])
    cols =["ad_spend_cpm", "historical_user_ctr", "is_peak_hour", "is_mobile", "intent_electronics", "intent_automotive", "is_shopping_site"]
    
    proba = float(model.predict_proba(df[cols])[0][1])
    do_bid = proba > 0.05
    
    feature_store.log(uid, category, do_bid, proba)
    
    adm = f"<div style='background:#fff3cd; padding:10px;'><h3>🎯 {category.upper()} Deal</h3><p>Personalized for you.</p></div>"
    return {"bid": do_bid, "adm": adm, "proba": proba, "category": category, "latency": round((time.perf_counter()-start)*1000, 2)}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    mem = psutil.Process().memory_info().rss / (1024 * 1024)
    total = max(1, feature_store.metrics["total"])
    win_rate = round((feature_store.metrics["bids"] / total) * 100, 1)
    return templates.TemplateResponse("dashboard.html", {
        "request": request, "total": feature_store.metrics["total"], 
        "win_rate": win_rate, "mem": round(mem, 2), "history": feature_store.history
    })

@app.get("/health")
def health():
    return {"status": "ready", "model_loaded": model is not None}