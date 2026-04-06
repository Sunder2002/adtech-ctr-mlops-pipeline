import time
import psutil
import hashlib
import random
import numpy as np
import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger(__name__)
app = FastAPI(title="MiQ Enterprise Bidding Platform v7.1")

# --- STATION 1: BEHAVIORAL FEATURE STORE (The "Memory") ---
class FeatureStore:
    def __init__(self):
        self._profiles = {} 
        self.history = []
        self.metrics = {"total": 0, "bids": 0, "spend": 0.0}

    def get_profile(self, uid):
        if uid not in self._profiles:
            # Cold Start: New users start with baseline noise
            self._profiles[uid] = {"electronics": 0.005, "fashion": 0.005, "automotive": 0.005}
        return self._profiles[uid]

    def update_intent(self, uid, query):
        profile = self.get_profile(uid)
        q = query.lower()
        # Behavioral Reinforcement: Keywords update the Feature Vector
        if any(x in q for x in ["phone", "laptop", "tech", "macbook"]): profile["electronics"] += 0.15
        if any(x in q for x in ["car", "tire", "tesla"]): profile["automotive"] += 0.15
        if any(x in q for x in ["dress", "shirt", "nike"]): profile["fashion"] += 0.15
        for k in profile: profile[k] = min(profile[k], 1.0)

    def log_auction(self, uid, cat, bid, proba, price):
        self.metrics["total"] += 1
        if bid:
            self.metrics["bids"] += 1
            self.metrics["spend"] += price
        self.history.insert(0, {
            "t": datetime.now().strftime("%H:%M:%S"),
            "u": uid[:8],
            "c": cat,
            "r": "WON ✅" if bid else "PASS ❌",
            "p": f"{proba*100:.1f}%"
        })
        self.history = self.history[:15]

feature_store = FeatureStore()
model = None

@app.on_event("startup")
def sync_model():
    global model
    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("MiQ_Ad_CTR_Prediction")
        if exp:
            runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"])
            if runs:
                model = mlflow.sklearn.load_model(f"runs:/{runs[0].info.run_id}/xgboost-ctr-model")
                logger.info("Predictive Engine Synchronized.")
    except Exception as e: logger.error(f"Registry Sync Failed: {e}")

# --- STATION 2: THE BIDDING ENGINE (ML-Driven) ---
class BidRequest(BaseModel):
    email: str
    query: str

@app.post("/bid")
async def handle_bid(req: BidRequest):
    start = time.perf_counter()
    if not model: raise HTTPException(status_code=503)
    
    uid = hashlib.sha256(req.email.encode()).hexdigest()
    feature_store.update_intent(uid, req.query)
    profile = feature_store.get_profile(uid)
    
    # Determine dominant category for the ML feature vector
    category = max(profile, key=profile.get)
    
    # ML FEATURE VECTORIZATION
    # This is where the "Generated Data" patterns are applied to "Live Data"
    features = {
        "ad_spend_cpm": np.float64(2.50),
        "historical_user_ctr": np.float64(profile[category]),
        "is_peak_hour": np.int32(1 if 17 <= datetime.now().hour <= 21 else 0),
        "intent_electronics": np.int32(1 if category == "electronics" and profile[category] > 0.01 else 0),
        "intent_automotive": np.int32(1 if category == "automotive" and profile[category] > 0.01 else 0),
        "is_shopping_site": np.int32(1 if profile[category] > 0.1 else 0)
    }
    
    df = pd.DataFrame([features])
    cols = ["ad_spend_cpm", "historical_user_ctr", "is_peak_hour", "intent_electronics", "intent_automotive", "is_shopping_site"]
    
    # The ML Model makes the decision. 
    # If you search 'icecream', intent_electronics is 0, proba will be < 0.04 -> PASS.
    proba = float(model.predict_proba(df[cols])[0][1])
    do_bid = proba > 0.04
    price = 2.50 * 1.15 if do_bid else 0.0
    
    feature_store.log_auction(uid, category, do_bid, proba, price)
    return {"bid": do_bid, "proba": proba, "category": category, "latency": round((time.perf_counter()-start)*1000, 2)}

# --- STATION 3: THE DASHBOARD (Fixed Division-by-Zero) ---
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    mem = psutil.Process().memory_info().rss / (1024 * 1024)
    total = max(1, feature_store.metrics["total"])
    win_rate = (feature_store.metrics["bids"] / total) * 100
    
    rows = "".join([f"<tr><td>{h['t']}</td><td>{h['u']}</td><td>{h['c']}</td><td>{h['r']}</td><td>{h['p']}</td></tr>" for h in feature_store.history])
    
    return f"""
    <html><head><title>MiQ RTB Monitor</title><meta http-equiv="refresh" content="2">
    <style>
        body {{ font-family: sans-serif; background: #121212; color: white; padding: 40px; }}
        .card-container {{ display: flex; gap: 20px; margin-bottom: 30px; }}
        .card {{ background: #1e1e1e; padding: 20px; border-radius: 10px; flex: 1; border-bottom: 4px solid #007bff; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #333; }}
    </style></head>
    <body>
        <h1>📊 MiQ Live Bidding Monitor</h1>
        <div class="card-container">
            <div class="card"><h3>Auctions</h3><p>{feature_store.metrics['total']}</p></div>
            <div class="card"><h3>Win Rate</h3><p>{win_rate:.1f}%</p></div>
            <div class="card"><h3>Memory</h3><p>{mem:.2f} MB</p></div>
        </div>
        <table>
            <thead><tr><th>Time</th><th>User</th><th>Category</th><th>Result</th><th>Confidence</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </body></html>
    """

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <body style="font-family:sans-serif; padding:50px; text-align:center; background:#f4f4f4;">
        <h1>TechStream News Portal</h1>
        <p>The ML model learns your intent. Try 'laptop' vs 'icecream'.</p>
        <input id="q" style="padding:10px; width:300px;" placeholder="Search..."> 
        <button onclick="go()" style="padding:10px; background:#007bff; color:white; border:none;">Search</button>
        <div id="ad" style="margin-top:30px; font-size:20px;"></div>
        <script>
            async function go() {
                const res = await fetch('/bid', {
                    method:'POST', headers: {'Content-Type':'application/json'},
                    body: JSON.stringify({email: 'hruthik@miq.com', query: document.getElementById('q').value})
                });
                const data = await res.json();
                document.getElementById('ad').innerHTML = data.bid ? 
                    "<div style='color:green'>WON BID: Showing " + data.category + " Ad (" + Math.round(data.proba*100) + "% confidence)</div>" : 
                    "<div style='color:red'>PASS: Low ML Confidence (" + Math.round(data.proba*100) + "%)</div>";
            }
        </script>
    </body>
    """