import yaml
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import xgboost as xgb
import ray
import os
import sys
import traceback
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from src.utils.logger import get_logger

logger = get_logger(__name__)

@ray.remote
def distributed_train_worker(config, X_train, y_train, X_test, y_test):
    """Ray Worker: Executes XGBoost training with strict type enforcement."""
    try:
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        
        with mlflow.start_run(nested=True, run_name=f"Ray_Node_{datetime.now().strftime('%H%M%S')}"):
            model = xgb.XGBClassifier(
                **config["model"]["params"],
                use_label_encoder=False,
                eval_metric="logloss"
            )
            model.fit(X_train, y_train)
            
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
            
            # SENIOR FIX: Explicit casting to resolve 'Unknown Target' errors
            y_true_clean = y_test.astype(int)
            preds_clean = preds.astype(int)
            
            metrics = {
                "accuracy": float(accuracy_score(y_true_clean, preds_clean)),
                "roc_auc": float(roc_auc_score(y_true_clean, proba)),
                "f1": float(f1_score(y_true_clean, preds_clean))
            }
            
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "xgboost-ctr-model", input_example=X_train.head(1))
            return metrics
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

def run_production_training():
    logger.info("🚀 Initializing MiQ Distributed Training Pipeline...")
    ray.init(include_dashboard=False, ignore_reinit_error=True, num_cpus=2)
    
    try:
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)

        processed_path = config["data"]["processed_path"]
        if not os.path.exists(processed_path):
            logger.error("Data missing. Run ETL first.")
            sys.exit(1)
            
        df = pd.read_parquet(processed_path).dropna(subset=[config["model"]["target"]])
        
        X = df.drop(columns=[config["model"]["target"], "user_id", "event_timestamp"])
        y = df[config["model"]["target"]].astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["model"]["test_size"], random_state=42)

        X_ref = ray.put(X_train)
        y_ref = ray.put(y_train)
        
        results = ray.get(distributed_train_worker.remote(config, X_ref, y_ref, X_test, y_test))
        
        if isinstance(results, dict) and "error" in results:
            logger.error(f"Worker Failed: {results['error']}")
            sys.exit(1)
            
        logger.info(f"✅ Training Success. ROC-AUC: {results['roc_auc']:.4f}")
    except Exception as e:
        logger.error(f"Pipeline Crash: {traceback.format_exc()}")
        sys.exit(1)
    finally:
        ray.shutdown()

if __name__ == "__main__":
    run_production_training()