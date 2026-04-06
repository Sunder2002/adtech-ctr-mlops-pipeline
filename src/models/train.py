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

# Initialize Production Logger
logger = get_logger(__name__)

@ray.remote
def distributed_train_worker(config, X_train, y_train, X_test, y_test):
    """
    Ray Remote Worker: Executes XGBoost training in an isolated process.
    Includes strict type enforcement to prevent 'Unknown Target' errors.
    """
    try:
        # Workers must set their own tracking URI in distributed mode
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        
        run_name = f"Ray_Node_{datetime.now().strftime('%H%M%S')}"
        with mlflow.start_run(nested=True, run_name=run_name):
            # Initialize Model with parameters from centralized config
            model = xgb.XGBClassifier(
                **config["model"]["params"],
                use_label_encoder=False,
                eval_metric="logloss"
            )
            
            # Execute Training
            model.fit(X_train, y_train)
            
            # Evaluation Phase
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
            
            # SENIOR FIX: Explicitly cast to int to resolve 'Unknown Target' errors
            # This ensures sklearn sees strict binary [0, 1] types
            y_true_clean = y_test.astype(int)
            preds_clean = preds.astype(int)
            
            metrics = {
                "accuracy": float(accuracy_score(y_true_clean, preds_clean)),
                "roc_auc": float(roc_auc_score(y_true_clean, proba)),
                "f1": float(f1_score(y_true_clean, preds_clean))
            }
            
            # Log Metrics and Model to Registry
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                sk_model=model, 
                artifact_path="xgboost-ctr-model",
                input_example=X_train.head(1)
            )
            
            return metrics
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

def run_production_training():
    """
    Main Orchestrator for the Distributed Training Pipeline.
    Optimized for Python 3.12 and resource-constrained environments.
    """
    logger.info("🚀 Initializing MiQ Distributed Training Pipeline (Ray 2.31+)...")
    
    # Initialize Ray Cluster (Headless mode for Codespaces)
    ray.init(
        include_dashboard=False,
        ignore_reinit_error=True,
        num_cpus=2 
    )
    
    try:
        # 1. Load Configuration
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # 2. Ingest Processed Data
        processed_path = config["data"]["processed_path"]
        if not os.path.exists(processed_path):
            logger.error(f"FATAL: Processed data missing at {processed_path}")
            sys.exit(1)
            
        df = pd.read_parquet(processed_path)
        
        # 3. Feature/Metadata Separation
        target = config["model"]["target"]
        metadata = ["user_id", "event_timestamp"]
        
        # Ensure no NaNs leaked into the training set
        df = df.dropna(subset=[target])
        
        X = df.drop(columns=[target] + metadata)
        y = df[target].astype(int) # Force binary target type
        
        logger.info(f"Feature Schema Validated. Training on: {list(X.columns)}")

        # 4. Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config["model"]["test_size"], 
            random_state=config["model"]["random_state"]
        )

        # 5. Ray Object Store Optimization
        logger.info("Broadcasting training data to Ray Object Store...")
        X_train_ref = ray.put(X_train)
        y_train_ref = ray.put(y_train)
        
        # 6. Launch Distributed Task
        worker_future = distributed_train_worker.remote(
            config, X_train_ref, y_train_ref, X_test, y_test
        )
        
        # Block until worker completes
        results = ray.get(worker_future)
        
        if isinstance(results, dict) and "error" in results:
            logger.error(f"Worker Process Failed: {results['error']}")
            print(results['trace'])
            sys.exit(1)
            
        logger.info(f"✅ Pipeline Success. Final ROC-AUC: {results['roc_auc']:.4f}")

    except Exception as e:
        logger.error(f"Pipeline Crash: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        ray.shutdown()
        logger.info("Ray cluster successfully decommissioned.")

if __name__ == "__main__":
    run_production_training()