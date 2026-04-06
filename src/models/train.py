import yaml
import pandas as pd
import mlflow
import xgboost as xgb
import os
import sys
import traceback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, f1_score
from src.utils.logger import get_logger

# Initialize Production Logger
logger = get_logger(__name__)

def train_model():
    """
    Orchestrates the training lifecycle of the CTR prediction model.
    Implements strict feature selection to isolate metadata from learning signals.
    """
    logger.info("Starting Model Training Pipeline...")
    
    try:
        # 1. Load Global Configuration
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)

        processed_path = config["data"]["processed_path"]
        if not os.path.exists(processed_path):
            logger.error(f"TRAIN_ERROR: Processed data not found at {processed_path}")
            sys.exit(1)

        # 2. Ingest Data
        logger.info(f"Loading processed feature set from {processed_path}")
        df = pd.read_parquet(processed_path)

        # 3. Production Feature Selection (CRITICAL FIX)
        # We define what the model learns from vs what is just metadata
        metadata_cols = ["user_id", "event_timestamp"]
        target_col = config["model"]["target"]
        
        # Features are everything EXCEPT target and metadata
        X = df.drop(columns=[target_col] + metadata_cols)
        y = df[target_col]

        logger.info(f"Feature Set identified: {list(X.columns)}")
        logger.info(f"Metadata columns excluded from training: {metadata_cols}")

        # 4. Data Splitting
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config["model"]["test_size"], 
            random_state=config["model"]["random_state"]
        )

        # 5. MLflow Tracking Setup
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])

        with mlflow.start_run(run_name=f"CTR_XGBoost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            logger.info("Initializing XGBoost Training with MLflow tracking...")
            
            # Log Hyperparameters from Config
            params = config["model"]["params"]
            mlflow.log_params(params)

            # Define and Fit Model
            # use_label_encoder=False and eval_metric are standard for modern XGBoost
            model = xgb.XGBClassifier(
                **params, 
                use_label_encoder=False, 
                eval_metric="logloss"
            )
            model.fit(X_train, y_train)

            # 6. Model Evaluation
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": float(accuracy_score(y_test, preds)),
                "roc_auc": float(roc_auc_score(y_test, proba)),
                "f1_score": float(f1_score(y_test, preds)),
                "precision": float(precision_score(y_test, preds, zero_division=0))
            }
            
            mlflow.log_metrics(metrics)
            logger.info(f"Training Complete. Performance Metrics: {metrics}")

            # 7. Model Serialization & Registry
            # We save the model into MLflow as a 'pyfunc' for unified inference
            mlflow.sklearn.log_model(
                sk_model=model, 
                artifact_path="xgboost-ctr-model",
                input_example=X_train.head(1)
            )
            logger.info("Model artifact successfully persisted to MLflow Registry.")

    except Exception as e:
        logger.error(f"TRAINING_PIPELINE_FAILED: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    finally:
        logger.info("Training pipeline process exited.")

if __name__ == "__main__":
    from datetime import datetime
    train_model()