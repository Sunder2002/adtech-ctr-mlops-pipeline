import yaml
import pandas as pd
import mlflow
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score
from src.utils.logger import get_logger

logger = get_logger(__name__)

def train_model():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    logger.info("Loading processed data...")
    df = pd.read_parquet(config["data"]["processed_path"])
    
    X = df.drop(columns=[config["model"]["target"]])
    y = df[config["model"]["target"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["model"]["test_size"], random_state=config["model"]["random_state"]
    )

    # Setup MLflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    logger.info("Starting MLflow run and training XGBoost model...")
    with mlflow.start_run():
        params = config["model"]["params"]
        mlflow.log_params(params)

        model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")
        model.fit(X_train, y_train)

        # Predictions & Metrics
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "precision": float(precision_score(y_test, preds, zero_division=0))
        }
        mlflow.log_metrics(metrics)
        logger.info(f"Model trained. Metrics: {metrics}")

        # Register Model
        mlflow.xgboost.log_model(model, "xgboost-ctr-model")
        logger.info("Model saved to MLflow.")

if __name__ == "__main__":
    train_model()