import pandas as pd
import yaml
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_drift_monitoring():
    logger.info("Starting Data Drift Monitoring Phase...")
    
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # Load the processed data (our 'reference' data used for training)
    reference_data = pd.read_parquet(config["data"]["processed_path"])
    
    # Simulate 'Current' live traffic data (Injecting artificial drift for demonstration)
    logger.info("Simulating live API traffic with modified behavior (Black Friday Ad Spend spike)...")
    current_data = reference_data.copy()
    current_data["ad_spend_cpm"] = current_data["ad_spend_cpm"] * 2.5  # Artificial drift
    
    # Run Evidently AI Drift Report
    logger.info("Calculating statistical drift across all features...")
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_data, current_data=current_data)
    
    # Save the report
    os.makedirs("logs", exist_ok=True)
    report_path = "logs/data_drift_report.html"
    drift_report.save_html(report_path)
    
    logger.info(f"Monitoring complete. Drift report saved to {report_path}")
    logger.warning("ALERT: Data Drift detected in 'ad_spend_cpm'. Model retraining recommended.")

if __name__ == "__main__":
    run_drift_monitoring()