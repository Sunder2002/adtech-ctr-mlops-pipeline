import pandas as pd
import yaml
import os
import sys
from pathlib import Path

# 1. HARDENED IMPORTS: Strictly aligned with Evidently 0.4.x
try:
    from evidently.report import Report
    from evidently.metric_presets import DataDriftPreset
except ImportError:
    print("Trying fallback import for older Evidently versions...")
    from evidently.metric_preset import DataDriftPreset

from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_production_monitoring():
    """
    Performs statistical drift analysis between training baseline and 
    simulated live production traffic.
    """
    logger.info("Initializing Data Drift Monitor...")
    
    # Define paths using pathlib for OS independence
    root_dir = Path(__file__).parent.parent.parent
    config_path = root_dir / "config" / "config.yaml"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    processed_data_path = root_dir / config["data"]["processed_path"]
    
    if not processed_data_path.exists():
        logger.error(f"FATAL: No data found at {processed_data_path}. Run 'make run_all' first.")
        return

    # 2. DATA PREPARATION
    logger.info(f"Loading reference data from {processed_data_path}...")
    df = pd.read_parquet(processed_data_path)
    
    # Drop non-predictive columns that cause noise in drift detection
    df_analysis = df.drop(columns=["user_id", "event_timestamp"])
    
    # Split data to simulate 'Training' (Reference) and 'Production' (Current)
    reference_data = df_analysis.sample(frac=0.5, random_state=42)
    current_data = df_analysis.sample(frac=0.5, random_state=42)
    
    # 3. DRIFT INJECTION (To prove the monitor works)
    # We simulate a 300% spike in CPM prices (Common in holiday seasons)
    logger.info("Simulating covariate shift in 'ad_spend_cpm'...")
    current_data["ad_spend_cpm"] = current_data["ad_spend_cpm"] * 3.0

    # 4. REPORT GENERATION
    logger.info("Calculating Kolmogorov-Smirnov drift metrics...")
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_data, current_data=current_data)
    
    # Ensure logs directory exists
    report_dir = root_dir / "logs"
    report_dir.mkdir(exist_ok=True)
    
    output_file = report_dir / "drift_report.html"
    drift_report.save_html(str(output_file))
    
    logger.info(f"SUCCESS: Monitoring report generated at {output_file}")

if __name__ == "__main__":
    run_production_monitoring()