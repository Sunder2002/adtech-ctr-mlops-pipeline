import pandas as pd
import yaml
import os
from pathlib import Path
import evidently
from evidently.report import Report
# Senior Fix: Use the most stable import path for 0.4.x
from evidently import metric_presets

from src.utils.logger import get_logger

logger = get_logger(__name__)

def run():
    logger.info("Starting Production Drift Analysis...")
    root_dir = Path(__file__).parent.parent.parent
    with open(root_dir / "config" / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    data_path = root_dir / config["data"]["processed_path"]
    if not data_path.exists():
        logger.error("Data missing.")
        return

    df = pd.read_parquet(data_path)
    df_clean = df.drop(columns=["user_id", "event_timestamp"])
    
    reference = df_clean.sample(frac=0.5, random_state=42)
    current = df_clean.sample(frac=0.5, random_state=42)
    current["ad_spend_cpm"] = current["ad_spend_cpm"] * 5.0 

    # Use the preset directly from the module
    report = Report(metrics=[metric_presets.DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    
    log_dir = root_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    report.save_html(str(log_dir / "drift_report.html"))
    logger.info("Success! Report saved to logs/drift_report.html")

if __name__ == "__main__":
    run()