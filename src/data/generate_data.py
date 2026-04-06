import pandas as pd
import numpy as np
import yaml
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

def generate_complex_data(output_path: str, n_rows: int = 50000) -> None:
    logger.info(f"Generating {n_rows} rows of complex, messy Ad-Tech data...")
    np.random.seed(42)
    
    # 1. Broad User Signals
    regions = ["North_America", "EMEA", "APAC", "LATAM"]
    interests_pool = ["Electronics", "Fashion", "Sports", "Automotive", "Travel", "FinTech"]
    
    data = {
        "user_id": np.random.randint(100000, 999999, n_rows),
        "region": np.random.choice(regions, n_rows),
        # Real-world mess: Mixed casing in device types
        "device_type": np.random.choice(["MOBILE", "mobile", "Desktop", "DESKTOP", "tablet"], n_rows),
        # Comma-separated strings represent cross-portal search history
        "user_interests": [",".join(np.random.choice(interests_pool, np.random.randint(1, 4))) for _ in range(n_rows)],
        # Historical performance (numerical behavior)
        "historical_user_ctr": np.random.uniform(0.001, 0.1, n_rows),
        "hour_of_day": np.random.randint(0, 24, n_rows),
        "site_category": np.random.choice(["news", "shopping", "social", "video"], n_rows)
    }
    
    df = pd.DataFrame(data)
    
    # 2. Simulate Realistic CPM (Cost per Mille) with NULLs (Messiness)
    df["ad_spend_cpm"] = np.random.uniform(0.5, 10.0, n_rows)
    # Inject 5% missing values to simulate tracking failures
    null_mask = np.random.random(n_rows) < 0.05
    df.loc[null_mask, "ad_spend_cpm"] = np.nan

    # 3. Decision Logic: Why would a user click?
    # A click isn't random; it's a function of Intention + History + Context
    click_prob = 0.01 + (df["historical_user_ctr"] * 0.5)
    click_prob += np.where(df["site_category"] == "shopping", 0.03, 0)
    click_prob += np.where(df["user_interests"].str.contains("Electronics"), 0.02, 0)
    click_prob += np.where(df["device_type"].str.lower() == "mobile", 0.02, 0)
    
    df["is_clicked"] = (np.random.random(n_rows) < click_prob).astype(int)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Complex data persisted. Overall CTR: {df['is_clicked'].mean()*100:.2f}%")

if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    generate_complex_data(config["data"]["raw_path"])