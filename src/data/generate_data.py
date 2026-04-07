import pandas as pd
import numpy as np
import yaml
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

def generate_realistic_data(output_path: str, n_rows: int = 50000) -> None:
    logger.info(f"Generating {n_rows} rows of High-Discrimination Ad-Tech data...")
    np.random.seed(42)
    
    # 1. User Segments
    segments = ["electronics", "automotive", "general"]
    user_segments = np.random.choice(segments, n_rows, p=[0.2, 0.2, 0.6])
    
    # 2. Raw Features (Including the missing device_type and site_category)
    data = {
        "user_id": np.random.randint(100000, 999999, n_rows),
        "segment": user_segments,
        "device_type": np.random.choice(["MOBILE", "mobile", "Desktop", "tablet"], n_rows),
        "site_category": np.random.choice(["news", "shopping", "social", "video"], n_rows),
        "hour_of_day": np.random.randint(0, 24, n_rows),
        "ad_spend_cpm": np.random.uniform(1.0, 5.0, n_rows)
    }
    df = pd.DataFrame(data)

    # 3. Inject Missing Values (Real-world noise)
    null_mask = np.random.random(n_rows) < 0.05
    df.loc[null_mask, "ad_spend_cpm"] = np.nan

    # 4. Behavioral Logic
    df["historical_user_ctr"] = 0.001 
    df.loc[df["segment"] == "electronics", "historical_user_ctr"] = np.random.uniform(0.05, 0.15, sum(df["segment"] == "electronics"))
    df.loc[df["segment"] == "automotive", "historical_user_ctr"] = np.random.uniform(0.03, 0.10, sum(df["segment"] == "automotive"))

    # Target Generation (The Click)
    click_prob = df["historical_user_ctr"] * 0.7
    click_prob += np.where((df["hour_of_day"] >= 17) & (df["hour_of_day"] <= 21), 0.02, 0)
    df["is_clicked"] = (np.random.random(n_rows) < click_prob).astype(int)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Data Generation Complete.")

if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    generate_realistic_data(config["data"]["raw_path"])