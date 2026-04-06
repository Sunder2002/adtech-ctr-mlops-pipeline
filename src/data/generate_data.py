import pandas as pd
import numpy as np
import yaml
from src.utils.logger import get_logger

logger = get_logger(__name__)

def generate_raw_data(output_path: str, n_rows: int = 50000) -> None:
    """Generates realistic synthetic programmatic ad bid stream data."""
    logger.info(f"Generating {n_rows} rows of realistic ad data...")
    np.random.seed(42)
    
    # 1. Base Features
    device_type = np.random.choice(["mobile", "desktop", "tablet"], n_rows, p=[0.6, 0.3, 0.1])
    site_category = np.random.choice(["news", "sports", "shopping", "entertainment"], n_rows)
    hour_of_day = np.random.randint(0, 24, n_rows)
    
    # 2. Simulate Real-World Ad Spend (CPM)
    # CPM is usually higher during peak evening hours (17-21)
    base_cpm = np.random.uniform(0.5, 5.0, n_rows)
    peak_multiplier = np.where((hour_of_day >= 17) & (hour_of_day <= 21), 2.5, 1.0)
    ad_spend_cpm = base_cpm * peak_multiplier

    # 3. Simulate Realistic Click-Through Rate (CTR) Behavior
    # Start with a base CTR of 1%
    click_probability = np.full(n_rows, 0.01)
    
    # Mobile users click more often
    click_probability = np.where(device_type == "mobile", click_probability + 0.02, click_probability)
    
    # Shopping sites have higher intent
    click_probability = np.where(site_category == "shopping", click_probability + 0.03, click_probability)
    
    # Peak hours have better engagement
    click_probability = np.where((hour_of_day >= 17) & (hour_of_day <= 21), click_probability + 0.015, click_probability)
    
    # Cap probability at 99% just in case
    click_probability = np.clip(click_probability, 0.0, 0.99)
    
    # 4. Generate actual clicks based on our calculated probabilities
    random_threshold = np.random.uniform(0, 1, n_rows)
    is_clicked = (random_threshold < click_probability).astype(int)
    
    # Create DataFrame
    data = {
        "user_id": np.random.randint(10000, 99999, n_rows),
        "site_category": site_category,
        "device_type": device_type,
        "ad_spend_cpm": ad_spend_cpm.round(2),
        "hour_of_day": hour_of_day,
        "is_clicked": is_clicked
    }
    
    df = pd.DataFrame(data)
    
    # Log some real-world stats
    logger.info(f"Overall CTR: {df['is_clicked'].mean() * 100:.2f}%")
    logger.info(f"Mobile CTR: {df[df['device_type'] == 'mobile']['is_clicked'].mean() * 100:.2f}%")
    
    df.to_parquet(output_path, index=False)
    logger.info(f"Raw data successfully saved to {output_path}")

if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    generate_raw_data(config["data"]["raw_path"])