import yaml
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType, IntegerType, DoubleType
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_pyspark_etl(raw_path: str, processed_path: str) -> None:
    spark = SparkSession.builder.appName("MiQ_Complex_ETL").master("local[*]").getOrCreate()
    
    try:
        logger.info("Reading messy raw data...")
        df = spark.read.parquet(raw_path)

        # 1. Handling Messiness: Clean strings and Impute missing values
        # Convert device to lowercase and fill missing CPM with the median
        median_cpm = df.approxQuantile("ad_spend_cpm", [0.5], 0.01)[0]
        df = df.withColumn("device_type", F.lower(F.col("device_type"))) \
               .fillna({"ad_spend_cpm": median_cpm})

        # 2. Feature Engineering: Intent Detection (Searching inside CSV-like strings)
        # We detect if 'Electronics' or 'Automotive' exists in their history
        df = df.withColumn("intent_electronics", F.when(F.col("user_interests").contains("Electronics"), 1).otherwise(0))
        df = df.withColumn("intent_automotive", F.when(F.col("user_interests").contains("Automotive"), 1).otherwise(0))

        # 3. Contextual Features
        df = df.withColumn("is_peak_hour", F.when((F.col("hour_of_day") >= 17) & (F.col("hour_of_day") <= 21), 1).otherwise(0))
        df = df.withColumn("is_shopping_site", F.when(F.col("site_category") == "shopping", 1).otherwise(0))
        
        # 4. Final Metadata
        df = df.withColumn("event_timestamp", F.current_timestamp().cast(TimestampType()))

        # Select Features for the Model
        final_cols = [
            "user_id", "event_timestamp", "ad_spend_cpm", "historical_user_ctr",
            "is_peak_hour", "intent_electronics", "intent_automotive", 
            "is_shopping_site", "is_clicked"
        ]
        
        # Enforce strict types to prevent XGBoost crashes
        df_final = df.select([F.col(c).cast(DoubleType()) if c in ["ad_spend_cpm", "historical_user_ctr"] 
                             else F.col(c).cast(IntegerType()) if c not in ["event_timestamp"]
                             else F.col(c) for c in final_cols])

        df_final.write.mode("overwrite").parquet(processed_path)
        logger.info("Complex ETL successful.")
    finally:
        spark.stop()

if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_pyspark_etl(config["data"]["raw_path"], config["data"]["processed_path"])