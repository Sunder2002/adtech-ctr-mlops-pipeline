import yaml
import sys
import os
import traceback
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType, DoubleType, IntegerType
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_pyspark_etl(raw_path: str, processed_path: str) -> None:
    logger.info("Initializing Enterprise-Scale PySpark Session...")
    try:
        spark = SparkSession.builder \
            .appName("MiQ_AdTech_ETL") \
            .master("local[*]") \
            .config("spark.driver.memory", "2g") \
            .getOrCreate()

        if not os.path.exists(raw_path):
            logger.error(f"Raw data missing at: {raw_path}")
            sys.exit(1)

        df = spark.read.parquet(raw_path)

        # 1. Clean Messy Data
        median_cpm = df.approxQuantile("ad_spend_cpm",[0.5], 0.01)[0]
        df = df.fillna({"ad_spend_cpm": median_cpm})
        df = df.withColumn("device_type", F.lower(F.col("device_type")))

        # 2. Feature Derivation
        df = df.withColumn("event_timestamp", F.current_timestamp().cast(TimestampType()))
        df = df.withColumn("is_peak_hour", F.when((F.col("hour_of_day") >= 17) & (F.col("hour_of_day") <= 21), 1).otherwise(0))
        df = df.withColumn("is_mobile", F.when(F.col("device_type") == "mobile", 1).otherwise(0))
        df = df.withColumn("is_shopping_site", F.when(F.col("site_category") == "shopping", 1).otherwise(0))
        df = df.withColumn("intent_electronics", F.when(F.col("segment") == "electronics", 1).otherwise(0))
        df = df.withColumn("intent_automotive", F.when(F.col("segment") == "automotive", 1).otherwise(0))

        # 3. Strict Schema Selection (Must match API exactly)
        final_df = df.select(
            F.col("user_id").cast(IntegerType()),
            "event_timestamp",
            F.col("ad_spend_cpm").cast(DoubleType()),
            F.col("historical_user_ctr").cast(DoubleType()),
            F.col("is_peak_hour").cast(IntegerType()),
            F.col("is_mobile").cast(IntegerType()),
            F.col("intent_electronics").cast(IntegerType()),
            F.col("intent_automotive").cast(IntegerType()),
            F.col("is_shopping_site").cast(IntegerType()),
            F.col("is_clicked").cast(IntegerType())
        )

        final_df.write.mode("overwrite").parquet(processed_path)
        logger.info(f"ETL Successful. Processed {final_df.count()} records.")

    except Exception as e:
        logger.error(f"ETL Pipeline Failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        if 'spark' in locals():
            spark.stop()

if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_pyspark_etl(config["data"]["raw_path"], config["data"]["processed_path"])