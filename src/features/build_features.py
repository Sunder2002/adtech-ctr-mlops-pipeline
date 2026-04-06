import yaml
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType, DoubleType, IntegerType, LongType
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_pyspark_etl(raw_path: str, processed_path: str) -> None:
    logger.info("Starting Production ETL Pipeline...")
    spark = SparkSession.builder.appName("MiQ_ETL").master("local[*]").getOrCreate()
    
    try:
        df = spark.read.parquet(raw_path)

        # 1. Add Event Metadata
        df = df.withColumn("event_timestamp", F.current_timestamp().cast(TimestampType()))

        # 2. Transform with STRICT types (Matches XGBoost expectations)
        df = df.withColumn("is_peak_hour", 
                 F.when((F.col("hour_of_day") >= 17) & (F.col("hour_of_day") <= 21), 1).otherwise(0).cast(IntegerType()))
        
        df = df.withColumn("is_mobile", 
                 F.when(F.col("device_type") == "mobile", 1).otherwise(0).cast(IntegerType()))
        
        df = df.withColumn("is_shopping_site", 
                 F.when(F.col("site_category") == "shopping", 1).otherwise(0).cast(IntegerType()))

        # Ensure numeric columns are double/long
        df = df.withColumn("ad_spend_cpm", F.col("ad_spend_cpm").cast(DoubleType()))
        df = df.withColumn("hour_of_day", F.col("hour_of_day").cast(LongType()))

        # 3. Final Selection
        cols = ["user_id", "event_timestamp", "ad_spend_cpm", "hour_of_day", "is_peak_hour", "is_mobile", "is_shopping_site", "is_clicked"]
        df.select(cols).write.mode("overwrite").parquet(processed_path)
        
        logger.info(f"ETL successful. Data persisted to {processed_path}")
    except Exception as e:
        logger.error(f"ETL Failed: {e}")
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_pyspark_etl(config["data"]["raw_path"], config["data"]["processed_path"])