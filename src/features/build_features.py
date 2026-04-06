import yaml
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_pyspark_etl(raw_path: str, processed_path: str) -> None:
    """Executes distributed ETL pipeline using Apache Spark."""
    logger.info("Initializing PySpark Session...")
    spark = SparkSession.builder \
        .appName("AdTech_Feature_Engineering") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    logger.info(f"Reading raw data from {raw_path}...")
    df = spark.read.parquet(raw_path)

    logger.info("Extracting features and applying transformations...")
    # 1. Feature Engineering: Time of day segmentation
    df = df.withColumn("is_peak_hour", F.when((F.col("hour_of_day") >= 17) & (F.col("hour_of_day") <= 21), 1).otherwise(0))
    
    # 2. Encoding categorical features using Spark SQL expressions
    df = df.withColumn("is_mobile", F.when(F.col("device_type") == "mobile", 1).otherwise(0))
    df = df.withColumn("is_shopping_site", F.when(F.col("site_category") == "shopping", 1).otherwise(0))

    # Select final features
    final_df = df.select(
        "ad_spend_cpm", "hour_of_day", "is_peak_hour", 
        "is_mobile", "is_shopping_site", "is_clicked"
    )

    logger.info(f"Writing processed features to {processed_path}...")
    final_df.write.mode("overwrite").parquet(processed_path)
    logger.info("PySpark ETL complete.")
    spark.stop()

if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_pyspark_etl(config["data"]["raw_path"], config["data"]["processed_path"])