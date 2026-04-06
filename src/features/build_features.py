import yaml
import sys
import os
import logging
import traceback
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType, DoubleType, IntegerType
from src.utils.logger import get_logger

# Initialize professional-grade logging
logger = get_logger(__name__)

class FeaturePipeline:
    """
    Encapsulates the logic for distributed feature engineering.
    Designed for MiQ's requirement for scalable, production-ready ETL.
    """
    def __init__(self, raw_path: str, processed_path: str):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.spark = self._init_spark()

    def _init_spark(self) -> SparkSession:
        logger.info("Initializing Enterprise-Scale PySpark Session...")
        return SparkSession.builder \
            .appName("MiQ_AdTech_Production_ETL") \
            .master("local[*]") \
            .config("spark.driver.memory", "2g") \
            .config("spark.sql.parquet.compression.codec", "snappy") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()

    def run(self):
        try:
            if not os.path.exists(self.raw_path):
                logger.error(f"IO_ERROR: Raw source missing at {self.raw_path}")
                sys.exit(1)

            logger.info(f"Ingesting raw bid-stream: {self.raw_path}")
            df = self.spark.read.parquet(self.raw_path)

            # 1. Temporal Feature Engineering
            # event_timestamp is mandatory for the Feast Feature Store
            logger.info("Generating temporal signals and event timestamps...")
            df_transformed = df.withColumn(
                "event_timestamp", 
                F.current_timestamp().cast(TimestampType())
            ).withColumn(
                "is_peak_hour", 
                F.when((F.col("hour_of_day") >= 17) & (F.col("hour_of_day") <= 21), 1).otherwise(0)
            )

            # 2. Categorical Behavioral Encoding
            logger.info("Encoding high-cardinality device and site signals...")
            df_encoded = df_transformed.withColumn(
                "is_mobile", 
                F.when(F.col("device_type") == "mobile", 1).otherwise(0)
            ).withColumn(
                "is_shopping_site", 
                F.when(F.col("site_category") == "shopping", 1).otherwise(0)
            )

            # 3. Schema Enforcement for Feature Store Alignment
            final_df = df_encoded.select(
                F.col("user_id").cast(IntegerType()),
                "event_timestamp",
                F.col("ad_spend_cpm").cast(DoubleType()),
                "hour_of_day",
                "is_peak_hour",
                "is_mobile",
                "is_shopping_site",
                "is_clicked"
            )

            # 4. Atomic Partitioned Write
            logger.info(f"Persisting Snappy-compressed features to: {self.processed_path}")
            final_df.write.mode("overwrite").parquet(self.processed_path)
            
            logger.info(f"Pipeline Succeeded. Processed Events: {final_df.count()}")

        except Exception as e:
            logger.error(f"ETL_CRASH: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        finally:
            self.spark.stop()
            logger.info("Spark session decommissioned.")

if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    pipeline = FeaturePipeline(
        raw_path=config["data"]["raw_path"], 
        processed_path=config["data"]["processed_path"]
    )
    pipeline.run()