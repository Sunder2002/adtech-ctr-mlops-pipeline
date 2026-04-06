from datetime import timedelta
import pandas as pd
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64

# 1. Define the 'User' as an Entity (Identity)
user = Entity(name="user_id", join_keys=["user_id"])

# 2. Define where the data comes from (Our PySpark output)
ad_source = FileSource(
    path="/workspaces/adtech-ctr-mlops-pipeline/data/processed/features.parquet",
    event_timestamp_column="event_timestamp", # Real world needs timestamps!
)

# 3. Create the Feature View (How the model sees the data)
ad_ctr_fv = FeatureView(
    name="ad_ctr_features",
    entities=[user],
    ttl=timedelta(days=1),
    schema=[
        Field(name="ad_spend_cpm", dtype=Float32),
        Field(name="is_peak_hour", dtype=Int64),
        Field(name="is_mobile", dtype=Int64),
    ],
    online=True,
    source=ad_source,
)