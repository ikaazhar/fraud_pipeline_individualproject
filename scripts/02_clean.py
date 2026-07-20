import os
os.environ["JAVA_TOOL_OPTIONS"] = "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType
from pyspark.sql import SparkSession

# Optimize Spark configurations to manage memory safely on a local machine
spark = SparkSession.builder \
    .appName("FraudDetectionPipeline") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.jars.packages", "io.delta:delta-spark_4.1_2.13:4.3.1") \
    .config("spark.driver.memory", "4g") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.sql.shuffle.partitions", "10") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Shared time anchor
ANCHOR_TIME = F.lit("2019-01-01 00:00:00").cast("timestamp")

# Map Source 1 — Kaggle Primary
df1 = spark.read.format("delta").load("./data/bronze/kaggle")
df1 = df1.select(
    F.col("trans_num").alias("transaction_id"),
    F.to_timestamp("trans_date_trans_time").alias("timestamp"),
    F.col("amt").cast(DoubleType()).alias("amount"),
    F.col("category").alias("transaction_type"),
    F.col("is_fraud").alias("fraud_label"),
    F.col("source_tag"),
    F.col("lat"), F.col("long"),
    F.col("merch_lat"), F.col("merch_long")
)

# Map Source 2 — MLG-ULB
df2 = spark.read.format("delta").load("./data/bronze/ulb")
df2 = df2.select(
    F.expr("uuid()").alias("transaction_id"),
    F.timestamp_add("second", F.col("Time").cast("long"), ANCHOR_TIME).alias("timestamp"),
    F.col("Amount").cast(DoubleType()).alias("amount"),
    F.lit("CREDIT_CARD").alias("transaction_type"),
    F.col("Class").alias("fraud_label"),
    F.col("source_tag")
)

# Map Source 3 — PaySim
df3 = spark.read.format("delta").load("./data/bronze/paysim")
df3 = df3.select(
    F.concat(F.col("step").cast(StringType()), F.col("nameOrig")).alias("transaction_id"),
    F.timestamp_add("hour", F.col("step").cast("long"), ANCHOR_TIME).alias("timestamp"),
    F.col("amount").cast(DoubleType()).alias("amount"),
    F.col("type").alias("transaction_type"),
    F.col("isFraud").alias("fraud_label"),
    F.col("source_tag")
)

# Union all datasets safely
merged = df1.unionByName(df2, allowMissingColumns=True) \
            .unionByName(df3, allowMissingColumns=True)

# Impute missing values directly
merged = merged.fillna({"amount": 0.0, "transaction_type": "UNKNOWN"})

# Deduplicate row entries
merged = merged.dropDuplicates(["transaction_id", "timestamp"])

# FIXED: Replaced invalid nested syntax with sequential DataFrame transformations
merged = merged.withColumn("amount", F.col("amount").cast(DoubleType()))
merged = merged.withColumn("is_outlier", F.when(F.col("amount") > 50000.0, 1).otherwise(0))

# Write final output table to the Silver layer
merged.write.format("delta").mode("overwrite").save("./data/silver/")

print("Silver layer processed and written completely!")