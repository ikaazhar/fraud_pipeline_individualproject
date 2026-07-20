import os
os.environ["JAVA_TOOL_OPTIONS"] = "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import TimestampType

spark = SparkSession.builder \
    .appName("FraudFeatureEngineering") \
    .config("spark.jars.packages", "io.delta:delta-spark_4.1_2.13:4.3.1") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "10") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Load Silver layer data
df = spark.read.format("delta").load("./data/silver/")

# Strict cast to TimestampType to enforce structural uniformity
df = df.withColumn("timestamp", F.col("timestamp").cast(TimestampType()))

# Time features
df = df.withColumn("hour_of_day", F.hour("timestamp")) \
       .withColumn("day_of_week", F.dayofweek("timestamp")) \
       .withColumn("is_weekend", F.when(F.col("day_of_week").isin(1,7), 1).otherwise(0))

# Transaction velocity over coordinates rounded to spatial grids
df = df.withColumn("lat_grid", F.round(F.col("lat"), 2)) \
       .withColumn("long_grid", F.round(F.col("long"), 2))

w = Window.partitionBy("lat_grid", "long_grid") \
          .orderBy(F.col("timestamp").cast("long")) \
          .rangeBetween(-3600, 0)
df = df.withColumn("tx_velocity", F.count("*").over(w))

df = df.drop("lat_grid", "long_grid")

# FIXED: Native Spark SQL Haversine Equation Implementation
# This removes the Python UDF overhead entirely and processes rows inside JVM memory.
R = 6371.0  # Earth's radius in km

# Convert degrees to radians natively
lat1_rad = F.radians(F.col("lat"))
lon1_rad = F.radians(F.col("long"))
lat2_rad = F.radians(F.col("merch_lat"))
lon2_rad = F.radians(F.col("merch_long"))

dlat = lat2_rad - lat1_rad
dlon = lon2_rad - lon1_rad

# Haversine formula component steps
a = F.pow(F.sin(dlat / 2.0), 2) + F.cos(lat1_rad) * F.cos(lat2_rad) * F.pow(F.sin(dlon / 2.0), 2)
c = 2.0 * F.asin(F.sqrt(a))
haversine_dist = R * c

# Apply calculation and fallback cleanly on null missing coordinate rows
df = df.withColumn("card_merchant_dist", F.coalesce(haversine_dist, F.lit(0.0)))

# Separate minority and majority sets cleanly
df_legit = df.filter(F.col("fraud_label") == 0)
df_fraud = df.filter(F.col("fraud_label") == 1)

# Balance data paths efficiently on local machine limits
fraud_oversampled = df_fraud.sample(withReplacement=True, fraction=5.0, seed=42)
df_balanced = df_legit.unionByName(fraud_oversampled)

# Write out optimized Gold Layer tables
df_balanced.write.format("delta").mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save("./data/gold/")

print("Gold layer feature table written successfully without Python bottlenecks!")