from pyspark.sql import SparkSession
import os
os.environ["JAVA_TOOL_OPTIONS"] = "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"

spark = SparkSession.builder \
    .appName("SchemaInspector") \
    .config("spark.jars.packages", "io.delta:delta-spark_4.1_2.13:4.3.1") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Bronze
print("=== BRONZE (Kaggle) ===")
spark.read.format("delta").load("./data/bronze/kaggle").printSchema()

print("=== BRONZE (ULB) ===")
spark.read.format("delta").load("./data/bronze/ulb").printSchema()

print("=== BRONZE (PaySim) ===")
spark.read.format("delta").load("./data/bronze/paysim").printSchema()

# Silver
print("=== SILVER ===")
spark.read.format("delta").load("./data/silver/").printSchema()

# Gold
print("=== GOLD ===")
spark.read.format("delta").load("./data/gold/").printSchema()

# Fraud scores output
print("=== FRAUD SCORES ===")
spark.read.format("delta").load("./data/gold/fraud_scores").printSchema()

# Fraud alerts output
print("=== FRAUD ALERTS ===")
spark.read.format("delta").load("./data/gold/fraud_alerts").printSchema()