import os
os.environ["JAVA_TOOL_OPTIONS"] = "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"

from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

spark = SparkSession.builder \
    .appName("FraudDetectionPipeline") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.jars.packages", "io.delta:delta-spark_4.1_2.13:4.3.1") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Source 1 — Kaggle primary
df1 = spark.read.csv("./data/raw/fraudTrain.csv", header=True, inferSchema=True) \
           .withColumn("source_tag", lit("kaggle_primary"))

# Source 2 — MLG-ULB
df2 = spark.read.csv("./data/raw/creditcard.csv", header=True, inferSchema=True) \
           .withColumn("source_tag", lit("ulb_benchmark"))

# Source 3 — PaySim
df3 = spark.read.csv("./data/raw/PS_2017*.csv", header=True, inferSchema=True) \
           .withColumn("source_tag", lit("paysim"))

# Write each to Bronze
for df, name in [(df1,"kaggle"),(df2,"ulb"),(df3,"paysim")]:
    df.write.format("delta").mode("overwrite") \
      .partitionBy("source_tag") \
      .save(f"./data/bronze/{name}")

print("Bronze layer written successfully.")