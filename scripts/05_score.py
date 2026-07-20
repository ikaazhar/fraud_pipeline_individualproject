import os
os.environ["JAVA_TOOL_OPTIONS"] = "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col
from pyspark.ml.functions import vector_to_array

spark = SparkSession.builder \
    .appName("FraudScoringEngine") \
    .config("spark.jars.packages", "io.delta:delta-spark_4.1_2.13:4.3.1") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "10") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("Loading compiled model pipeline...")
model = PipelineModel.load("./models/fraud_rf_pipeline")

# FIXED: Read from the feature-rich Gold layer instead of the un-engineered Silver layer
# This ensures columns like 'tx_velocity' and 'card_merchant_dist' exist for the model's VectorAssembler.
print("Reading target transaction records for inference...")
new_txns = spark.read.format("delta").load("./data/gold/")

print("Generating model predictions and calculating fraud probabilities...")
scored = model.transform(new_txns)

# Extract individual category indices cleanly from Spark's underlying probability vector
scored = scored.withColumn("fraud_probability", vector_to_array(col("probability"))[1])

# Select key informative identifier columns to save out
final_output = scored.select(
    "transaction_id", 
    "timestamp", 
    "amount", 
    "transaction_type", 
    "source_tag", 
    "fraud_probability", 
    "prediction"
)

# Write all outputs to their final directories
final_output.write.format("delta").mode("overwrite").save("./data/gold/fraud_scores")

# Extract and isolate critical high-priority alert queues (Fraud prediction = 1, confidence >= 75%)
alerts = final_output.filter((col("prediction") == 1) & (col("fraud_probability") >= 0.75))
alerts.write.format("delta").mode("overwrite").save("./data/gold/fraud_alerts")

print("-" * 50)
print(f"Scoring stage complete!")
print(f"Total Records Processed : {final_output.count()}")
print(f"High-Risk Alerts Flagged: {alerts.count()}")
print("-" * 50)