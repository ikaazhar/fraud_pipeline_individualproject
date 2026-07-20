import os
os.environ["JAVA_TOOL_OPTIONS"] = "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("FraudModelTraining") \
    .config("spark.jars.packages", "io.delta:delta-spark_4.1_2.13:4.3.1") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "10") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Load your successful Gold feature dataset
df = spark.read.format("delta").load("./data/gold/")

# OPTIMIZATION: Down-sample the massive balanced dataset for training on a local laptop
# This preserves the 50/50 balanced class distribution but keeps memory overhead safe.
df_train_sample = df.sample(withReplacement=False, fraction=0.15, seed=42)

# Train/test split (80% training, 20% validation evaluation)
train_df, test_df = df_train_sample.randomSplit([0.8, 0.2], seed=42)

# Pipeline Stages Configuration
indexer = StringIndexer(inputCol="transaction_type", outputCol="tx_type_idx", handleInvalid="keep")
encoder = OneHotEncoder(inputCols=["tx_type_idx"], outputCols=["tx_type_vec"])

# Feature schema maps exactly to your engineered Gold columns
feature_cols = [
    "amount", 
    "hour_of_day", 
    "day_of_week", 
    "is_weekend", 
    "tx_velocity", 
    "is_outlier", 
    "card_merchant_dist", 
    "tx_type_vec"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")

# Tuned Random Forest parameters optimized for local run speed and accuracy
rf = RandomForestClassifier(
    labelCol="fraud_label", 
    featuresCol="features", 
    numTrees=30,      # Slightly reduced from 50 to optimize training speeds on your CPU
    maxDepth=8, 
    seed=42
)

pipeline = Pipeline(stages=[indexer, encoder, assembler, rf])

print("Training Machine Learning Model Pipeline over balanced features...")
model = pipeline.fit(train_df)

# Save the trained Pipeline model structure safely
os.makedirs("./models", exist_ok=True)
model.write().overwrite().save("./models/fraud_rf_pipeline")
print("Model saved completely to ./models/fraud_rf_pipeline")

# Generate Evaluation Predictions
preds = model.transform(test_df)

evaluators = {
    "Precision": MulticlassClassificationEvaluator(labelCol="fraud_label", predictionCol="prediction", metricName="weightedPrecision"),
    "Recall": MulticlassClassificationEvaluator(labelCol="fraud_label", predictionCol="prediction", metricName="weightedRecall"),
    "F1-Score": MulticlassClassificationEvaluator(labelCol="fraud_label", predictionCol="prediction", metricName="f1"),
    "AUC-ROC": BinaryClassificationEvaluator(labelCol="fraud_label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
}

print("\n=== Model Evaluation Results ===")
for metric, evaluator in evaluators.items():
    try:
        score = evaluator.evaluate(preds)
        print(f"{metric:12s}: {score:.4f}")
    except Exception as e:
        print(f"Error evaluating {metric}: {str(e)}")