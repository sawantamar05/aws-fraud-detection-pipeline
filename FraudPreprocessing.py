from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("FraudPreprocessing").getOrCreate()

# Input and output paths
input_path = "s3://projectamar/fraud-detection/input/financial_fraud_detection_dataset.csv"
output_path = "s3://projectamar/fraud-detection/output/processed_data/"

# Read CSV
df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)

# Basic transformation: remove nulls, filter negative transactions
df_clean = df.dropna().filter(col("amount") > 0)

# Cast fraud label to integer (if needed)
df_clean = df_clean.withColumn("is_fraud", col("is_fraud").cast("int"))

# Save as Parquet
df_clean.write.mode("overwrite").parquet(output_path)

spark.stop()
