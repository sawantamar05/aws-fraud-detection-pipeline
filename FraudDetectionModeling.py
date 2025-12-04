sc.install_pypi_package("numpy")
sc.install_pypi_package("matplotlib")
sc.install_pypi_package("seaborn")
sc.install_pypi_package("boto3")

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import matplotlib.pyplot as plt

spark = SparkSession.builder \
    .appName('FraudDetectionModeling') \
    .getOrCreate()
    
data_path = 's3://projectamar/fraud-detection/output/processed_data/'
df = spark.read.parquet(data_path)
df.printSchema()
df.show(5)

feature_cols = [
    'amount',
    'time_since_last_transaction',
    'spending_deviation_score',
    'velocity_score',
    'geo_anomaly_score'
]
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
df_features = assembler.transform(df).select('features', 'is_fraud')

train_data, test_data = df_features.randomSplit([0.8, 0.2], seed=42feature_cols = [
    'amount',
    'time_since_last_transaction',
    'spending_deviation_score',
    'velocity_score',
    'geo_anomaly_score'
]
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
df_features = assembler.transform(df).select('features', 'is_fraud'))
print(f'Training data count: {train_data.count()}')
print(f'Testing data count: {test_data.count()}')

lr = LogisticRegression(labelCol='is_fraud', featuresCol='features')
lr_model = lr.fit(train_data)
print('Model trained successfully!')

predictions = lr_model.transform(test_data)
predictions.select('features', 'is_fraud', 'prediction', 'probability').show(5)

binary_evaluator = BinaryClassificationEvaluator(labelCol='is_fraud')
auc = binary_evaluator.evaluate(predictions)
print(f'Area Under ROC: {auc}')

pred_pd = predictions.select('prediction', 'is_fraud').toPandas()

import os
import boto3

pred_pd = predictions.select('prediction', 'is_fraud').toPandas()
fraud_pred_counts = pred_pd.groupby(['is_fraud', 'prediction']).size().unstack().fillna(0)
fraud_pred_counts.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Fraud vs Non-Fraud Predictions')
plt.xlabel('Actual Label')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Predicted')
plt.tight_layout()
os.makedirs('/tmp', exist_ok=True)
plt.savefig('/tmp/fraud_prediction_plot.png')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import boto3


sample_df = df.select(
    "amount", "velocity_score", "geo_anomaly_score",
    "time_since_last_transaction", "spending_deviation_score"
).dropna().sample(False, 0.01).toPandas()


sns.pairplot(sample_df)
plt.suptitle("Fraud Detection Metrics Pairwise Plot", y=1.02)
plt.savefig("/tmp/pairplot.png")


s3 = boto3.client("s3")
s3.upload_file("/tmp/pairplot.png", "projectamar", "projectamar/output/pairplot.png")

print("Pairplot saved to s3://projectamar/output/pairplot.png")

