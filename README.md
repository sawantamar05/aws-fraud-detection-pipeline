# AWS Fraud Detection Pipeline

This project is a distributed, scalable fraud detection pipeline built using **AWS** and **Apache Spark**.  
It processes large-scale financial transaction data, engineers behavioral features, trains a machine learning model, and automates the full ETL + ML workflow using **AWS Step Functions** and **EMR**. :contentReference[oaicite:1]{index=1}

## Tech Stack

- AWS EMR, S3, Glue, Athena, Lambda, Step Functions, (optionally QuickSight)
- Apache Spark / PySpark
- Spark MLlib (Logistic Regression for fraud detection)
- Kaggle dataset: Financial Transactions Dataset for Fraud Detection

## Pipeline Overview

1. Raw CSV data uploaded to S3.
2. EMR / Spark jobs:
   - Data cleaning (null handling, invalid records removal).
   - Feature engineering (velocity_score, geo_anomaly_score, spending_deviation_score, etc.).
   - Save transformed data to S3 in Parquet format.
3. Glue Crawler updates the data catalog.
4. Athena queries on curated data.
5. ML pipeline in Spark:
   - VectorAssembler → Logistic Regression.
   - Train/test split (80/20).
   - Evaluation using AUC.
6. Orchestration via AWS Step Functions.

## How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/sawantamar05/aws-fraud-detection-pipeline.git
   cd aws-fraud-detection-pipeline

