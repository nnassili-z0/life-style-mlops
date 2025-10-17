from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import pandas as pd
import logging
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandera as pa
from pandera import Column, DataFrameSchema
import time
import numpy as np

DATA_PATH = "/opt/airflow/data/raw/Final_data.csv"
ARTIFACT_DIR = "/opt/airflow/demo_artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model_{ts}.pkl")

# Default arguments
default_args = {
    "owner": "naoufal",
    "start_date": datetime(2024, 1, 1),
    "retries": 3,
    "retry_delay": timedelta(minutes=1),
}

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

with DAG(
    dag_id="mlops_level1_pipeline",
    default_args=default_args,
    schedule_interval="@once",
    catchup=False,
    tags=["mlops", "level1"]
) as dag:
    def ingest_data(**context):
        import subprocess
        ts = context['execution_date'].strftime("%Y%m%d_%H%M%S")
        artifact_base = os.path.join(ARTIFACT_DIR, ts)
        ingest_dir = os.path.join(artifact_base, "ingest")
        os.makedirs(ingest_dir, exist_ok=True)
        # run the python script
        result = subprocess.run(["python", "/opt/airflow/data/ingest_kaggle.py"], capture_output=True, text=True)
        # save log
        log_path = os.path.join(ingest_dir, f"ingest_log_{ts}.txt")
        with open(log_path, 'w') as f:
            f.write("Ingest command output:\n")
            f.write(result.stdout)
            if result.stderr:
                f.write("Errors:\n")
                f.write(result.stderr)
        if result.returncode != 0:
            raise Exception(f"Ingest failed: {result.stderr}")
        logging.info(f"Data ingested, log saved to {log_path}")

    def validate_raw(**context):
        """Validate raw dataset schema."""
        logging.info("Validating raw dataset with Pandera...")
        schema = DataFrameSchema({
            "Gender": Column(str),
            "Age": Column(float),
            "Height (m)": Column(float),
            "Weight (kg)": Column(float),
            "Session_Duration (hours)": Column(float),
            "Avg_BPM": Column(float),
            "BMI": Column(float),
            "Calories": Column(float)
        })
        df = pd.read_csv(DATA_PATH)
        schema.validate(df)
        logging.info("Raw data validation passed.")
        ts = context['execution_date'].strftime("%Y%m%d_%H%M%S")
        artifact_base = os.path.join(ARTIFACT_DIR, ts)
        validate_dir = os.path.join(artifact_base, "validate")
        os.makedirs(validate_dir, exist_ok=True)
        log_path = os.path.join(validate_dir, f"validation_log_{ts}.txt")
        with open(log_path, 'w') as f:
            f.write("Raw data validation passed.\n")
        logging.info(f"Validation log saved to {log_path}")

    def preprocess(**context):
        """Preprocess data: encode, clean, and save artifact."""
        logging.info("Preprocessing data...")
        df = pd.read_csv(DATA_PATH)
        # Rename columns to match expected
        df = df.rename(columns={
            "Height (m)": "Height",
            "Weight (kg)": "Weight",
            "Session_Duration (hours)": "Duration",
            "Avg_BPM": "Heart_Rate"
        })
        df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
        # Encode all categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
        # Fill missing values with mean for numeric columns
        df = df.fillna(df.select_dtypes(include=[np.number]).mean())
        ts = context['execution_date'].strftime("%Y%m%d_%H%M%S")
        artifact_base = os.path.join(ARTIFACT_DIR, ts)
        preprocess_dir = os.path.join(artifact_base, "preprocess_task")
        os.makedirs(preprocess_dir, exist_ok=True)
        out_path = os.path.join(preprocess_dir, f"preprocessed_{ts}.csv")
        df.to_csv(out_path, index=False)
        logging.info(f"Preprocessed data saved to {out_path}")
        return out_path

    def split_data(**context):
        """Split data into train/test and save artifacts."""
        logging.info("Splitting data...")
        in_path = context['ti'].xcom_pull(task_ids='preprocess')
        df = pd.read_csv(in_path)
        X = df.drop("Calories", axis=1)
        y = df["Calories"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        ts = context['execution_date'].strftime("%Y%m%d_%H%M%S")
        artifact_base = os.path.join(ARTIFACT_DIR, ts)
        split_dir = os.path.join(artifact_base, "split")
        os.makedirs(split_dir, exist_ok=True)
        X_train_path = os.path.join(split_dir, f"X_train_{ts}.csv")
        X_test_path = os.path.join(split_dir, f"X_test_{ts}.csv")
        y_train_path = os.path.join(split_dir, f"y_train_{ts}.csv")
        y_test_path = os.path.join(split_dir, f"y_test_{ts}.csv")
        X_train.to_csv(X_train_path, index=False)
        X_test.to_csv(X_test_path, index=False)
        y_train.to_csv(y_train_path, index=False)
        y_test.to_csv(y_test_path, index=False)
        logging.info("Data split complete.")
        return {
            "X_train": X_train_path,
            "X_test": X_test_path,
            "y_train": y_train_path,
            "y_test": y_test_path
        }

    def train_model(**context):
        """Train model and save artifact."""
        logging.info("Training model...")
        paths = context['ti'].xcom_pull(task_ids='split_data')
        X_train = pd.read_csv(paths["X_train"])
        y_train = pd.read_csv(paths["y_train"])
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train.values.ravel())
        ts = context['execution_date'].strftime("%Y%m%d_%H%M%S")
        artifact_base = os.path.join(ARTIFACT_DIR, ts)
        train_dir = os.path.join(artifact_base, "train")
        os.makedirs(train_dir, exist_ok=True)
        model_path = os.path.join(train_dir, f"model_{ts}.pkl")
        joblib.dump(model, model_path)
        logging.info(f"Model training complete. Saved to {model_path}")
        return model_path

    def evaluate_model(**context):
        """Evaluate model and save metrics/artifacts."""
        logging.info("Evaluating model...")
        model_path = context['ti'].xcom_pull(task_ids='train_model')
        paths = context['ti'].xcom_pull(task_ids='split_data')
        model = joblib.load(model_path)
        X_test = pd.read_csv(paths["X_test"])
        y_test = pd.read_csv(paths["y_test"])
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        ts = context['execution_date'].strftime("%Y%m%d_%H%M%S")
        artifact_base = os.path.join(ARTIFACT_DIR, ts)
        evaluate_dir = os.path.join(artifact_base, "evaluate")
        os.makedirs(evaluate_dir, exist_ok=True)
        metrics_path = os.path.join(evaluate_dir, f"regression_metrics_{ts}.csv")
        pd.DataFrame({"MSE": [mse], "MAE": [mae], "R2": [r2]}).to_csv(metrics_path, index=False)
        logging.info(f"Model evaluation complete. MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
        return {"mse": mse, "mae": mae, "r2": r2, "metrics_path": metrics_path}

    def validate_model(**context):
        """Validate model performance."""
        logging.info("Validating model performance...")
        eval_results = context['ti'].xcom_pull(task_ids='evaluate_model')
        r2 = eval_results["r2"]
        if r2 < 0.7:
            raise ValueError(f"Model R2 score too low: {r2}")
        logging.info("Model passed validation.")
        ts = context['execution_date'].strftime("%Y%m%d_%H%M%S")
        artifact_base = os.path.join(ARTIFACT_DIR, ts)
        validate_model_dir = os.path.join(artifact_base, "validate_model_task")
        os.makedirs(validate_model_dir, exist_ok=True)
        log_path = os.path.join(validate_model_dir, f"validation_log_{ts}.txt")
        with open(log_path, 'w') as f:
            f.write("Model validation passed.\n")
        logging.info(f"Validation log saved to {log_path}")

    def register_model(**context):
        """Register model with MLflow."""
        logging.info("Registering model with MLflow...")
        model_path = context['ti'].xcom_pull(task_ids='train_model')
        eval_results = context['ti'].xcom_pull(task_ids='evaluate_model')
        mlflow.set_tracking_uri("https://dbc-935124bd-e5fd.cloud.databricks.com/api/2.0/mlflow")
        mlflow.set_experiment("life-style-mlops")
        with mlflow.start_run():
            model = joblib.load(model_path)
            mlflow.log_artifact(model_path, artifact_path="model")
            mlflow.log_artifact(eval_results["metrics_path"], artifact_path="metrics")
            mlflow.log_metric("mse", eval_results["mse"])
            mlflow.log_metric("mae", eval_results["mae"])
            mlflow.log_metric("r2", eval_results["r2"])
            mlflow.sklearn.log_model(model, "model")
        logging.info("Model registered with MLflow.")
        ts = context['execution_date'].strftime("%Y%m%d_%H%M%S")
        artifact_base = os.path.join(ARTIFACT_DIR, ts)
        register_dir = os.path.join(artifact_base, "register")
        os.makedirs(register_dir, exist_ok=True)
        log_path = os.path.join(register_dir, f"registration_log_{ts}.txt")
        with open(log_path, 'w') as f:
            f.write("Model registered with MLflow.\n")
        logging.info(f"Registration log saved to {log_path}")

    def upload_to_databricks(**context):
        """Upload preprocessed data to Databricks catalog."""
        import requests
        import base64
        import gzip
        logging.info("Uploading data to Databricks...")
        in_path = context['ti'].xcom_pull(task_ids='preprocess')
        with open(in_path, 'rb') as f:
            content = f.read()
        # Compress the content
        compressed = gzip.compress(content)
        encoded = base64.b64encode(compressed).decode('utf-8')
        headers = {'Authorization': f'Bearer {os.environ["DATABRICKS_TOKEN"]}'}
        ts = context['execution_date'].strftime("%Y%m%d_%H%M%S")
        dbfs_path = f"/lifestyle_mlops/preprocessed_{ts}.csv.gz"
        put_response = requests.post('https://dbc-935124bd-e5fd.cloud.databricks.com/api/2.0/dbfs/put', headers=headers, json={"path": dbfs_path, "contents": encoded, "overwrite": True}, verify=False)
        if put_response.status_code != 200:
            raise Exception(f"Failed to upload to DBFS: {put_response.text}")
        # Create table
        query = f"CREATE TABLE IF NOT EXISTS lifestyle_mlops_catalog.processed.preprocessed_data USING DELTA AS SELECT * FROM csv.`dbfs:{dbfs_path}`"
        sql_response = requests.post('https://dbc-935124bd-e5fd.cloud.databricks.com/api/2.0/sql/statements', headers=headers, json={"warehouse_id": "7bb142c4f4ff862e", "query": {"query_text": query}}, verify=False)
        if sql_response.status_code != 200:
            raise Exception(f"Failed to create table: {sql_response.text}")
        logging.info("Data uploaded to Databricks catalog.")

    def create_summary(**context):
        ts = context['execution_date'].strftime("%Y%m%d_%H%M%S")
        artifact_base = os.path.join(ARTIFACT_DIR, ts)
        summary_dir = os.path.join(artifact_base, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        # collect all files
        summary_data = {}
        for task in ["ingest", "summary_stats", "validate", "preprocess_task", "split", "train", "evaluate", "validate_model_task", "register"]:
            task_dir = os.path.join(artifact_base, task)
            if os.path.exists(task_dir):
                files = os.listdir(task_dir)
                summary_data[task] = files
        summary_path = os.path.join(summary_dir, f"pipeline_summary_{ts}.json")
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=4)
        logging.info(f"Pipeline summary saved to {summary_path}")

    def summary_statistics(**context):
        """Compute and log summary/descriptive statistics for raw data."""
        import pandas as pd
        import logging
        df = pd.read_csv(DATA_PATH)
        stats = df.describe(include='all')
        logging.info(f"Summary statistics:\n{stats}")
        # Save to artifact dir for traceability
        ts = context['execution_date'].strftime("%Y%m%d_%H%M%S")
        artifact_base = os.path.join(ARTIFACT_DIR, ts)
        summary_dir = os.path.join(artifact_base, "summary_stats")
        os.makedirs(summary_dir, exist_ok=True)
        stats_path = os.path.join(summary_dir, f"summary_stats_{ts}.csv")
        stats.to_csv(stats_path)
        logging.info(f"Summary statistics saved to {stats_path}")
        # Optionally, log feature types and missing values
        logging.info(f"Feature types: {df.dtypes}")
        missing = df.isnull().sum()
        logging.info(f"Missing values per column:\n{missing}")
        missing_path = os.path.join(summary_dir, f"missing_values_{ts}.csv")
        missing.to_csv(missing_path)
        logging.info(f"Missing values saved to {missing_path}")
        return stats_path

    # DAG tasks
    ingest = PythonOperator(task_id="ingest_data", python_callable=ingest_data)
    validate = PythonOperator(task_id="validate_raw", python_callable=validate_raw)
    preprocess_task = PythonOperator(task_id="preprocess", python_callable=preprocess)
    upload = PythonOperator(task_id="upload_to_databricks", python_callable=upload_to_databricks)
    split = PythonOperator(task_id="split_data", python_callable=split_data)
    train = PythonOperator(task_id="train_model", python_callable=train_model)
    evaluate = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)
    validate_model_task = PythonOperator(task_id="validate_model", python_callable=validate_model)
    register = PythonOperator(task_id="register_model", python_callable=register_model)
    summary_stats = PythonOperator(task_id="summary_statistics", python_callable=summary_statistics)
    summary = PythonOperator(task_id="create_summary", python_callable=create_summary)

    # Data lineage: ingest -> summary_stats -> validate -> preprocess -> upload -> split -> train -> evaluate -> validate_model -> register -> summary
    ingest >> summary_stats >> validate >> preprocess_task >> upload >> split >> train >> evaluate >> validate_model_task >> register >> summary
