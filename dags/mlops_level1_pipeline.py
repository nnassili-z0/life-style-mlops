from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import logging
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import pandera as pa
from pandera import Column, DataFrameSchema

# Paths
DATA_PATH = "data/raw/Final_data.csv"
MODEL_PATH = "src/models/model.pkl"

# Default arguments
default_args = {
    "owner": "naoufal",
    "start_date": datetime(2024, 1, 1),
    "retries": 3,
    "retry_delay": timedelta(minutes=1),
}

# DAG definition
with DAG(
    dag_id="mlops_level1_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["mlops", "level1"]
) as dag:

    def validate_data():
        logging.info("Validating dataset with Pandera...")
        schema = DataFrameSchema({
            "Gender": Column(str),
            "Age": Column(int),
            "Height": Column(float),
            "Weight": Column(float),
            "Duration": Column(int),
            "Heart_Rate": Column(int),
            "Body_Temp": Column(float),
            "BMI": Column(float),
            "Calories": Column(float)
        })
        df = pd.read_csv(DATA_PATH)
        schema.validate(df)
        logging.info("Data validation passed.")

    def prepare_data():
        logging.info("Preparing data...")
        df = pd.read_csv(DATA_PATH)
        df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
        X = df.drop("Calories", axis=1)
        y = df["Calories"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        os.makedirs("demo_artifacts", exist_ok=True)
        X_train.to_csv("demo_artifacts/X_train.csv", index=False)
        X_test.to_csv("demo_artifacts/X_test.csv", index=False)
        y_train.to_csv("demo_artifacts/y_train.csv", index=False)
        y_test.to_csv("demo_artifacts/y_test.csv", index=False)
        logging.info("Data preparation complete.")

    def train_model():
        logging.info("Training model...")
        X_train = pd.read_csv("demo_artifacts/X_train.csv")
        y_train = pd.read_csv("demo_artifacts/y_train.csv")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train.values.ravel())
        joblib.dump(model, MODEL_PATH)
        logging.info("Model training complete.")

    def evaluate_model():
        logging.info("Evaluating model...")
        model = joblib.load(MODEL_PATH)
        X_test = pd.read_csv("demo_artifacts/X_test.csv")
        y_test = pd.read_csv("demo_artifacts/y_test.csv")
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = f1_score(y_test, y_pred, average="weighted")
        pd.DataFrame(report).to_csv("demo_artifacts/classification_report.csv")
        with open("demo_artifacts/f1_score.txt", "w") as f:
            f.write(str(f1))
        logging.info(f"Model evaluation complete. F1 Score: {f1}")

    def validate_model():
        logging.info("Validating model performance...")
        with open("demo_artifacts/f1_score.txt", "r") as f:
            f1 = float(f.read())
        if f1 < 0.7:
            raise ValueError(f"Model F1 score too low: {f1}")
        logging.info("Model passed validation.")

    def register_model():
        logging.info("Registering model with MLflow...")
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("life-style-mlops")
        with mlflow.start_run():
            model = joblib.load(MODEL_PATH)
            mlflow.log_artifact(MODEL_PATH, artifact_path="model")
            mlflow.log_artifact("demo_artifacts/classification_report.csv", artifact_path="report")
            mlflow.log_metric("f1_score", float(open("demo_artifacts/f1_score.txt").read()))
            mlflow.sklearn.log_model(model, "model")
        logging.info("Model registered with MLflow.")

    # Task definitions
    from airflow.operators.bash import BashOperator

    t0 = BashOperator(
        task_id="download_dataset",
        bash_command="python /opt/airflow/data/ingest_kaggle.py",
    )

    t1 = PythonOperator(task_id="validate_data", python_callable=validate_data)
    t2 = PythonOperator(task_id="prepare_data", python_callable=prepare_data)
    t3 = PythonOperator(task_id="train_model", python_callable=train_model)
    t4 = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)
    t5 = PythonOperator(task_id="validate_model", python_callable=validate_model)
    t6 = PythonOperator(task_id="register_model", python_callable=register_model)

    # Task dependencies
    t1 >> t2 >> t3 >> t4 >> t5 >> t6
