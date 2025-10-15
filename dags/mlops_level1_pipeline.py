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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import pandera as pa
from pandera import Column, DataFrameSchema
import time

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
    def validate_raw():
        """Validate raw dataset schema."""
        logging.info("Validating raw dataset with Pandera...")
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
        logging.info("Raw data validation passed.")

    def preprocess():
        """Preprocess data: encode, clean, and save artifact."""
        logging.info("Preprocessing data...")
        df = pd.read_csv(DATA_PATH)
        df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
        ts = get_timestamp()
        out_path = os.path.join(ARTIFACT_DIR, f"preprocessed_{ts}.csv")
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
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
        ts = get_timestamp()
        X_train_path = os.path.join(ARTIFACT_DIR, f"X_train_{ts}.csv")
        X_test_path = os.path.join(ARTIFACT_DIR, f"X_test_{ts}.csv")
        y_train_path = os.path.join(ARTIFACT_DIR, f"y_train_{ts}.csv")
        y_test_path = os.path.join(ARTIFACT_DIR, f"y_test_{ts}.csv")
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
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train.values.ravel())
        ts = get_timestamp()
        model_path = os.path.join(ARTIFACT_DIR, f"model_{ts}.pkl")
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
        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = f1_score(y_test, y_pred, average="weighted")
        ts = get_timestamp()
        report_path = os.path.join(ARTIFACT_DIR, f"classification_report_{ts}.csv")
        f1_path = os.path.join(ARTIFACT_DIR, f"f1_score_{ts}.txt")
        pd.DataFrame(report).to_csv(report_path)
        with open(f1_path, "w") as f:
            f.write(str(f1))
        logging.info(f"Model evaluation complete. F1 Score: {f1}")
        return {"f1": f1, "f1_path": f1_path, "report_path": report_path}

    def validate_model(**context):
        """Validate model performance."""
        logging.info("Validating model performance...")
        eval_results = context['ti'].xcom_pull(task_ids='evaluate_model')
        f1 = eval_results["f1"]
        if f1 < 0.7:
            raise ValueError(f"Model F1 score too low: {f1}")
        logging.info("Model passed validation.")

    def register_model(**context):
        """Register model with MLflow."""
        logging.info("Registering model with MLflow...")
        model_path = context['ti'].xcom_pull(task_ids='train_model')
        eval_results = context['ti'].xcom_pull(task_ids='evaluate_model')
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("life-style-mlops")
        with mlflow.start_run():
            model = joblib.load(model_path)
            mlflow.log_artifact(model_path, artifact_path="model")
            mlflow.log_artifact(eval_results["report_path"], artifact_path="report")
            mlflow.log_metric("f1_score", eval_results["f1"])
            mlflow.sklearn.log_model(model, "model")
        logging.info("Model registered with MLflow.")

    def summary_statistics():
        """Compute and log summary/descriptive statistics for raw data."""
        import pandas as pd
        import logging
        df = pd.read_csv(DATA_PATH)
        stats = df.describe(include='all')
        logging.info(f"Summary statistics:\n{stats}")
        # Save to artifact dir for traceability
        ts = get_timestamp()
        stats_path = os.path.join(ARTIFACT_DIR, f"summary_stats_{ts}.csv")
        stats.to_csv(stats_path)
        logging.info(f"Summary statistics saved to {stats_path}")
        # Optionally, log feature types and missing values
        logging.info(f"Feature types: {df.dtypes}")
        missing = df.isnull().sum()
        logging.info(f"Missing values per column:\n{missing}")
        missing_path = os.path.join(ARTIFACT_DIR, f"missing_values_{ts}.csv")
        missing.to_csv(missing_path)
        logging.info(f"Missing values saved to {missing_path}")
        return stats_path

    # DAG tasks
    ingest = BashOperator(
        task_id="ingest_data",
        bash_command="python /opt/airflow/data/ingest_kaggle.py",
    )
    validate = PythonOperator(task_id="validate_raw", python_callable=validate_raw)
    preprocess_task = PythonOperator(task_id="preprocess", python_callable=preprocess)
    split = PythonOperator(task_id="split_data", python_callable=split_data)
    train = PythonOperator(task_id="train_model", python_callable=train_model)
    evaluate = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)
    validate_model_task = PythonOperator(task_id="validate_model", python_callable=validate_model)
    register = PythonOperator(task_id="register_model", python_callable=register_model)
    summary_stats = PythonOperator(task_id="summary_statistics", python_callable=summary_statistics)

    # Data lineage: ingest -> summary_stats -> validate -> preprocess -> split -> train -> evaluate -> validate_model -> register
    ingest >> summary_stats >> validate >> preprocess_task >> split >> train >> evaluate >> validate_model_task >> register
