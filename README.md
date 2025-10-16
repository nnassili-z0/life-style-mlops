# Life Style MLOps Project

## 🧠 Project Overview
This project demonstrates a full MLOps Level 1 pipeline using the Life Style dataset from Kaggle. It implements an end-to-end automated ML workflow for predicting calories burned based on lifestyle factors, covering data ingestion, preprocessing, model training, evaluation, and model registration with MLflow. The pipeline uses Apache Airflow for orchestration, Scikit-learn for modeling, and integrates with Databricks for advanced analytics.

## 🏗️ Architecture
The pipeline includes:
- Automated ETL with Airflow
- Data preprocessing with imputation and encoding
- Model training (RandomForest Regressor) and evaluation
- Experiment tracking and model registration with MLflow
- Containerized execution with Docker
- Integration with Databricks for scalable compute

## 🛠️ Tech Stack
- **Orchestrator**: Apache Airflow  
- **Data Processing**: Pandas, Pandera (validation)  
- **ML Framework**: Scikit-learn (RandomForest Regressor)  
- **Model Tracking**: MLflow  
- **Compute**: Databricks (serverless)  
- **Containerization**: Docker Compose  
- **Data Source**: Kaggle API  
- **Monitoring**: Prometheus + Grafana (target for metrics and dashboards)
- **Monitoring**: Prometheus + Grafana  

## 🚀 Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/nnassili-z0/life-style-mlops.git
   cd life-style-mlops
   ```
2. Set up Kaggle credentials:
   - Get your Kaggle API key from [Kaggle](https://www.kaggle.com/account)
   - Create a `.env` file with: `KAGGLE_KEY=your_api_key_here`
3. Build and start Airflow with Docker Compose:
   ```bash
   docker-compose -f docker-compose-airflow.yml up --build
   ```
4. Access Airflow UI at [localhost:8080](http://localhost:8080) (admin/admin)
5. Trigger the `mlops_level1_pipeline` DAG to run the full pipeline

## 📊 Pipeline Stages
The Airflow DAG automates the following steps:
- **Data Ingestion**: Download raw data from Kaggle using the API (runs on Docker startup).
- **Summary Statistics**: Compute and log descriptive statistics for the raw dataset.
- **Data Validation**: Validate schema and data integrity using Pandera.
- **Preprocessing**: Clean data, encode categorical features, impute missing values, save artifacts.
- **Data Splitting**: Split data into train/test sets.
- **Model Training**: Train a RandomForest Regressor on calories burned prediction.
- **Model Evaluation**: Evaluate with regression metrics (MSE, MAE, R2).
- **Model Validation**: Ensure R2 >= 0.7.
- **Model Registration**: Log model and metrics to MLflow (runs in-container).
- **Summary**: Collect all artifacts into a JSON summary for full data lineage.

## 🧩 ML Use Case
- **Regression**: Predict calories burned based on age, height, weight, session duration, BPM, BMI, and gender.
- **Performance**: Achieves R2 ~0.98 on test set with RandomForest Regressor.

## 🔧 Recent Updates
- **Artifact Organization**: All pipeline artifacts saved in timestamped subfolders (e.g., `demo_artifacts/20251016_142256/train/model_20251016_142256.pkl`) for better data lineage.
- **Summary Task**: Added final task to collect all files into `pipeline_summary_{ts}.json`.
- **MLflow Integration**: Fixed URI to `localhost:5000` with in-container server for reliable registration.
- **Docker Automation**: Data ingestion runs on container startup via `airflow-init`.
- **Successful Runs**: Pipeline completes end-to-end with model registration and artifact traceability.
- **Databricks Integration**: Connected to Databricks workspace with serverless compute. Catalog 'lifestyle_mlops_catalog' created with schemas for data storage. MLflow experiments logged to Databricks. Pipeline uploads preprocessed data to catalog tables.
- **Next Steps**: Set up Unity Catalog, deploy LLM endpoint, integrate pipeline with Databricks for scalable compute and AI explanations.

## 📁 Folder Structure
- `airflow/`: Airflow project files
- `dags/`: `mlops_level1_pipeline.py` (main DAG)
- `data/`: 
  - `ingest_kaggle.py` (Kaggle ingestion script)
  - `raw/`: Raw datasets from Kaggle
- `demo_artifacts/`: 
  - `sample_run/`: Example artifacts from a successful pipeline run (CSVs, logs, JSON summary)
  - Runtime artifacts (ignored in git)
- `docker/`: Dockerfiles for containerization
- `logs/`: Airflow logs
- `mlflow/`: MLflow tracking (when server is running)
- `notebooks/`: Databricks integration examples
- `src/`: Custom utilities (future)
- `tests/`: Unit tests (future)
- `requirements.txt`: Python dependencies
- `docker-compose-airflow.yml`: Airflow setup

## 📝 How to Run
1. Ensure `.env` has `KAGGLE_KEY`.
2. Start Airflow: `docker-compose -f docker-compose-airflow.yml up --build`
3. Trigger DAG in UI or via CLI.
4. Monitor logs; models and metrics saved to `demo_artifacts/`.
5. For MLflow UI, start server separately if needed.

## 📦 Requirements
Install with `pip install -r requirements.txt`. Key packages: airflow, pandas, scikit-learn, mlflow, pandera, requests.