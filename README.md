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
- **Compute**: Databricks (optional)  
- **Containerization**: Docker Compose  
- **Data Source**: Kaggle API  

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
- **Data Ingestion**: Download raw data from Kaggle using the API.
- **Summary Statistics**: Compute and log descriptive statistics for the raw dataset.
- **Data Validation**: Validate schema and data integrity using Pandera.
- **Preprocessing**: Clean data, encode categorical features, impute missing values, save artifacts.
- **Data Splitting**: Split data into train/test sets.
- **Model Training**: Train a RandomForest Regressor on calories burned prediction.
- **Model Evaluation**: Evaluate with regression metrics (MSE, MAE, R2).
- **Model Validation**: Ensure R2 >= 0.7.
- **Model Registration**: Log model and metrics to MLflow.

## 🧩 ML Use Case
- **Regression**: Predict calories burned based on age, height, weight, session duration, BPM, BMI, and gender.

## 🔧 Recent Fixes
- Switched model from RandomForestClassifier to RandomForestRegressor for continuous target.
- Added imputation for missing values in preprocessing to prevent training failures.
- Updated metrics to regression (MSE, MAE, R2) instead of classification.
- Cleaned up requirements.txt and Docker Compose files.
- Fixed data ingestion script for reliable Kaggle downloads.

## 📁 Folder Structure
- `airflow/`: Airflow project files
- `dags/`: `mlops_level1_pipeline.py` (main DAG)
- `data/`: 
  - `ingest_kaggle.py` (Kaggle ingestion script)
  - `raw/`: Raw datasets from Kaggle
- `demo_artifacts/`: Model and metrics artifacts
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