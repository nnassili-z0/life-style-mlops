# Life Style MLOps Project

## 🧠 Project Overview
This project demonstrates a full MLOps Level 1 pipeline using the Life Style dataset from Kaggle. It covers automated data ingestion, preprocessing, model training, experiment tracking, CI/CD, deployment, and monitoring — all built with open-source tools and Databricks integration.

## 🏗️ Architecture
The pipeline includes:
- Automated ETL with Airflow
- Experiment tracking with MLflow
- CI/CD with GitLab
- Model registry and deployment via Databricks or Kubernetes
- Monitoring with Prometheus and Grafana

## 🛠️ Tech Stack
- **Version Control**: GitLab  
- **CI/CD**: GitLab CI/CD  
- **Orchestrator**: Apache Airflow  
- **Model Registry**: MLflow  
- **Compute**: Databricks  
- **Serving**: Kubernetes / Databricks  
- **Monitoring**: Prometheus + Grafana  
- **Container Registry**: DockerHub  
- **Feature Store**: Optional (Feast or Databricks Feature Store)

## 🚀 Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/nnassili-z0/life-style-mlops.git
   cd life-style-mlops
   ```
2. Copy the example environment file and update credentials:
   ```bash
   cp .env.example .env
   ```
3. Build and start Airflow with Docker Compose:
   ```bash
   docker-compose -f docker-compose-airflow.yml up --build
   ```
4. Access Airflow UI at [localhost:8080](http://localhost:8080) (admin/admin)
5. Trigger the `mlops_level1_pipeline` DAG to run the full pipeline

## 📊 Pipeline Stages
The Airflow DAG automates the following steps:
- **Data Ingestion**: Download raw data from Kaggle.
- **Summary Statistics**: Compute and log descriptive statistics for the raw dataset (mean, std, min, max, missing values, feature types).
- **Data Validation**: Validate schema and data integrity using Pandera.
- **Preprocessing**: Clean and encode features, save artifacts with timestamps.
- **Data Splitting**: Split data into train/test sets, save splits as artifacts.
- **Model Training**: Train a RandomForest model, save model artifact.
- **Model Evaluation**: Evaluate model performance (classification report, F1-score), save metrics.
- **Model Validation**: Ensure model meets minimum performance criteria.
- **Model Registration**: Register model and metrics in MLflow for traceability.

## 🏗️ Level 1 MLOps Architecture
This project implements a Level 1 MLOps architecture:
- End-to-end automation of the ML lifecycle (see attached architecture diagram)
- Reproducibility and traceability with MLflow
- Scalable orchestration using Airflow and Docker
- CI/CD integration (GitLab pipelines)
- Monitoring and observability (Prometheus, Grafana planned)
- Containerization and future Kubernetes support
- Feature engineering and transformation pipelines (DBT planned)
- Model explainability, A/B testing, and feature store integration (future)

## 🧩 ML Use Cases
- Classification: Predict lifestyle categories
- Regression: Estimate continuous outcomes
- Clustering: Group individuals for recommendations
- Deep Learning: For expanded datasets
- Anomaly Detection: Identify outliers

## 🔜 Next Steps
- Implement feature engineering and model training pipelines
- Integrate MLflow tracking and model registry
- Set up automated CI/CD workflows
- Deploy models via REST API or batch inference
- Add monitoring dashboards and alerting
- Transition to Kubernetes for scalability


## 📁 Folder Structure
- `airflow/`: Airflow project files
- `dags/`: Contains `mlops_level1_pipeline.py` (main DAG), `mlops_level1_pipeline.pyc`, and `__pycache__`
- `data/`: 
   - `ingest_kaggle.py` (Kaggle ingestion script)
   - `raw/`: `Final_data.csv`, `expanded_fitness_data.csv`, `meal_metadata.csv`
   - `processed/`: (future processed datasets)
- `demo_artifacts/`: Placeholder for model and metrics artifacts
- `docker/`: `Dockerfile`, `Dockerfile_new` for containerization
- `logs/`: Airflow and pipeline logs, including DAG run folders
- `mlflow/`: MLflow tracking artifacts (future use)
- `notebooks/`: 
   - `databricks_catalog_schema_test.ipynb` (Databricks Connect, Delta table test)
   - `databricks_connect_mlflow_example.ipynb` (MLflow/Databricks usage example)
- `out/`: (empty)
- `plugins/`: (empty, for custom Airflow plugins)
- `resources/`: (empty, for future resources)
- `src/`: (empty, for custom operators/utilities)
- `tests/`: (empty, for unit/integration tests)
- `typings/`: (empty, for type stubs)

## 🔒 Security
- Sensitive credentials are managed via `.env` (not committed)
- Example `.env.example` provided for safe sharing
- Fernet key used for Airflow metadata encryption


## 📝 How to Run
1. Build and start Airflow with Docker Compose:
   ```bash
   docker-compose -f docker-compose-airflow.yml up --build
   ```
2. Access Airflow UI at [localhost:8080](http://localhost:8080) (admin/admin)
3. Trigger the `mlops_level1_pipeline` DAG to run the full pipeline
4. Data ingestion is handled by `data/ingest_kaggle.py` (downloads and extracts Kaggle dataset to `data/raw/Final_data.csv`).
5. The pipeline performs validation, preprocessing, splitting, model training, evaluation, and MLflow registration.
6. For Databricks integration, see `notebooks/databricks_catalog_schema_test.ipynb` and `databricks_connect_mlflow_example.ipynb`.


## 📦 Requirements
All dependencies are listed in `requirements.txt`. Install with:
```bash
pip install -r requirements.txt
```
Main packages: apache-airflow, pandas, pandera, scikit-learn, mlflow, requests, python-dotenv, joblib

## 🖼️ Architecture Diagram
See the attached PNG for the full Level 1 MLOps architecture.