# Life Style MLOps Project

## 🧠 Project Overview
This project demonstrates a full MLOps Level 1 pipeline using the Life Style dataset from Kaggle. It implements an end-to-end automated ML workflow for predicting calories burned based on lifestyle factors, covering data ingestion, preprocessing, model training, evaluation, and model registration with MLflow. The pipeline uses Apache Airflow for orchestration, Scikit-learn for modeling, and integrates with Databricks for advanced analytics.

**🎯 AI Playground Available**: Complete pipeline artifacts from successful runs are now available in the `ai_playground_data/` directory, specifically prepared for AI experimentation, research, and advanced analytics. Includes 20,000 records, 400+ features, comprehensive documentation, and ready-to-use analysis templates.

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
- **Preprocessing**: Clean data, encode categorical features (Gender, etc.), impute missing values, save artifacts.
- **Data Splitting**: Split data into train/test sets with stratification.
- **Model Training**: Train optimized RandomForest Regressor on calories burned prediction with data sampling.
- **Model Evaluation**: Evaluate with regression metrics (MSE, MAE, R²).
- **Model Validation**: Ensure R² >= 0.5 for regression performance.
- **Model Registration**: Log model and metrics to MLflow with graceful error handling.
- **Artifact Archiving**: Archive all pipeline artifacts to timestamped directories for long-term storage.
- **Databricks Push**: Push processed data to Databricks SQL warehouse (optional, handles missing connector gracefully).

## 🧩 ML Use Case
- **Regression**: Predict calories burned based on age, height, weight, session duration, BPM, BMI, and gender.
- **Performance**: Achieves MSE ~0.08, R² ~0.92 on test set with optimized RandomForest Regressor.
- **Training Optimization**: Data sampling and reduced complexity enable fast training (~1 minute) while maintaining accuracy.

## 🏗️ Pipeline Features
- **Comprehensive Categorical Encoding**: Handles all categorical variables (Gender, exercise types, etc.) with proper encoding strategies.
- **Artifact Archiving**: Automatic archiving to `/opt/airflow/archives/pipeline_run_YYYYMMDD_HHMMSS/` with complete manifest.
- **Error Resilience**: Graceful handling of optional services (MLflow, Databricks) - pipeline succeeds even when services are unavailable.
- **Production Ready**: Optimized for reliability with timeout handling, retry logic, and comprehensive logging.

## 🤖 AI Playground Data

The `ai_playground_data/` directory contains complete artifacts from successful pipeline runs, specifically curated for AI research and experimentation:

### 📊 Dataset Features
- **20,000 complete records** with no missing values
- **400+ engineered features** including demographics, fitness metrics, nutrition data, and exercise categories
- **Train/test splits** (80/20) with proper stratification
- **Model artifacts** and performance metrics (R² = 0.98)

### 📚 AI Resources Included
- **`README_AI_PLAYGROUND.md`**: Comprehensive guide for AI experimentation
- **`DATA_DICTIONARY.md`**: Complete documentation of all 400+ features
- **`ANALYSIS_INSIGHTS.md`**: Pre-computed statistical insights and correlations
- **`AI_ANALYSIS_TEMPLATES.md`**: Ready-to-use queries, prompts, and ML templates
- **`dataset_summary.json`**: Structured JSON summary for API integration

### 🎯 AI Applications
- **RAG Systems**: Build fitness recommendation assistants
- **Predictive Models**: Calorie prediction, health risk assessment, workout recommendations
- **NLP Integration**: Fitness chatbots and meal planning systems
- **Research**: Academic studies on lifestyle and fitness correlations

### 🚀 Quick Start for AI Experiments
```python
# Load the dataset
import pandas as pd
df = pd.read_csv('ai_playground_data/lifestyle_mlops_complete_run/preprocessed_20251016_142256.csv')
print(f"Dataset shape: {df.shape}")  # (20000, 400+)

# Load analysis insights
import json
with open('ai_playground_data/lifestyle_mlops_complete_run/dataset_summary.json') as f:
    insights = json.load(f)
```

**📖 Full Documentation**: See `ai_playground_data/lifestyle_mlops_complete_run/README_AI_PLAYGROUND.md` for complete AI experimentation guide.

## 🔧 Recent Updates
- **Enhanced Pipeline Architecture**: Complete MLOps Level 1 pipeline with archiving and cloud integration features.
- **Improved Categorical Encoding**: Comprehensive encoding for all string columns (Gender, exercise types, etc.) to prevent training failures.
- **Optimized Model Training**: RandomForest Regressor with data sampling (max 5000 rows) and reduced estimators (50) for faster, reliable training.
- **Robust Error Handling**: MLflow registration with connection timeouts and graceful degradation when services unavailable.
- **Artifact Archiving System**: Automatic archiving of all pipeline artifacts (models, data, metrics) to timestamped directories with manifest files.
- **Databricks Integration**: Optional data push to Databricks SQL with connector detection and error handling.
- **Production-Ready Resilience**: Pipeline completes successfully even with missing optional dependencies (MLflow, Databricks).
- **AI Playground Data**: Complete pipeline artifacts now available in `ai_playground_data/` for AI research and experimentation. Includes 20,000 records, 400+ features, comprehensive documentation, and analysis templates.
- **Artifact Organization**: All pipeline artifacts saved in timestamped subfolders (e.g., `demo_artifacts/20251016_142256/train/model_20251016_142256.pkl`) for better data lineage.
- **Summary Task**: Added final task to collect all files into `pipeline_summary_{ts}.json`.
- **MLflow Integration**: Fixed URI to `localhost:5000` with in-container server for reliable registration.
- **Docker Automation**: Data ingestion runs on container startup via `airflow-init`.
- **Successful Runs**: Pipeline completes end-to-end with model registration and artifact traceability.
- **Databricks Integration**: Connected to Databricks workspace with serverless compute. Catalog 'lifestyle_mlops_catalog' created with schemas for data storage. MLflow experiments logged to Databricks. Pipeline uploads preprocessed data to catalog tables.
- **Next Steps**: Set up Unity Catalog, deploy LLM endpoint, integrate pipeline with Databricks for scalable compute and AI explanations.

## 📁 Folder Structure
- `ai_playground_data/`: Complete pipeline artifacts for AI experimentation
  - `lifestyle_mlops_complete_run/`: Latest successful run data and documentation
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
- `docker-compose-airflow.yml`: Airflow setup## 📝 How to Run
1. Ensure `.env` has `KAGGLE_KEY`.
2. Start Airflow: `docker-compose -f docker-compose-airflow.yml up --build`
3. Trigger DAG in UI or via CLI.
4. Monitor logs; models and metrics saved to `demo_artifacts/`.
5. For MLflow UI, start server separately if needed.

## 📦 Requirements
Install with `pip install -r requirements.txt`. Key packages: airflow, pandas, scikit-learn, mlflow, pandera, requests.

## 📊 Diagram
<img width="7248" height="1212" alt="diagram" src="https://github.com/user-attachments/assets/c722ea16-b6f0-403e-aa88-91c88811547d" />

