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