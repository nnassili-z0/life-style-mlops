FROM apache/airflow:2.8.1

# Switch to root to install dependencies
USER root

# Copy requirements.txt into the image
COPY requirements.txt /requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /requirements.txt

# Switch back to airflow user
USER airflow
