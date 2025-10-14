import os
import logging
import requests
import base64
import zipfile
import io
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define dataset and target directory
OWNER_SLUG = 'jockeroika'
DATASET_SLUG = 'life-style-data'
TARGET_DIR = 'data/raw'

def download_kaggle_dataset():
    """
    Downloads and extracts a dataset from Kaggle using the REST API.
    """
    try:
        os.makedirs(TARGET_DIR, exist_ok=True)

        # Check if the dataset is already downloaded
        # Note: This assumes a file from the dataset; you might need to adjust the filename.
        expected_file = os.path.join(TARGET_DIR, 'Life Style Data.csv')
        if os.path.exists(expected_file):
            logging.info(f"Dataset already exists at {expected_file}. Skipping download.")
            return

        # 1: Preparing the URL.
        base_url = "https://www.kaggle.com/api/v1"
        url = f"{base_url}/datasets/download/{OWNER_SLUG}/{DATASET_SLUG}"

        # 2: Encoding the credentials and preparing the request header.
        username = os.getenv("KAGGLE_USERNAME")
        key = os.getenv("KAGGLE_KEY")

        if not username or not key:
            logging.error("Kaggle username or key not found in environment variables.")
            return

        creds = base64.b64encode(bytes(f"{username}:{key}", "ISO-8859-1")).decode("ascii")
        headers = {
          "Authorization": f"Basic {creds}"
        }

        # 3: Sending a GET request to the URL with the encoded credentials.
        logging.info(f"Downloading dataset '{OWNER_SLUG}/{DATASET_SLUG}' to {TARGET_DIR}...")
        response = requests.get(url, headers=headers, verify=False) # Setting verify=False to bypass SSL issues
        response.raise_for_status()  # Raise an exception for bad status codes

        # 4: Loading the response as a file via io and opening it via zipfile.
        zf = zipfile.ZipFile(io.BytesIO(response.content))
        
        # Log the list of files in the zip archive
        logging.info(f"Files in zip archive: {zf.namelist()}")

        # 5: Extracting the files to the target directory.
        zf.extractall(TARGET_DIR)
        
        logging.info("Download and extraction complete.")

        # 6: Reading the CSV from the target directory and converting it to a dataframe.
        file_name = "Final_data.csv"
        df = pd.read_csv(os.path.join(TARGET_DIR, file_name))

        # 7: Printing the dataframe head.
        print("Dataset head:")
        print(df.head())

    except requests.exceptions.RequestException as e:
        logging.error(f"Error during dataset download: {e}")
    except zipfile.BadZipFile:
        logging.error("Failed to unzip the file. The downloaded file may be corrupt or not a zip file.")
        logging.error(f"Response content: {response.text[:500]}") # Log first 500 chars of response
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    download_kaggle_dataset()