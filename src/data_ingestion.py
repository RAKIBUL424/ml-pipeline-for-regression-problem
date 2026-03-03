import os
import pandas as pd
import logging
import yaml
from sklearn.model_selection import train_test_split


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

console_handeler = logging.StreamHandler()
console_handeler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "data_ingestion.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handeler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handeler)
logger.addHandler(file_handler)

def load_params(params_path: str) ->dict:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logger.info(f"Parameters loaded successfully from {params_path}")
        return params
    except Exception as e:
        logger.debug(f"Error loading parameters: {e}")
        raise e


def load_data(data_url: str) -> pd.DataFrame:
    try:
        logger.info(f"Loading data from url: {data_url}")
        data = pd.read_csv(data_url)
        logger.info(f"Data loaded successdully with shape: {data.shape}")
        return data
    except Exception as e:
        logger.debug(f"Error loading data: {e}")
        raise e

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Preprocessing data")
        df = df.drop(columns=["Car_Name"], axis=1)
        df["Age"] = 2020 - df["Year"]
        df.drop(columns=["Year"], axis=1, inplace=True)
        logger.info("Data preprocessing completed successfully")
        return df
    except Exception as e:
        logger.debug(f"Error preprocessing data: {e}")
        raise e

def save_data(saved_data: pd.DataFrame, data_path: str) -> None:
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        saved_data.to_csv(os.path.join(raw_data_path, 'saved_data.csv'), index=False)
        logger.debug(f"Data saved successfully to {raw_data_path}")
    except Exception as e:
        logger.debug(f"Error saving data: {e}")
        raise e



def main():
    try:
        params = load_params("params.yaml")
        data_url = params["data_ingestion"]["data_url"]
        data = load_data(data_url)
        preprocessed_data = preprocess_data(data)
        save_data(preprocessed_data, data_path="./data")
    except Exception as e:
        logger.debug(f"Error in data ingestion process: {e}")
        raise e



if __name__ == "__main__":
    main()
