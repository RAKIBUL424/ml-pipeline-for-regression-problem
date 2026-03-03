import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
import yaml


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")


console_handeler = logging.StreamHandler()
console_handeler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "data_preprocessing.log")
file_handeler = logging.FileHandler(log_file_path)
file_handeler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handeler.setFormatter(formatter)
file_handeler.setFormatter(formatter)

logger.addHandler(console_handeler)
logger.addHandler(file_handeler)

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logger.debug(f"Parameters loaded successfully from {params_path}")
        return params
    except Exception as e:
        logger.debug(f"Error loading parameters: {e}")
        raise e

def load_data(data_path: str) -> pd.DataFrame:
    try:
        logger.debug(f"Loading data from path: {data_path}")
        data = pd.read_csv(data_path)
        logger.debug(f"Data loaded successfully with shape: {data.shape}")
        return data
    except Exception as e:
        logger.debug(f"Error loading data: {e}")
        raise e

def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug("Starting data preprocessing")
        df = pd.get_dummies(df,columns=["Fuel_Type", "Seller_Type", "Transmission"], drop_first=True)
        logger.debug("Data preprocessing completed successfully")
        return df
    except Exception as e:
        logger.debug(f"Error in data preprocessing: {e}")
        raise e

def save_data(df: pd.DataFrame, target_column: str, output_path: str) -> None:
    try:
        test_size = load_params("params.yaml")["data_preprocessing"]["test_size"]
        os.makedirs(output_path, exist_ok=True)
        X = df.drop(columns=[target_column], axis=1)
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        X_train.to_csv(os.path.join(output_path, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(output_path, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(output_path, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(output_path, "y_test.csv"), index=False)
        logger.debug(f"Data saved successfully at {output_path}")
    except Exception as e:
        logger.debug(f"Error saving data: {e}")
        raise e
    
def main():
    try:
        df = load_data('data/raw/saved_data.csv')
        preprocessed_df = data_preprocessing(df)
        save_data(preprocessed_df, "Selling_Price", "./data/processed")
        logger.debug("Data preprocessing pipeline completed successfully")
    except Exception as e:
        logger.debug(f"Error in data preprocessing pipeline: {e}")
        raise e

if __name__ == "__main__":
    main()