import os
import yaml
import pandas as pd
import logging
import joblib
from sklearn.linear_model import LinearRegression

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("model_training")
logger.setLevel("DEBUG")

console_handeler = logging.StreamHandler()
console_handeler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "model_training.log")
file_handeler = logging.FileHandler(log_file_path)
file_handeler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
    
def load_data(X_train_path: str, y_train_path: str):
    try:
        logger.debug(f"Loading training data from {X_train_path} and {y_train_path}")
        X_train = pd.read_csv(X_train_path)
        y_train = pd.read_csv(y_train_path)
        logger.debug(f"Training data loaded successfully with shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
        return X_train, y_train
    except Exception as e:
        logger.debug(f"Error loading training data: {e}")
        raise e
    
def train_model(X_train:pd.DataFrame, y_train: pd.DataFrame):
    try:
        logger.debug("Starting model training")
        model = LinearRegression()
        model.fit(X_train, y_train)
        logger.debug("Model training completed successfully")
        return model
    except Exception as e:
        logger.debug(f"Error during model training: {e}")
        raise e
def save_model(model, model_output_path: str):
    try:
        logger.debug(f"Saving model to {model_output_path}")
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump(model, model_output_path)
        logger.debug("Model saved successfully")
    except Exception as e:
        logger.debug(f"Error saving model: {e}")
        raise e

def main():
    try:
        params = load_params("params.yaml")
        X_train_path = params["model_training"]["X_train"]
        y_train_path = params["model_training"]["y_train"]
        model_output_path = params["model_training"]["model_output"]
        X_train, y_train = load_data(X_train_path, y_train_path)
        model = train_model(X_train, y_train)
        save_model(model, model_output_path)
    except Exception as e:
        logger.debug(f"Error in main function: {e}")
        raise e


if __name__ == "__main__":
    main()