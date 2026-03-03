import os
import joblib
import joblib
import pandas as pd
import logging
import yaml
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("model_evaluation")
logger.setLevel("DEBUG")

console_handeler = logging.StreamHandler()
console_handeler.setLevel("DEBUG")  
log_file_path = os.path.join(log_dir, "model_evaluation.log")
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
    
def load_data(X_test_path: str, y_test_path: str):
    try:
        logger.debug(f"Loading test data from {X_test_path} and {y_test_path}")
        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path)
        logger.debug(f"Test data loaded successfully with shapes: X_test: {X_test.shape}, y_test: {y_test.shape}")
        return X_test, y_test
    except Exception as e:
        logger.debug(f"Error loading test data: {e}")
        raise e

def load_model(model_path: str):
    try:
        logger.debug(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        logger.debug("Model loaded successfully")
        return model
    except Exception as e:
        logger.debug(f"Error loading model: {e}")
        raise e

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    try:
        logger.debug("Starting model evaluation")
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.debug(f"Model evaluation completed successfully with MSE: {mse}, R2 Score: {r2}")
        return {"mean_squared_error": mse, "r2_score": r2}
    except Exception as e:
        logger.debug(f"Error during model evaluation: {e}")
        raise e
def save_evaluation_results(results: dict, output_path: str) -> dict:
    try:
        logger.debug(f"Saving evaluation results to {output_path}")
        os.makedirs(output_path, exist_ok=True)
        results_path = os.path.join(output_path, "evaluation_results.yaml")
        with open(results_path, "w") as file:
            yaml.dump(results, file)
        logger.debug("Evaluation results saved successfully")
    except Exception as e:
        logger.debug(f"Error saving evaluation results: {e}")
        raise e
def main():
    try:
        params = load_params("params.yaml")
        X_test_path = params["model_evaluation"]["X_test"]
        y_test_path = params["model_evaluation"]["y_test"]
        model_path = params["model_training"]["model_output"]
        evaluation_output_path = params["model_evaluation"]["evaluation_output"]

        X_test, y_test = load_data(X_test_path, y_test_path)
        model = load_model(model_path)
        evaluation_results = evaluate_model(model, X_test, y_test)
        save_evaluation_results(evaluation_results, evaluation_output_path)
    except Exception as e:
        logger.debug(f"Error in main function: {e}")
        raise e

if __name__ == "__main__":
    main()
