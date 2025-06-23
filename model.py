import pickle
import numpy as np
import os
import logging
from sklearn.datasets import load_iris # To get feature names dynamically
from config import config_by_name # Import config_by_name from config.py

env_name = os.getenv('FLASK_ENV', 'development')
current_config = config_by_name[env_name]

# Configure logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set default logging level

if logger.handlers:
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

log_file_path = current_config.LOG_FILE # Use the LOG_FILE path from the loaded config
file_handler = logging.FileHandler(log_file_path)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

_model = None
_feature_names = None

def load_iris_model(model_path):
    """Loads the pre-trained Iris classification model."""
    global _model, _feature_names
    if _model is None:
        logger.info(f"Attempting to load model from: {model_path}")
        try:
            with open(model_path, 'rb') as file:
                _model = pickle.load(file)
            logger.info("Iris model loaded successfully.")
            # Dynamically get feature names from the dataset
            _feature_names = load_iris().feature_names
        except FileNotFoundError:
            logger.error(f"Model file not found at {model_path}. Please run train_model.py first.")
            _model = None
            raise FileNotFoundError(f"Model file not found at {model_path}")
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            _model = None
            raise RuntimeError(f"Failed to load model: {e}")
    return _model

def get_feature_names():
    """Returns the feature names used by the model."""
    if _feature_names is None:
        # Try to load model to get feature names if not already loaded
        logger.warning("Feature names requested before model was loaded. Attempting to load model.")
        # This will raise an error if model path is invalid based on current_config.MODEL_PATH
        load_iris_model(current_config.MODEL_PATH)
    return _feature_names

def predict_iris(features):
    if _model is None:
        logger.warning("Model is not loaded. Attempting to load model before prediction.")
        try:
            # Attempt to load the model using the configured path
            load_iris_model(current_config.MODEL_PATH)
            if _model is None: # Check again if loading failed
                logger.error("Failed to load model after attempt. Cannot make prediction.")
                return -1, []
        except Exception as e:
            logger.error(f"Error during model loading attempt for prediction: {e}")
            return -1, []

    try:
        input_array = np.array(features).reshape(1, -1).astype(float)
        prediction_proba = _model.predict_proba(input_array)[0]
        predicted_class = np.argmax(prediction_proba)

        logger.info(f"Prediction made for features {features}: Class {predicted_class}, Probabilities {prediction_proba.tolist()}")
        return int(predicted_class), prediction_proba.tolist()
    except ValueError as ve:
        logger.error(f"ValueError during prediction: {ve}. Ensure input features are numeric and correctly shaped.")
        return -1, []
    except Exception as e:
        logger.error(f"An unexpected error occurred during prediction: {e}")
        return -1, []
