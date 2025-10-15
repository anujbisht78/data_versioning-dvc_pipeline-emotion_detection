import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import json
import logging
import os
import sys

# ==================== LOGGER SETUP ====================
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# File handler (for errors)
log_path = os.path.join('src', 'model')
os.makedirs(log_path, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_path, 'model_evaluation_error.log'))
file_handler.setLevel(logging.ERROR)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ==================== MODEL EVALUATION ====================
try:
    logger.info("Starting model evaluation...")

    # Load test data
    test_data_path = 'data/processed/test_tfidf.csv'
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found at {test_data_path}")

    test_data = pd.read_csv(test_data_path)
    logger.info(f"Test data loaded successfully with shape {test_data.shape}")

    x_test = test_data.iloc[:, 0:-1].values
    y_test = test_data.iloc[:, -1].values

    # Load the trained model
    model_path = 'models/model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully.")

    # Predictions
    logger.info("Generating predictions...")
    y_pred = model.predict(x_test)

    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(x_test)
    else:
        raise AttributeError("Model does not support probability prediction (predict_proba).")

    # Evaluate metrics
    logger.info("Calculating evaluation metrics...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    # Fix for ROC-AUC
    if y_pred_proba.shape[1] == 2:
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')

    metrics_dict = {
        'Accuracy': round(accuracy, 4),
        'Precision Score': round(precision, 4),
        'Recall Score': round(recall, 4),
        'AUC-ROC': round(auc, 4)
    }

    # Save metrics
    with open('reports/metrics.json', 'w') as file:
        json.dump(metrics_dict, file, indent=4)

    logger.info(f"Model evaluation completed successfully. Metrics saved to 'metrics.json'.")
    logger.info(metrics_dict)

except FileNotFoundError as fnf:
    logger.error(f"File not found: {fnf}")
    sys.exit(1)

except AttributeError as ae:
    logger.error(f"Attribute Error: {ae}")
    sys.exit(1)

except Exception as e:
    logger.error(f"Unexpected error during model evaluation: {str(e)}", exc_info=True)
    sys.exit(1)
