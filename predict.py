import os
import logging
import numpy as np
import pandas as pd
from tensorflow import keras
import config
from utils.data_preprocessing import preprocess_prediction_data

logger = logging.getLogger("SHM_Prediction")

class SHMPredictor:
    def __init__(self):
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Loads the compiled Keras model from disk."""
        if not os.path.exists(config.MODEL_PATH):
            logger.warning(f"Model file not found at: {config.MODEL_PATH}")
            self.model = None
            return
            
        try:
            self.model = keras.models.load_model(config.MODEL_PATH)
            logger.info("Successfully loaded Keras model for prediction.")
        except Exception as e:
            logger.error(f"Error loading model from {config.MODEL_PATH}: {e}")
            self.model = None
            
    def is_model_available(self):
        """Checks if model is loaded and ready."""
        # Refresh reference if it was not loaded originally
        if self.model is None:
            self._load_model()
        return self.model is not None
        
    def predict(self, file_path):
        """Performs structural health analysis on a CSV or Excel file."""
        if not self.is_model_available():
            raise FileNotFoundError("Trained ANN model is not available. Please train the model first.")
            
        # Preprocess features
        X_scaled, df_cleaned = preprocess_prediction_data(file_path)
        
        # Run model inference
        logger.info(f"Running inference on {len(X_scaled)} records...")
        pred_probs = self.model.predict(X_scaled)
        
        # Extract predicted class labels and confidence percentages
        pred_classes = np.argmax(pred_probs, axis=1)
        pred_confidences = np.max(pred_probs, axis=1) * 100
        
        # Create output DataFrame columns
        df_results = df_cleaned.copy()
        
        # Add labels and confidence values
        df_results['Predicted_Class'] = pred_classes
        df_results['Predicted_Label'] = [config.CLASS_LABELS[c] for c in pred_classes]
        df_results['Confidence'] = pred_confidences
        
        # Compute counts
        total_samples = len(pred_classes)
        healthy_count = int(np.sum(pred_classes == 0))
        minor_count = int(np.sum(pred_classes == 1))
        severe_count = int(np.sum(pred_classes == 2))
        
        # Compute percentages
        healthy_pct = (healthy_count / total_samples) * 100
        minor_pct = (minor_count / total_samples) * 100
        severe_pct = (severe_count / total_samples) * 100
        
        # Determine overall structural health based on majority class
        # In case of a tie, prioritize the more severe condition for safety (2 > 1 > 0)
        counts = {0: healthy_count, 1: minor_count, 2: severe_count}
        majority_class = max(counts, key=lambda k: (counts[k], k))
        overall_health = config.CLASS_LABELS[majority_class]
        
        # Overall prediction confidence is the average confidence of the predictions
        overall_confidence = float(np.mean(pred_confidences))
        
        logger.info(f"Prediction Summary - Total: {total_samples}, H: {healthy_count} ({healthy_pct:.1f}%), "
                    f"M: {minor_count} ({minor_pct:.1f}%), S: {severe_count} ({severe_pct:.1f}%)")
        logger.info(f"Overall Structural Health: {overall_health} (Avg Conf: {overall_confidence:.2f}%)")
        
        return {
            "overall_health": overall_health,
            "overall_confidence": overall_confidence,
            "total_samples": total_samples,
            "healthy_count": healthy_count,
            "minor_count": minor_count,
            "severe_count": severe_count,
            "healthy_pct": healthy_pct,
            "minor_pct": minor_pct,
            "severe_pct": severe_pct,
            "detailed_results": df_results,
            "pred_classes": pred_classes,
            "pred_confidences": pred_confidences
        }
