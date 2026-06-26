import os

# Base paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = r"C:\Users\Jebin Raj\OneDrive\文档\professional\research work\SHM using ANN\building_health_monitoring_dataset.csv"

# Output directories
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
REPORTS_DIR = os.path.join(PROJECT_DIR, "reports")
TRAINING_GRAPHS_DIR = os.path.join(REPORTS_DIR, "training_graphs")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")
ASSETS_DIR = os.path.join(PROJECT_DIR, "assets")
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")

# File paths
MODEL_PATH = os.path.join(MODELS_DIR, "ann_model.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")
LOG_FILE = os.path.join(LOGS_DIR, "shm_app.log")

# Column specifications
FEATURES = ["Timestamp_numeric", "Accel_X", "Accel_Y", "Accel_Z", "Strain", "Temperature"]
TARGET_COL = "Condition Label"

# Dataset Header column mapping for user upload files (fuzzy matching)
HEADER_MAPPINGS = {

    "timestamp": "Timestamp",
    "time": "Timestamp",

    "accel_x (m/s^2)": "Accel_X",
    "accel_x": "Accel_X",
    "accel x": "Accel_X",

    "accel_y (m/s^2)": "Accel_Y",
    "accel_y": "Accel_Y",
    "accel y": "Accel_Y",

    "accel_z (m/s^2)": "Accel_Z",
    "accel_z": "Accel_Z",
    "accel z": "Accel_Z",

    "strain (µε)": "Strain",
    "strain (με)": "Strain",      # ADD THIS
    "strain": "Strain",

    "temp (°c)": "Temperature",
    "temp (°C)": "Temperature",   # ADD THIS
    "temp": "Temperature",
    "temperature": "Temperature",
    "temperature (°c)": "Temperature"
}

# ANN Model Architecture & Training settings
INPUT_DIM = 6
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.20
VAL_SPLIT = 0.20
RANDOM_STATE = 42

# Early stopping configuration
PATIENCE = 10

# Structural Health Classes
CLASS_LABELS = {
    0: "Healthy",
    1: "Minor Damage",
    2: "Severe Damage"
}

# Engineering Recommendations
RECOMMENDATIONS = {
    "Healthy": [
        "No significant structural damage detected.",
        "Continue routine inspection."
    ],
    "Minor Damage": [
        "Minor abnormalities detected.",
        "Schedule detailed inspection."
    ],
    "Severe Damage": [
        "Immediate structural inspection required.",
        "Restrict access if necessary."
    ]
}

# GUI configuration constants
WINDOW_TITLE = "Structural Health Monitoring Dashboard"
WINDOW_SIZE = "1400x800"
THEME_MODE = "light"
COLOR_THEME = "blue"

# Visual layout styling
COLOR_HEALTHY = "#2ECC71"  # Emerald Green
COLOR_MINOR = "#F39C12"    # Orange / Amber
COLOR_SEVERE = "#E74C3C"   # Alizarin Red
COLOR_CARD_BG = "#FFFFFF"
COLOR_PRIMARY = "#3498DB"  # Peter River Blue
COLOR_BG = "#F8F9FA"       # Off-white background
COLOR_TEXT = "#2C3E50"
COLOR_MUTED = "#7F8C8D"
