import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
import config

# Reconfigure stdout to support UTF-8 on Windows console
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Setup directories
for directory in [config.MODELS_DIR, config.REPORTS_DIR, config.TRAINING_GRAPHS_DIR, config.LOGS_DIR, config.ASSETS_DIR, config.DATASET_DIR]:
    os.makedirs(directory, exist_ok=True)

# Setup Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("SHM_Preprocessing")

MEDIANS_PATH = os.path.join(config.MODELS_DIR, "medians.pkl")

def load_file(file_path):
    """Loads CSV, XLS, or XLSX files into a pandas DataFrame."""
    logger.info(f"Attempting to load file: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")
        
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.csv':
            df = pd.read_csv(file_path)
        elif ext in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path, engine='openpyxl')
        else:
            raise ValueError(f"Unsupported file extension: {ext}. Only CSV, XLS, and XLSX are supported.")
        logger.info(f"Successfully loaded file. Rows: {len(df)}, Columns: {len(df.columns)}")
        return df
    except Exception as e:
        logger.exception("Prediction failed")
    error = e
    self.after(0, lambda err=error: self._handle_prediction_error(err))

def align_columns(df, is_training=False):
    """Maps columns using keyword matching and validates required columns."""
    df_aligned = df.copy()

    clean_cols = {col: str(col).strip().lower() for col in df_aligned.columns}

    mapped_cols = {}

    for original, clean in clean_cols.items():

        if "timestamp" in clean or clean == "time":
            mapped_cols[original] = "Timestamp"

        elif "accel_x" in clean or "accel x" in clean:
            mapped_cols[original] = "Accel_X"

        elif "accel_y" in clean or "accel y" in clean:
            mapped_cols[original] = "Accel_Y"

        elif "accel_z" in clean or "accel z" in clean:
            mapped_cols[original] = "Accel_Z"

        elif "strain" in clean:
            mapped_cols[original] = "Strain"

        elif "temp" in clean or "temperature" in clean:
            mapped_cols[original] = "Temperature"

        elif "condition" in clean:
            mapped_cols[original] = config.TARGET_COL

    df_aligned.rename(columns=mapped_cols, inplace=True)

    required_features = [
        "Timestamp",
        "Accel_X",
        "Accel_Y",
        "Accel_Z",
        "Strain",
        "Temperature",
    ]

    if is_training:
        required_features.append(config.TARGET_COL)

    missing_cols = [col for col in required_features if col not in df_aligned.columns]

    if missing_cols:
        logger.error(f"Missing columns: {missing_cols}. Available columns: {list(df_aligned.columns)}")
        raise KeyError(f"Missing required columns in dataset: {', '.join(missing_cols)}")

    keep_cols = [col for col in required_features if col in df_aligned.columns]

    return df_aligned[keep_cols]

def clean_missing_and_duplicates(df, is_training=False):
    """Handles duplicates and cleans missing values by imputing them with medians."""
    df_cleaned = df.copy()
    
    # Remove duplicates
    initial_rows = len(df_cleaned)
    df_cleaned.drop_duplicates(inplace=True)
    duplicate_count = initial_rows - len(df_cleaned)
    if duplicate_count > 0:
        logger.info(f"Removed {duplicate_count} duplicate rows.")
        
    # Convert Timestamp to datetime and fill NaNs in Timestamp first (using forward/backward fill)
    df_cleaned['Timestamp'] = pd.to_datetime(df_cleaned['Timestamp'], errors='coerce')
    if df_cleaned['Timestamp'].isna().any():
        logger.warning("NaNs found in Timestamp column. Applying forward/backward fill.")
        df_cleaned['Timestamp'] = df_cleaned['Timestamp'].ffill().bfill()
        # If still NaN (e.g. all empty), assign a dummy date range
        if df_cleaned['Timestamp'].isna().all():
            logger.error("All timestamps are invalid. Filling with dummy date range.")
            df_cleaned['Timestamp'] = pd.date_range(start='2025-01-01', periods=len(df_cleaned), freq='S')
            
    # Convert Timestamp to Unix epoch numeric float
    df_cleaned['Timestamp_numeric'] = df_cleaned['Timestamp'].astype('int64') // 10**9
    df_cleaned.drop(columns=['Timestamp'], inplace=True)
    
    # Identify numeric columns for imputation
    numeric_cols = ["Accel_X", "Accel_Y", "Accel_Z", "Strain", "Temperature"]
    
    # Impute missing values
    if is_training:
        # Calculate medians of numeric features and save them
        medians = {}
        for col in numeric_cols:
            median_val = df_cleaned[col].median()
            # If median_val is NaN (e.g. column is completely empty), default to 0.0
            if pd.isna(median_val):
                median_val = 0.0
            medians[col] = median_val
            df_cleaned[col] = df_cleaned[col].fillna(median_val)
            
        # Median for Timestamp_numeric
        t_median = df_cleaned['Timestamp_numeric'].median()
        medians['Timestamp_numeric'] = t_median if not pd.isna(t_median) else 0.0
        
        # Save medians for prediction use
        joblib.dump(medians, MEDIANS_PATH)
        logger.info(f"Calculated and saved feature medians for training: {medians}")
        
        # Drop rows where target label is missing
        if config.TARGET_COL in df_cleaned.columns:
            df_cleaned.dropna(subset=[config.TARGET_COL], inplace=True)
            df_cleaned[config.TARGET_COL] = df_cleaned[config.TARGET_COL].astype(int)
    else:
        # Prediction mode: load medians and impute
        if not os.path.exists(MEDIANS_PATH):
            raise FileNotFoundError("Medians statistics file not found. Train the model first to generate training medians.")
        medians = joblib.load(MEDIANS_PATH)
        
        for col in ["Timestamp_numeric"] + numeric_cols:
            fill_val = medians.get(col, 0.0)
            nan_count = df_cleaned[col].isna().sum()
            if nan_count > 0:
                logger.warning(f"Imputing {nan_count} missing values in '{col}' using training median: {fill_val}")
                df_cleaned[col] = df_cleaned[col].fillna(fill_val)
                
    # Return structured features in exact order
    ordered_cols = config.FEATURES.copy()
    if is_training and config.TARGET_COL in df_cleaned.columns:
        ordered_cols.append(config.TARGET_COL)
        
    return df_cleaned[ordered_cols]

def preprocess_training_data():
    """Entire pipeline to load and preprocess training dataset."""
    logger.info("Starting training data preprocessing pipeline...")
    # Load training dataset
    df = load_file(config.DATASET_PATH)
    # Check dataset validation
    if len(df) < 10:
        raise ValueError(f"Training dataset is too small. Found only {len(df)} records.")
        
    # Align headers
    df_aligned = align_columns(df, is_training=True)
    # Clean duplicates and missing values
    df_cleaned = clean_missing_and_duplicates(df_aligned, is_training=True)
    
    # Split features and target
    X = df_cleaned[config.FEATURES]
    y = df_cleaned[config.TARGET_COL]
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler
    joblib.dump(scaler, config.SCALER_PATH)
    logger.info(f"Scaler successfully fit and saved to: {config.SCALER_PATH}")
    
    return X_scaled, y.values

def preprocess_prediction_data(file_path):
    """Entire pipeline to load, validate, clean and scale new prediction data."""
    logger.info(f"Starting prediction data preprocessing pipeline for file: {file_path}")
    # Load data
    df = load_file(file_path)
    if len(df) == 0:
        raise ValueError("The uploaded file is empty.")
        
    # Align headers
    df_aligned = align_columns(df, is_training=False)
    # Clean duplicates and missing values
    df_cleaned = clean_missing_and_duplicates(df_aligned, is_training=False)
    
    # Load scaler
    if not os.path.exists(config.SCALER_PATH):
        raise FileNotFoundError("Fitted Scaler not found at models/scaler.pkl. Train the model first.")
        
    scaler = joblib.load(config.SCALER_PATH)
    
    # Scale features using training scaler
    X_scaled = scaler.transform(df_cleaned[config.FEATURES])
    logger.info("Successfully preprocessed and scaled prediction data.")
    
    return X_scaled, df_cleaned
