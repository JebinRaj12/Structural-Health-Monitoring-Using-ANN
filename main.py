import os
import sys
import logging
import tkinter as tk
from tkinter import messagebox

import config
from train_model import train

def main():
    # Reconfigure stdout to support UTF-8 on Windows console
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    # Setup basic logging to console for startup sequence
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("SHM_Startup")
    logger.info("Initializing Structural Health Monitoring Application...")

    # Check directories existence
    for folder in [config.MODELS_DIR, config.REPORTS_DIR, config.TRAINING_GRAPHS_DIR, config.LOGS_DIR]:
        os.makedirs(folder, exist_ok=True)

    # Check whether the model, scaler, and metrics exist
    model_exists = os.path.exists(config.MODEL_PATH)
    scaler_exists = os.path.exists(config.SCALER_PATH)
    metrics_exists = os.path.exists(config.METRICS_PATH)

    if not (model_exists and scaler_exists and metrics_exists):
        logger.info("Trained model, scaler, or metrics JSON not found. Triggering automated model training sequence...")
        
        # If the training dataset doesn't exist, we cannot train
        if not os.path.exists(config.DATASET_PATH):
            err_msg = (
                f"Critical Error: Training dataset building_health_monitoring_dataset.csv "
                f"could not be located at:\n{config.DATASET_PATH}\n\n"
                f"Please ensure the training dataset is in the folder: "
                f"C:\\Users\\Jebin Raj\\OneDrive\\文档\\professional\\research work\\SHM using ANN"
            )
            logger.critical(err_msg)
            
            # Show a graphical error box if Tkinter can initialize
            try:
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror("Dataset Missing", err_msg)
                root.destroy()
            except Exception:
                pass
            sys.exit(1)
            
        try:
            # Run model training
            success = train()
            if not success:
                raise RuntimeError("Model training returned failure status.")
            logger.info("Automated model training completed successfully.")
        except Exception as train_error:
            err_msg = f"Critical Error: Automated model training failed:\n{str(train_error)}\n\nCheck logs for details."
            logger.critical(err_msg)
            try:
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror("Training Failure", err_msg)
                root.destroy()
            except Exception:
                pass
            sys.exit(1)
    else:
        logger.info("Saved model, scaler, and metrics found. Bypassing training sequence.")

    # Launch GUI
    try:
        logger.info("Launching CustomTkinter Graphical User Interface...")
        from ui.main_window import SHMApp
        app = SHMApp()
        app.mainloop()
        logger.info("Application closed normally.")
    except Exception as gui_error:
        logger.critical(f"Critical error during GUI execution: {gui_error}", exc_info=True)
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Execution Error", f"A critical error occurred while running the application:\n{str(gui_error)}")
            root.destroy()
        except Exception:
            pass
        sys.exit(1)

if __name__ == "__main__":
    main()
