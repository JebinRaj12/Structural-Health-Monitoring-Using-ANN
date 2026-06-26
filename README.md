# Structural Health Monitoring (SHM) using Artificial Neural Network (ANN)

A professional desktop application developed in Python to evaluate the structural health condition of buildings using sensor telemetry data (accelerometer axes, strain, and temperature) and a Feedforward Artificial Neural Network (MLP) built with TensorFlow/Keras.

---

## Project Structure

The project directory is structured as follows:

```text
ML model/
│
├── config.py                 # Application configurations, paths, hyper-parameters, and GUI themes
├── main.py                   # Application entry point with automated model check and GUI bootstrapper
├── train_model.py            # Neural network architecture definition, training loop, and graphing utility
├── predict.py                # Wrapper module to run inference on files and calculate health classification stats
├── requirements.txt          # Python dependency specifications
├── README.md                 # Complete documentation and user instruction manual
│
├── dataset/                  # Folder for auxiliary dataset resources
├── models/                   # Saved artifacts (ann_model.keras, scaler.pkl, medians.pkl, model_metrics.json)
├── ui/                       # CustomTkinter GUI panel layouts
│   └── main_window.py        # Principal GUI dashboard frame and handlers
│
├── utils/                    # Preprocessing pipelines and report utilities
│   ├── data_preprocessing.py # Duplicates removal, column validation, median imputation, and timestamp scaling
│   └── pdf_generator.py      # Automates reportlab PDF compilation and matplotlib distribution embedding
│
├── reports/                  # Generated structural analysis PDF reports
│   └── training_graphs/      # Neural Network curves (accuracy, loss, confusion matrix)
│
├── assets/                   # Images and static layout components
└── logs/                     # Application logging logs (shm_app.log)
```

---

## Data Features

The models ingest 6 input features to determine the structural condition:
1. **Timestamp**: Represented as numeric Unix seconds epoch (standardized with `StandardScaler`).
2. **Accel_X**: Acceleration along the X-axis (m/s²).
3. **Accel_Y**: Acceleration along the Y-axis (m/s²).
4. **Accel_Z**: Acceleration along the Z-axis (m/s²).
5. **Strain**: Strain telemetry readings (με).
6. **Temperature**: Environmental temperature (°C).

### Class Labels
- `0` = **Healthy**
- `1` = **Minor Damage**
- `2` = **Severe Damage**

---

## Neural Network (MLP) Specifications

- **Framework**: TensorFlow 2.x + Keras
- **Architecture**:
  - **Input Layer**: 6 features
  - **Hidden Layer 1**: 64 Neurons, ReLU activation, Dropout 0.20
  - **Hidden Layer 2**: 32 Neurons, ReLU activation, Dropout 0.20
  - **Output Layer**: 3 Neurons, Softmax activation
- **Compiler**:
  - **Optimizer**: Adam (Learning Rate = 0.001)
  - **Loss**: Sparse Categorical Crossentropy
- **Parameters**:
  - Epochs: 100
  - Batch Size: 32
  - Validation Split: 20%
  - EarlyStopping: Monitor `val_loss`, patience = 10, restore best weights

---

## Preprocessing Pipeline

- **Column Alignment**: Automatically matches column headers using fuzzy matching rules (e.g. mapping `Temp (°C)` to standard `Temperature` feature).
- **Duplicate & NaN Handling**: Removes identical rows. Imputes missing numeric parameters using training medians (stored in `medians.pkl`).
- **Feature Scaling**: Computes Z-score normalization utilizing `StandardScaler`. Scaler parameters are saved to `models/scaler.pkl` and loaded on prediction. No new scaling configuration is generated during prediction.

---

## Installation & Setup

1. Open your terminal in VS Code and verify you are running Python 3.8 to 3.11.
2. Navigate to the project root directory:
   ```powershell
   cd "C:\Users\Jebin Raj\OneDrive\文档\professional\research work\SHM using ANN\ML model"
   ```
3. Install required libraries:
   ```powershell
   pip install -r requirements.txt
   ```

---

## How to Run

Launch the application using:
```powershell
python main.py
```

### Dynamic Prediction Workflow
1. If launched for the first time, the application detects that the model is missing and automatically initiates the ANN training sequence using the training dataset located at the parent directory (`C:\Users\Jebin Raj\OneDrive\文档\professional\research work\SHM using ANN\building_health_monitoring_dataset.csv`).
2. Performance curves and the confusion matrix are saved to `reports/training_graphs/`.
3. Validation metrics (Accuracy, Precision, Recall, F1) are generated and stored in `models/model_metrics.json`.
4. Once training concludes, the CustomTkinter GUI dashboard launches, displaying validation performance.
5. In subsequent executions, the application loads the trained files instantly without repeating the training sequence.
6. Click **Browse CSV / Excel File** to load a test file.
7. Click **Analyze Structure** to run classification.
8. The overall structural health is calculated using a **majority-voting logic** of all prediction rows. The dashboard visualizes counts, percentage distributions, and automatically exports a detailed PDF report containing a pie chart of the distribution to the `reports/` folder.
