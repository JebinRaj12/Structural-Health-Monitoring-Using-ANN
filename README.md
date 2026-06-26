# Structural Health Monitoring Using Artificial Neural Network (ANN)

## Overview

Structural Health Monitoring (SHM) is an intelligent system used to assess the condition of civil structures using sensor measurements. This project presents a desktop application that utilizes an Artificial Neural Network (ANN) to classify the structural health condition based on sensor data.

The application accepts Excel or CSV sensor files, preprocesses the data automatically, predicts the structural condition using a trained ANN model, visualizes the prediction results, and generates a professional PDF report.

---

## Dashboard Preview

> Save the screenshot below inside the **assets** folder as:

```text
assets/dashboard.png
```

Then the image will automatically appear here.

![Structural Health Monitoring Dashboard](assets/dashboard.png)

---

## Features

* Upload CSV and Excel (.csv, .xls, .xlsx) files
* Automatic data preprocessing
* Missing value handling
* Artificial Neural Network (ANN) prediction
* Structural health classification
* Prediction confidence score
* Damage distribution analysis
* Engineering recommendation generation
* Pie chart visualization
* Automatic PDF report generation
* User-friendly desktop interface using CustomTkinter

---

## Technologies Used

* Python 3.12
* TensorFlow / Keras
* Artificial Neural Network (ANN)
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* CustomTkinter
* ReportLab
* OpenPyXL

---

## Project Structure

```text
Structural-Health-Monitoring-Using-ANN
│
├── assets/
├── dataset/
├── models/
│   ├── ann_model.keras
│   ├── scaler.pkl
│   ├── medians.pkl
│   └── model_metrics.json
│
├── ui/
│   └── main_window.py
│
├── utils/
│   ├── data_preprocessing.py
│   └── pdf_generator.py
│
├── config.py
├── main.py
├── predict.py
├── train_model.py
├── requirements.txt
└── README.md
```

---

## Artificial Neural Network Architecture

* Input Layer

  * Timestamp
  * Acceleration X
  * Acceleration Y
  * Acceleration Z
  * Strain
  * Temperature

↓

* Hidden Layer 1

  * 64 Neurons
  * ReLU Activation
  * Dropout (20%)

↓

* Hidden Layer 2

  * 32 Neurons
  * ReLU Activation
  * Dropout (20%)

↓

* Output Layer

  * Healthy
  * Minor Damage
  * Severe Damage

---

## Workflow

1. Upload a CSV or Excel file.
2. Validate the input data.
3. Preprocess the dataset.
4. Scale the input features.
5. Load the trained ANN model.
6. Predict the structural condition.
7. Display prediction confidence.
8. Generate engineering recommendations.
9. Visualize the damage distribution.
10. Export the analysis as a PDF report.

---

## Model Performance

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 79.00% |
| Precision | 0.7565 |
| Recall    | 0.7900 |
| F1-Score  | 0.7667 |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/JebinRaj12/Structural-Health-Monitoring-Using-ANN.git
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python main.py
```

---

## Input Dataset

Supported formats:

* CSV
* XLS
* XLSX

Required input features:

* Timestamp
* Accel_X
* Accel_Y
* Accel_Z
* Strain
* Temperature

---

## Output

The application provides:

* Overall Structural Health Status
* Prediction Confidence
* Healthy Sample Count
* Minor Damage Count
* Severe Damage Count
* Engineering Recommendation
* Damage Distribution Chart
* PDF Report

---

## Future Improvements

* CNN and LSTM model comparison
* Real-time IoT sensor integration
* Cloud database support
* Live dashboard
* Mobile application
* Structural anomaly detection
* Multi-model ensemble prediction

---

## Author

**Jebin Raj J**

Bachelor of Engineering (Civil Engineering)

Machine Learning | Artificial Intelligence | Structural Health Monitoring | Civil Engineering

GitHub:
https://github.com/JebinRaj12

---

## License

This project is developed for academic and educational purposes.
