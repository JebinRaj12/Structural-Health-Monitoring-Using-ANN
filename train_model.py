import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
# Use non-interactive backend for Matplotlib to avoid Tkinter issues during training
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import config
from utils.data_preprocessing import preprocess_training_data

# Reconfigure stdout to support UTF-8 on Windows console
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Set random seeds for reproducibility
np.random.seed(config.RANDOM_STATE)
tf.random.set_seed(config.RANDOM_STATE)

logger = logging.getLogger("SHM_Training")

def build_model():
    """Builds the Feedforward MLP Neural Network based on user specifications."""
    model = keras.Sequential([
        # Input Layer & Hidden Layer 1
        layers.Dense(64, activation='relu', input_shape=(config.INPUT_DIM,), name="hidden_1"),
        layers.Dropout(config.DROPOUT_RATE, name="dropout_1"),
        
        # Hidden Layer 2
        layers.Dense(32, activation='relu', name="hidden_2"),
        layers.Dropout(config.DROPOUT_RATE, name="dropout_2"),
        
        # Output Layer (3 classes: Healthy, Minor Damage, Severe Damage)
        layers.Dense(3, activation='softmax', name="output")
    ])
    
    # Configure optimizer with learning rate
    opt = optimizers.Adam(learning_rate=config.LEARNING_RATE)
    
    # Compile model
    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_and_save_graphs(history, y_true, y_pred_classes):
    """Generates and saves performance graphs in reports/training_graphs/."""
    os.makedirs(config.TRAINING_GRAPHS_DIR, exist_ok=True)
    
    # 1. Accuracy Curve
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='#3498DB', linewidth=2)
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#2ECC71', linewidth=2)
    plt.title('ANN Training & Validation Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    acc_path = os.path.join(config.TRAINING_GRAPHS_DIR, "accuracy_curve.png")
    plt.savefig(acc_path, dpi=150)
    plt.close()
    logger.info(f"Accuracy curve saved to: {acc_path}")
    
    # 2. Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss', color='#E74C3C', linewidth=2)
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss', color='#F39C12', linewidth=2)
    plt.title('ANN Training & Validation Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    loss_path = os.path.join(config.TRAINING_GRAPHS_DIR, "loss_curve.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()
    logger.info(f"Loss curve saved to: {loss_path}")
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    classes = [config.CLASS_LABELS[i] for i in range(3)]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)
    
    # Add labels to cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=12, fontweight='bold')
                     
    plt.ylabel('True Class', fontsize=11, fontweight='semibold')
    plt.xlabel('Predicted Class', fontsize=11, fontweight='semibold')
    plt.tight_layout()
    cm_path = os.path.join(config.TRAINING_GRAPHS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved to: {cm_path}")

def train():
    """Runs the full model training and evaluation process."""
    logger.info("Initializing ANN training sequence...")
    
    try:
        # Preprocess dataset
        X, y = preprocess_training_data()
        
        # Split into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=config.RANDOM_STATE, stratify=y
        )
        logger.info(f"Data split - Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples.")
        
        # Build neural network
        model = build_model()
        model.summary(print_fn=lambda x: logger.info(x))
        
        # Setup early stopping callback
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train model
        logger.info("Starting model fitting...")
        history = model.fit(
            X_train, y_train,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            validation_split=config.VAL_SPLIT,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate model on the test set
        logger.info("Evaluating trained model on held-out test data...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        # Predict classes for test data
        y_pred_probs = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        
        # Calculate Precision, Recall, and F1-Score (Weighted Average)
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y_test, y_pred_classes, average='weighted'
        )
        
        # Log evaluation metrics
        logger.info(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        logger.info(f"Weighted Precision: {precision:.4f}")
        logger.info(f"Weighted Recall: {recall:.4f}")
        logger.info(f"Weighted F1-Score: {f1_score:.4f}")
        
        # Save evaluation metrics to JSON
        metrics = {
            "accuracy": float(test_accuracy * 100),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score)
        }
        
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        with open(config.METRICS_PATH, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Evaluation metrics saved to: {config.METRICS_PATH}")
        
        # Save Keras Model
        model.save(config.MODEL_PATH)
        logger.info(f"Trained model saved to: {config.MODEL_PATH}")
        
        # Generate and save evaluation plots
        plot_and_save_graphs(history, y_test, y_pred_classes)
        
        logger.info("ANN model training and saving completed successfully.")
        return True
    except Exception as e:
        logger.critical(f"Critical error during model training: {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    train()
