# DriverGuard AI - Eye Classifier Model

This directory houses the trained Convolutional Neural Network (CNN) weights, metadata, and evaluation artifacts for DriverGuard AI's real-time eye state classification system.

## Model Summary

- **Architecture**: Custom 3-Block Deep Convolutional Network (`driverguard_eye_cnn`)
  - Conv2D + BatchNormalization + ReLU + MaxPooling + Spatial Dropout
  - GlobalAveragePooling2D
  - Dense (128 units) + Dropout (0.4)
  - Dense (1 unit, Sigmoid activation)
- **Input Dimensions**: `(64, 64, 3)` RGB float32 normalized to `[0.0, 1.0]`
- **Classes**:
  - `0`: **CLOSED** (Eye shut / eyelid closure)
  - `1`: **OPEN** (Eye open / visible pupil & iris)
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam with learning rate reduction on plateau (`ReduceLROnPlateau`)
- **Regularization**: L2 Weight Decay (`1e-4`), Spatial Dropout (0.15 - 0.30), Batch Normalization

## Artifacts in this Directory

- `eye_classifier.keras`: Saved trained Keras model (native `.keras` format).
- `model_metadata.json`: Exported metadata containing exact training hyper-parameters, timestamps, and evaluation metrics.
- `confusion_matrix.png`: Visual confusion matrix from test set evaluation.
- `training_curves.png`: Loss and accuracy curves across training and validation epochs.

## Dataset Setup Instructions

To retrain or benchmark this model on an open-source dataset such as the **MRL Eye Dataset**:

1. Download the dataset from the official academic source:
   - [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset) or [Kaggle MRL Eye Dataset](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset)
2. Extract the dataset into:
   ```
   data/raw/
   ├── Open_Eyes/
   └── Closed_Eyes/
   ```
3. Run the training pipeline:
   ```bash
   python -m src.models.cnn_model
   ```
   or open `notebooks/02_eye_classifier_training.ipynb`.

## Runtime Fallback Behavior

If `models/eye_classifier.keras` is not yet generated, the real-time inference pipeline automatically switches to a calibrated **MediaPipe Eye Aspect Ratio (EAR) Geometric Baseline**. The Streamlit dashboard and telemetry engine clearly reflect this state so system operation is never interrupted.
