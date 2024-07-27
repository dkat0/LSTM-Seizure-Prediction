# Epileptic Seizure Detection Using RNN, LSTM, and GRU Models

This project focuses on detecting epileptic seizures from EEG data using advanced deep learning models. The models implemented include Recurrent Neural Networks (RNN), Long Short-Term Memory networks (LSTM), and Gated Recurrent Unit networks (GRU). The project employs sophisticated data preprocessing, hyperparameter tuning, and model evaluation techniques to achieve high accuracy in seizure detection.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Data Preparation](#data-preparation)
- [Model Architectures](#model-architectures)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Epilepsy is a chronic neurological disorder characterized by recurrent seizures, and accurate detection of seizures from EEG data can significantly improve patient care and management. This project aims to build robust deep learning models to detect seizures from EEG signals, leveraging state-of-the-art techniques in model building and hyperparameter optimization.

## Features
- **Data Preprocessing**: Combining EEG data chunks from the same patient into a single sequence and handling imbalanced data using SMOTE.
- **Model Architectures**: Implementing and comparing RNN, LSTM, and GRU models with multiple layers and dropout for regularization.
- **Hyperparameter Tuning**: Utilizing Keras Tuner for hyperparameter optimization to find the best model configurations.
- **Early Stopping**: Preventing overfitting by stopping training when validation loss stops improving.
- **Visualization**: Plotting EEG sequences and training/validation accuracy over epochs.
- **Evaluation**: Comprehensive model evaluation using accuracy, ROC curves, and other relevant metrics.

## Data Preparation
The EEG data is sourced from the [UCI Epileptic Seizure Recognition dataset](https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition). The data preprocessing steps include:
1. **Combining EEG Chunks**: Merging 23 chunks from the same patient into one continuous sequence.
2. **Handling Imbalanced Data**: Using SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset and ensure equal representation of seizure and non-seizure samples.

## Model Architectures
Three types of deep learning models are implemented and compared:
1. **Recurrent Neural Networks (RNN)**
2. **Long Short-Term Memory Networks (LSTM)**
3. **Gated Recurrent Unit Networks (GRU)**

Each model architecture can have up to three layers, with tunable units per layer and optional dropout for regularization.

## Hyperparameter Tuning
Keras Tuner is used for hyperparameter optimization. The following hyperparameters are tuned:
- Number of layers (1 to 3)
- Units per layer (32 to 128, step 32)
- Dropout rate (0.1 to 0.5, step 0.1)
- Optimizer (Adam, RMSprop)

## Training and Evaluation
The models are trained using the following techniques:
- **Early Stopping**: Monitors validation loss and stops training when it stops improving.
- **Model Checkpointing**: Saves the best model based on validation accuracy during training.

## Results
The best performing models for RNN, LSTM, and GRU are selected based on validation accuracy. The results demonstrate high accuracy and robustness in detecting seizures from EEG data.

### Best Accuracies
- **Best RNN Accuracy**: 0.83125
- **Best LSTM Accuracy**: 0.975
- **Best GRU Accuracy**: 0.96875

## Usage
### Prerequisites
- Python 3.7+
- TensorFlow
- Keras
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Keras Tuner

### Installation
```bash
git clone https://github.com/dkat0/LSTM-Seizure-Prediction.git
cd LSTM-Seizure-Prediction
pip install -r requirements.txt
python prediction.py