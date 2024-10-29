# Overview of the Multi-Model Architecture for Time Series Classification

This project implements a multi-model architecture for time series classification using Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM) networks, Gated Recurrent Units (GRU), and attention mechanisms. The models are designed to process time series data effectively, leveraging both spatial and temporal features.

## Models Overview

### Model 1: CNN + Bidirectional LSTM + Bidirectional GRU with Attention
- **Architecture**: 
  - Convolutional layers for feature extraction.
  - Bidirectional LSTM layers to capture temporal dependencies.
  - Attention layers to focus on important features.
  - Bidirectional GRU layers for further temporal processing.
  - Dense output layer for classification.
  
### Model 2: CNN Only
- **Architecture**: 
  - A straightforward CNN architecture for feature extraction followed by a dense layer for classification.

### Model 3: CNN + Bidirectional LSTM
- **Architecture**: 
  - Similar to Model 1 but without GRU layers. It uses CNN for feature extraction followed by Bidirectional LSTM and attention.

### Model 4: CNN + GRU
- **Architecture**: 
  - Combines CNN for feature extraction with GRU layers for temporal processing, followed by a dense layer for classification.

### Model 5: CNN + Multi-Head Attention + Bidirectional LSTM + Bidirectional GRU
- **Architecture**: 
  - Similar to Model 1 but incorporates multi-head attention to enhance the model's ability to focus on different parts of the input sequence.

## Model Summary

Each model is compiled with the following parameters:
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy

## Training and Evaluation

The models are trained using the following callbacks:
- **EarlyStopping**: Stops training when a monitored metric has stopped improving.
- **ReduceLROnPlateau**: Reduces the learning rate when a metric has stopped improving.
- **ModelCheckpoint**: Saves the model after every epoch if the validation loss improves.

## Data Preprocessing
- **Label Encoding**: Converts categorical labels into numerical format.
- **Min-Max Scaling**: Scales the features to a range between 0 and 1.

## Requirements
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- tqdm

## Usage
1. **Data Preparation**: Load and preprocess your time series data.
2. **Model Training**: Instantiate and train the models using the training data.
3. **Model Evaluation**: Evaluate the models on validation/test data to assess performance.

## Final Model Performance
Here are the final statistics for each model after training:

### Model 1
- **Final Accuracy**: 0.9650
- **Final Validation Accuracy**: 0.9085
- **Final Loss**: 0.0957
- **Final Validation Loss**: 0.3040

### Model 2
- **Final Accuracy**: 0.9727
- **Final Validation Accuracy**: 0.8793
- **Final Loss**: 0.0761
- **Final Validation Loss**: 0.4433

### Model 3
- **Final Accuracy**: 0.9751
- **Final Validation Accuracy**: 0.9036
- **Final Loss**: 0.0703
- **Final Validation Loss**: 0.3388

### Model 4
- **Final Accuracy**: 0.9714
- **Final Validation Accuracy**: 0.9012
- **Final Loss**: 0.0817
- **Final Validation Loss**: 0.3441

### Model 5
- **Final Accuracy**: 0.9757
- **Final Validation Accuracy**: 0.8980
- **Final Loss**: 0.0666
- **Final Validation Loss**: 0.3886
