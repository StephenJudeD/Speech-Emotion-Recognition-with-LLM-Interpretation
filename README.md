# README: Speech Emotion Recognition Using Multi-Model Architecture

This project implements a multi-model architecture for Speech Emotion Recognition (SER) using various neural network techniques, including Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM) networks, Gated Recurrent Units (GRU), and attention mechanisms. The models are designed to process audio data from six different datasets, extracting features that capture emotional nuances in speech.

## Feature Extraction

The feature extraction process is crucial for SER. The function `get_features` is designed to:
- Load audio files and normalize them.
- Trim silence and ensure each audio sample is exactly 3 seconds long.
- Extract various features from the audio, including:
  - Features from the original audio.
  - Features with added background noise.
  - Features from audio that has been stretched and pitched.
  - Features from shifted audio.
  - Features from audio with echo and reverb effects.

This comprehensive feature extraction helps the models learn from diverse audio conditions, improving their robustness and accuracy.

## Models Overview

### Model 1: CNN + Bidirectional LSTM + Bidirectional GRU with Attention
- **Final Accuracy**: 0.9650
- **Final Validation Accuracy**: 0.9085
- **Final Loss**: 0.0957
- **Final Validation Loss**: 0.3040

**Evaluation**: 
This model excels in capturing both spatial and temporal features, making it highly effective for complex emotional patterns in speech. The attention mechanism enhances its ability to focus on significant features, resulting in high accuracy and validation performance.

### Model 2: CNN Only
- **Final Accuracy**: 0.9727
- **Final Validation Accuracy**: 0.8793
- **Final Loss**: 0.0761
- **Final Validation Loss**: 0.4433

**Evaluation**: 
The CNN-only model achieves high accuracy, indicating effective feature extraction. However, its lower validation accuracy suggests it may not generalize as well to unseen data, making it suitable for simpler datasets or tasks.

### Model 3: CNN + Bidirectional LSTM
- **Final Accuracy**: 0.9751
- **Final Validation Accuracy**: 0.9036
- **Final Loss**: 0.0703
- **Final Validation Loss**: 0.3388

**Evaluation**: 
This model effectively combines CNNs with LSTMs, capturing both spatial and temporal features. It shows strong validation accuracy, indicating good generalization. The absence of GRUs and attention may limit its performance on more complex datasets compared to Model 1.

### Model 4: CNN + GRU
- **Final Accuracy**: 0.9714
- **Final Validation Accuracy**: 0.9012
- **Final Loss**: 0.0817
- **Final Validation Loss**: 0.3441

**Evaluation**: 
The CNN + GRU model performs well, with validation accuracy similar to Model 3. However, it lacks the attention mechanism, which may limit its ability to focus on important features in the data. This model is effective for datasets where GRUs can capture the necessary temporal dynamics.

### Model 5: CNN + Multi-Head Attention + Bidirectional LSTM + Bidirectional GRU
- **Final Accuracy**: 0.9757
- **Final Validation Accuracy**: 0.8980
- **Final Loss**: 0.0666
- **Final Validation Loss**: 0.3886

**Evaluation**: 
This model incorporates multi-head attention, allowing it to focus on multiple aspects of the input sequence simultaneously. While it achieves high accuracy, its validation accuracy is slightly lower than Model 3, suggesting a potential risk of overfitting. It is best suited for complex datasets where attention can significantly enhance performance.

## Conclusion

In summary, each model has its strengths and weaknesses based on the architecture and techniques used:

- **Model 1** excels in capturing complex emotional patterns due to its combination of CNN, LSTM, GRU, and attention.
- **Model 2** is effective for simpler tasks but may not generalize as well.
- **Model 3** balances spatial and temporal features well, making it a strong contender.
- **Model 4** is effective but lacks the attention mechanism, which may limit its performance on complex datasets.
- **Model 5** offers advanced capabilities with multi-head attention but may risk overfitting.

Choosing the right model depends on the specific characteristics of the datasets and the importance of capturing emotional nuances in speech.

## Requirements
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- tqdm
- librosa (for audio processing)

## Usage
1. **Data Preparation**: Load and preprocess your audio data using the `get_features` function.
2. **Model Training**: Instantiate and train the models using the training data.
3. **Model Evaluation**: Evaluate the models on validation/test data to assess performance.
