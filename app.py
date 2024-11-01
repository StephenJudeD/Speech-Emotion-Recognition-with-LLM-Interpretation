import os
import joblib
import numpy as np
import librosa
import speech_recognition as sr
from pydub import AudioSegment
import scipy.signal
import openai
import requests
from flask import Flask, request, jsonify
from pyngrok import ngrok
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load models and scaler
def load_models():
    ensemble_models = []
    ensemble_paths = [
        "model/multi_model1a.keras",
        "model/multi_model2a.keras",
        "model/multi_model3a.keras",
        "model/multi_model4a.keras",
        "model/multi_model5a.keras",
    ]

    for path in ensemble_paths:
        model = load_model(path)
        ensemble_models.append(model)

    scaler_path = "label/scaler_multi.joblib"
    scaler = joblib.load(scaler_path)

    return ensemble_models, scaler

ensemble_models, scaler = load_models()

# Define feature extraction
def frft(x, alpha):
    N = len(x)
    t = np.arange(N)
    kernel = np.exp(-1j * np.pi * alpha * t**2 / N)
    return scipy.signal.fftconvolve(x, kernel, mode='same')

def extract_features(data, sample_rate):
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13).T, axis=0)
    delta_mfcc = np.mean(librosa.feature.delta(mfcc).T, axis=0)
    acceleration_mfcc = np.mean(librosa.feature.delta(mfcc, order=2).T, axis=0)
    mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)

    alpha_values = np.linspace(0.1, 0.9, 9)
    frft_features = np.array([])

    for alpha in alpha_values:
        frft_result = frft(data, alpha)
        frft_features = np.hstack((frft_features, np.mean(frft_result.real, axis=0)))

    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=data, sr=sample_rate).T, axis=0)

    return np.hstack((mfcc, delta_mfcc, acceleration_mfcc, mel_spectrogram, frft_features, spectral_centroid))

# Normalize audio
def normalize_audio(audio):
    original_max = np.abs(audio).max()
    audio = audio.astype(np.float32)
    normalized_audio = np.clip(audio / original_max, -1.0, 1.0)
    return normalized_audio

# Trim silences
def trim_silences(data, sr, top_db=35):
    trimmed_data, _ = librosa.effects.trim(data, top_db=top_db)
    return trimmed_data

# Function to generate sliding windows
def generate_windows(data, window_size, hop_size, sr):
    num_samples = len(data)
    window_samples = int(window_size * sr)  # Convert window size to samples
    hop_samples = int(hop_size * sr)  # Convert hop size to samples

    # Generate sliding windows
    windows = []
    for i in range(0, num_samples - window_samples + 1, hop_samples):
        window = data[i:i + window_samples]
        windows.append(window)

    return windows

# Emotion prediction
label_encoder_path = "label/label_multi.joblib"
label_encoder = joblib.load(label_encoder_path)

def predict_emotion(audio_file, window_size=3.0, hop_size=0.5):
    data, sr = librosa.load(audio_file, sr=16000)
    data = trim_silences(data, sr)  # Trim silences
    data = normalize_audio(data)  # Normalize audio

    # Generate sliding windows
    windows = generate_windows(data, window_size, hop_size, sr)

    if len(windows) == 0:
        return {label: "0.00%" for label in label_encoder.classes_}  # Return zero probabilities if no windows

    # Initialize an array to capture cumulative probabilities
    emotion_probs = np.zeros(len(label_encoder.classes_))

    for window in windows:
        features = extract_features(window, sr)
        features_scaled = scaler.transform(features.reshape(1, -1))
        features_reshaped = features_scaled.reshape(1, features_scaled.shape[1], 1)

        # Use ensemble models to predict probabilities for each class
        window_probs = np.mean([model.predict(features_reshaped)[0] for model in ensemble_models], axis=0)
        emotion_probs += window_probs

    # Normalize the accumulated probabilities to get averaged probabilities
    emotion_probs /= len(windows)

    # Format probabilities as percentages with two decimal places
    emotion_probability_distribution = {
        label: f"{prob * 100:.2f}%" for label, prob in zip(label_encoder.classes_, emotion_probs)
    }

    return emotion_probability_distribution

def transcribe_audio(audio_file_path):
    # Convert audio file to WAV format
    audio = AudioSegment.from_file(audio_file_path)
    wav_file_path = "/tmp/uploaded_audio.wav"
    audio.export(wav_file_path, format="wav")

    # Transcribe the audio
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_file_path) as source:
        audio_data = recognizer.record(source)
        try:
            transcription = recognizer.recognize_google(audio_data)
            return transcription
        except sr.UnknownValueError:
            return "[unrecognized]"
        except sr.RequestError as e:
            return f"Transcription error: {e}"

def process_audio_file(audio_file):
    # Get emotion prediction
    prediction = predict_emotion(audio_file)

    # Get audio transcription
    transcription = transcribe_audio(audio_file)

    return prediction, transcription

# Define LLM integration
def get_llm_interpretation(emotional_results, transcription):
    openai.api_key = os.getenv('OPENAI_API_KEY')
    prompt = f"""
        You are an expert in audio emotion recognition and analysis. Given the following information:

            Audio data details:
            - Emotional recognition results: {emotional_results}
            - Transcript: {transcription}

            Your task is to provide a comprehensive and insightful interpretation of the emotional content captured in the audio data, considering both the emotion recognition results and the transcript.

            In your response, please:

            <thinking>
            - Summarize the key emotions detected by the model and their relative strengths.
            - Discuss how the emotions expressed in the transcript align with or differ from the model's predictions.
            - Analyze any notable patterns or trends in the emotional content, especially changes in emotional state over time, differences between speakers, or contextual factors influencing the emotions.
            - Highlight the most salient and informative aspects of the emotional data that would be valuable for understanding the overall emotional experience captured in the audio.
            </thinking>

            <result>
            Based on the provided information, your comprehensive and insightful interpretation of the emotional content in the audio data is:
            </result>
        """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000  # Limit the response to 100 tokens
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    audio_file_path = '/tmp/uploaded_audio.wav'
    file.save(audio_file_path)

    predictions, transcription = process_audio_file(audio_file_path)
    llm_interpretation = get_llm_interpretation(predictions, transcription)

    return jsonify({
        'emotion_probabilities': predictions,
        'transcription': transcription,
        'llm_interpretation': llm_interpretation
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
