from flask import Flask, request, jsonify
import librosa
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load models
svm_model = joblib.load("svm_model.pkl")
mlp = load_model("mlp_model.h5")

def extract_mfcc(audio_path):
    audio, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=60)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean.reshape(1, -1)

def get_alert_level(audio_path):
    features = extract_mfcc(audio_path)
    svm_pred = svm_model.predict(features)[0]
    mlp_pred = np.argmax(mlp.predict(features), axis=1)[0]

    if svm_pred == 1 and mlp_pred == 1:
        return "High Alert"
    elif svm_pred == 1 or mlp_pred == 1:
        return "Moderate Alert"
    else:
        return "Normal"

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    temp_path = "temp.wav"
    file.save(temp_path)

    alert_level = get_alert_level(temp_path)
    os.remove(temp_path)

    return jsonify({'alert_level': alert_level})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)