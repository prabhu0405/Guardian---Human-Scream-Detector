from flask import Flask, request, jsonify
import numpy as np
import joblib
import os
from scipy.io import wavfile
from python_speech_features import mfcc
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load models
svm_model = joblib.load("svm_model.pkl")
mlp_model = load_model("mlp_model.h5")

def extract_mfcc(file_path):
    rate, signal = wavfile.read(file_path)
    
    # Ensure the signal is mono
    if len(signal.shape) > 1:
        signal = signal[:, 0]

    # Extract MFCC with proper NFFT to avoid truncation warning
    mfcc_feat = mfcc(signal, samplerate=rate, numcep=26, nfft=2048)
    mfcc_mean = np.mean(mfcc_feat, axis=0)
    return mfcc_mean.reshape(1, -1)

def get_alert_level(file_path):
    features = extract_mfcc(file_path)
    
    # Predict using SVM and MLP
    svm_pred = svm_model.predict(features)[0]
    mlp_pred = np.argmax(mlp_model.predict(features), axis=1)[0]

    # Decide alert level
    if svm_pred == 1 and mlp_pred == 1:
        return "High Alert"
    elif svm_pred == 1 or mlp_pred == 1:
        return "Moderate Alert"
    else:
        return "Normal"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    f = request.files['file']
    file_path = 'temp.wav'
    f.save(file_path)

    try:
        alert = get_alert_level(file_path)
        return jsonify({'alert_level': alert})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)
