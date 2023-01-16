# Goal: Predict whether a cough is COVID-19 positive or negative

import sys
import skimage
import tensorflow as tf
import numpy as np
import os
import librosa

# Settings
n_mels = 64 # Number of Mel banks to generate
n_fft = 1024 # Interval we consider to apply FFT. Measured in # of samples
hop_length = 441 # Sliding window for FFT. Measured in # of samples
fmax = 22 # Maximum frequency we want to consider
sampling_rate = 44100 # 44.1kHz sampling rate
fmax = 22050

def extract_melspectrogram(audio_path):
    audio, sr = librosa.load(audio_path)
    melspectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmax=fmax)
    melspectrogram = np.log(melspectrogram + 1e-9) # add small number to avoid log(0)
    return melspectrogram

def resize_melspectrogram(melspectrogram):
    return skimage.transform.resize(melspectrogram, (128, 128))


if __name__ == '__main__':
    
    # Get mel-spectrogram from audio file
    audio_path = sys.argv[1]
    melspectrogram = extract_melspectrogram(audio_path)
    melspectrogram = resize_melspectrogram(melspectrogram)
    
    # Load model
    model = tf.keras.models.load_model('covid_cough_classifier.h5')
    # Make prediction
    prediction = model.predict(melspectrogram.reshape(1, 128, 128, 1))
    
    print("Prediction: covid-19 positive = {:.2f}%, covid-19 negative = {:.2f}%".format(prediction[0][1] * 100 , prediction[0][0] * 100))
    print("Conclusion:")
    if prediction[0][1] > prediction[0][0]:
        print(" Cough is COVID-19 positive")
    else:
        print(" Cough is COVID-19 negative")
