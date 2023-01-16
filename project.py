import sys
import skimage
import tensorflow as tf
import numpy as np
import os
import librosa

# Settings
n_mels = 128
n_fft = 2048
hop_length = 512
fmax = 8000

def extract_melspectrogram(audio_path):
    audio, sr = librosa.load(audio_path)
    melspectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmax=fmax)
    melspectrogram = np.log(melspectrogram + 1e-9) # add small number to avoid log(0)
    return melspectrogram

def resize_melspectrogram(melspectrogram):
    return skimage.transform.resize(melspectrogram, (128, 128))


if __name__ == '__main__':
    
    audio_path = sys.argv[1]
    melspectrogram = extract_melspectrogram(audio_path)
    
    melspectrogram = resize_melspectrogram(melspectrogram)
    
    model = tf.keras.models.load_model('covid_cough_model.h5')
    
    prediction = model.predict(melspectrogram.reshape(1, 128, 128, 1))
    
    print(prediction)
     
    if prediction[0][1] > prediction[0][0]:
        print("Cough is COVID-19 positive")
    else:
        print("Cough is COVID-19 negative")
