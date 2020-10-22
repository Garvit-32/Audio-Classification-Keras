import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

audio_file = 'genres_original/blues/blues.00001.wav'

signal, sr = librosa.load(audio_file)

mfccs = librosa.feature.mfcc(signal, n_mfcc=13, sr=sr)
print(mfccs.shape)


plt.figure(figsize=(25, 10))

librosa.display.specshow(mfccs, x_axis='time', sr=sr)
plt.colorbar()
plt.show()
