import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


audio_file = 'genres_original/blues/blues.00001.wav'

signal, sr = librosa.load(audio_file)


filters_bank = librosa.filters.mel(n_fft=2048, sr=sr, n_mels=10)


# (10,1025) (no of mel,nyquist frequency=1024 + 1)
print(filters_bank.shape)

# plt.figure(figsize=(20, 10))
# librosa.display.specshow(filters_bank, x_axis='linear')
# plt.colorbar()
# plt.show()


mel_spectrogram = librosa.feature.melspectrogram(signal)

# (128,1293) no of mel filter, no. of frame or temperal bin that we extract from signal
print(mel_spectrogram.shape)


# log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
log_mel_spectrogram = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)

librosa.display.specshow(log_mel_spectrogram, x_axis='time', y_axis='mel')
plt.colorbar()
plt.show()
