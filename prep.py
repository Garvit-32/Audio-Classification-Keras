import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = 'genres_original/blues/blues.00001.wav'


signal, sr = librosa.load(file)

# librosa.display.waveplot(signal)
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.show()

# signal -> ftt
fft = np.fft.fft(signal)

magnitude = np.abs(fft)
frequncy = np.linspace(0, sr, len(magnitude))

left_frequency = frequncy[:int(len(frequncy)/2)]
left_magnitude = magnitude[:int(len(magnitude)/2)]

# plt.plot(left_frequency, left_magnitude)
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
# plt.show()

# fft -> stft
n_fft = 2048
hop_length = 512
stft = librosa.core.stft(signal, n_fft, hop_length)

# stft => spectrogram
spectrogram = np.abs(stft)
# log_spectrogram
log_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)


# librosa.display.specshow(log_spectrogram, hop_length=hop_length)
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.colorbar()
# plt.show()

# MFCC
MFCCs = librosa.feature.mfcc(
    signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=15)
# librosa.display.specshow(MFCCs, hop_length=hop_length)
# plt.xlabel("Time")
# plt.ylabel("MFCC")
# plt.colorbar()
# plt.show()


# Melspectorgram
melspectrogram = librosa.feature.melspectrogram(
    signal, n_fft=n_fft, hop_length=hop_length)
s_db = librosa.power_to_db(melspectrogram, ref=np.max)
# librosa.display.specshow(s_db, hop_length=hop_length)
# plt.xlabel("Time")
# plt.ylabel("Mel")
# plt.colorbar()
# plt.show()
