import os
import numpy as np
import librosa
import math
import glob
import json
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm

dataset_path = 'genres_original'
json_path = 'data_10.json'

sample_rate = 22050
duration = 30  # ( in seconds)
samples_per_track = sample_rate * duration


def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    # dictionary to store data
    data = {
        'mapping': [],
        'mfcc': [],
        'labels': [],
    }

    num_samples_per_segment = int(samples_per_track / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(
        num_samples_per_segment / hop_length)

    # loop through all genre
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # print(i, dirpath)
        # ensure that we are not at the root level
        if dirpath is not dataset_path:

            # save the semantic_labels
            dirpath_components = dirpath.split('/')
            semantic_labels = dirpath_components[-1]
            data['mapping'].append(semantic_labels)

            # process file for a specific genre
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                try:
                    signal, _ = librosa.load(file_path, sr=sample_rate)
                except RuntimeError:
                    continue

                # process segments extracting mfcc and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    mfcc = librosa.feature.mfcc(
                        signal[start_sample:finish_sample], sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

                    mfcc = mfcc.T
                    # store mfcc for segment if it has the expected length
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data['mfcc'].append(mfcc.tolist())
                        data['labels'].append(i-1)
                        print('{},segment:{}'.format(file_path, s))

    with open(json_path, 'w') as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(dataset_path, json_path, num_segments=10)
