# Read the TIMIT training data

# Keras to train given a generator with the training data (list of [MFCC, label] pairs)

# Save the weights of the model (as well as the normalization values of the mean of each entry in an MFCC vector)

import os
import glob
import librosa
import scipy
import numpy as np


class InputVector:
    def __init__(self, mfcc, label):
        self.mfcc = mfcc
        self.label = label

def train():
    for input_vecs_in_wavfile in read_wavfiles():
        for input_vec in input_vecs_in_wavfile:
            print(input_vec.mfcc)
            print(input_vec.label)
            break

def read_wavfiles():
    train_dir = os.environ['TIMIT_TRAINING_DATA_PATH']

    wavfiles = sorted(glob.glob(train_dir + '/*/*/*.WAV'))
    labels_files = sorted(glob.glob(train_dir + '/*/*/*.PHN'))

    for wavfile, labels_file in zip(wavfiles, labels_files):
        yield read_wavfile(wavfile, labels_file)

def read_wavfile(wavfile, labels_file):
    sampling_rate, frames = scipy.io.wavfile.read(wavfile)

    segment_duration_ms = 20
    segment_duration_frames = int((segment_duration_ms / 1000.) * sampling_rate)

    hop_duration_ms = 10
    hop_duration_frames = int((hop_duration_ms / 1000.) * sampling_rate)

    mfccs = librosa.feature.mfcc(
        y=frames,
        sr=sampling_rate,
        hop_length=hop_duration_frames,
        n_fft=segment_duration_frames
    )

    ############################################

    # Pass through the file with the phones
    labels = []

    with open(labels_file, 'r') as f:
        for line in f.readlines():
            start_frame, end_frame, label = line.split(' ')
            start_frame, end_frame = int(start_frame), int(end_frame)

            phn_frames = end_frame - start_frame
            labels.extend([label] * phn_frames)

    ###########################################

    classified = []
    curr_frame = curr_mfcc = 0

    while (curr_frame < (len(labels) - segment_duration_frames)):
        label = max(labels[curr_frame:(curr_frame + segment_duration_frames)])
        input_vec = InputVector(mfcc=mfccs[:,curr_mfcc], label=label)
        classified.append(input_vec)

        curr_mfcc += 1
        curr_frame += hop_duration_frames

    return classified


if __name__ == '__main__':
    train()
