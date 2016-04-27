import os
import glob
import librosa
import scipy
import numpy as np
from sklearn import preprocessing


class ModelDataPath(type):
    data_dir = os.path.join(os.environ['PROJECT_ROOT'], 'model', 'data')

    def __getattr__(self, name):
        return os.path.join(self.data_dir, name + '.npy')

class ModelData(object):
    """
    Allows us to map model data to their respective .npy paths.

    Ex.
        ModelData.X_train --> '~/speech-analysis/model/params/X_train.npy'
    """
    __metaclass__ = ModelDataPath

def load_training_data():
    """
    Returns:
        X_train --> [num_of_training_mfcc_vectors, 20]
        y_train --> [num_of_training_mfcc_vectors, 1]
    """
    print('Loading training data...')

    if all(os.path.exists(p) for p in [ModelData.X_train, ModelData.y_train]):
        print('Found .npy files for X_train and y_train. Loading...')
        X_train = np.load(ModelData.X_train)
        y_train = np.load(ModelData.y_train)

    else:
        print('Did not find .npy files for X_train and y_train. Parsing dataset...')
        X_train_raw, y_train = read_labeled_wavfiles(os.environ['TIMIT_TRAINING_PATH'])

        print('Normalizing X_train around each MFCC coefficient\'s mean...')
        scaler = preprocessing\
            .StandardScaler(with_mean=True, with_std=False)\
            .fit(X_train_raw)

        X_train = scaler.transform(X_train_raw)

        np.save(ModelData.mfcc_means, scaler.mean_)
        np.save(ModelData.X_train, X_train)
        np.save(ModelData.y_train, y_train)

    return X_train, y_train

def load_test_data():
    """
    Returns:
        X_test  --> [num_of_testing_mfcc_vectors, 20]
        y_test  --> [num_of_testing_mfcc_vectors, 1]
    """
    if all(os.path.exists(p) for p in [ModelData.X_test, ModelData.y_test]):
        print('Found .npy files for X_test and y_test. Loading...')
        X_test = np.load(ModelData.X_test)
        y_test = np.load(ModelData.y_test)

    else:
        print('Did not find .npy files for X_test and y_test. Parsing dataset...')
        X_test_raw, y_test = read_labeled_wavfiles(os.environ['TIMIT_TESTING_PATH'])

        # Use the MFCC means from the training set to normalize X_train
        scaler = preprocessing.StandardScaler(with_mean=True, with_std=False)
        scaler.mean_ = np.load(ModelData.mfcc_means)

        X_test = scaler.fit_transform(X_test_raw)

        np.save(ModelData.X_test, X_test)
        np.save(ModelData.y_test, y_test)

    return X_test, y_test

def load_unique_phonemes_as_class_numbers():
    this = load_unique_phonemes_as_class_numbers

    if not hasattr(this, 'phonemes'):
        this.phonemes = {}

        with open(os.environ['PHONE_LIST_PATH'], 'r') as f:
            class_number = 0

            for ph in map(lambda p: p.strip(), f.readlines()):
                this.phonemes[ph] = class_number
                class_number += 1

    return this.phonemes

def read_labeled_wavfiles(root_timit_path):
    wavfiles = sorted(glob.glob(root_timit_path + '/*/*/*.WAV'))
    labels_files = sorted(glob.glob(root_timit_path + '/*/*/*.PHN'))

    X, y = [], []

    for wf, lf in zip(wavfiles, labels_files):
        for mfccs, label in read_labeled_wavfile(wf, lf):
            X.append(mfccs)
            y.append(label)

    # Convert phoneme strings in y_train to class numbers
    phonemes = load_unique_phonemes_as_class_numbers()
    y = [phonemes[y[i]] for i in range(len(y))]

    return np.array(X), np.array(y)

def read_labeled_wavfile(wavfile, labels_file):
    sampling_rate, frames = scipy.io.wavfile.read(wavfile)

    segment_duration_ms = 20
    segment_duration_frames = int((segment_duration_ms / 1000.) * sampling_rate)

    hop_duration_ms = 10
    hop_duration_frames = int((hop_duration_ms / 1000.) * sampling_rate)

    mfcc_count = 13

    mfccs = librosa.feature.mfcc(
        y=frames,
        sr=sampling_rate,
        n_mfcc=mfcc_count,
        hop_length=hop_duration_frames,
        n_fft=segment_duration_frames
    )
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    mfccs_and_deltas = np.vstack([mfccs, mfcc_delta, mfcc_delta2])

    ############################################

    # Pass through the file with the phones
    labels = []

    with open(labels_file, 'r') as f:
        for line in f.readlines():
            start_frame, end_frame, label = line.split(' ')
            start_frame, end_frame = int(start_frame), int(end_frame)
            label = label.strip('\n')

            phn_frames = end_frame - start_frame
            labels.extend([label] * phn_frames)

    ###########################################

    classified = []
    curr_frame = curr_mfcc = 0

    while (curr_frame < (len(labels) - segment_duration_frames)):
        label = max(labels[curr_frame:(curr_frame + segment_duration_frames)])

        yield mfccs_and_deltas[:,curr_mfcc], label

        curr_mfcc += 1
        curr_frame += hop_duration_frames
