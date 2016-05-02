import os
import glob
import librosa
import scipy
import numpy as np
import itertools
from sklearn import preprocessing
from . import utils


class TIMITReader:
    def __init__(self, model_name):
        """
        Necessary env. vars.:
            - PROJECT_ROOT
            - TIMIT_TRAINING_PATH
            - TIMIT_TESTING_PATH
            - PHONE_LIST_PATH
        """
        self.train_dataset_path = os.environ['TIMIT_TRAINING_PATH']
        self.test_dataset_path = os.environ['TIMIT_TESTING_PATH']

        data_root = os.path.join(os.environ['PROJECT_ROOT'], 'model', 'data')

        if model_name == 'speech2phonemes':
            self.reader_func = self._read_labeled_wavfiles
            self.data_dir = os.path.join(data_root, 'speech2phonemes')

        elif model_name == 'phonemes2text':
            self.reader_func = self._read_labeled_phnfiles
            self.data_dir = os.path.join(data_root, 'phonemes2text')

    def params(self, name, ext='npy'):
        return os.path.join(self.data_dir, name + '.%s' % ext)

    def load_train_data(self, limit=None):
        """
        For self.model == 'speech2phonemes', returns:
            X_train --> [num_of_training_mfcc_vectors, 20]
            y_train --> [num_of_training_mfcc_vectors, 1]
        """
        print('Loading training data...')

        if all(map(os.path.exists, [self.params('X_train'), self.params('y_train')])):
            print('Found .npy files for X_train and y_train. Loading...')
            X_train = np.load(self.params('X_train'))
            y_train = np.load(self.params('y_train'))

        else:
            print('Did not find .npy files for X_train and y_train. Parsing dataset...')
            X_train_raw, y_train = self.reader_func(self.train_dataset_path)

            print('Normalizing X_train around each MFCC coefficient\'s mean...')
            scaler = preprocessing\
                .StandardScaler(with_mean=True, with_std=False)\
                .fit(X_train_raw)

            X_train = scaler.transform(X_train_raw)

            np.save(self.params('mfcc_means'), scaler.mean_)
            np.save(self.params('X_train'), X_train)
            np.save(self.params('y_train'), y_train)

        if limit:
            print('Returning %d/%d of the training data...' % (limit, X_train.shape[0]))
            X_train = X_train[:limit, :]
            y_train = y_train[:limit]

        return X_train, y_train

    def load_test_data(self, limit=None):
        """
        For self.model == 'speech2phonemes', returns:
            X_test  --> [num_of_testing_mfcc_vectors, 20]
            y_test  --> [num_of_testing_mfcc_vectors, 1]
        """
        if all(map(os.path.exists, [self.params('X_test'), self.params('y_test')])):
            print('Found .npy files for X_test and y_test. Loading...')
            X_test = np.load(self.params('X_test'))
            y_test = np.load(self.params('y_test'))

        else:
            print('Did not find .npy files for X_test and y_test. Parsing dataset...')
            X_test_raw, y_test = self.reader_func(self.test_dataset_path)

            # Use the MFCC means from the training set to normalize X_train
            scaler = preprocessing.StandardScaler(with_mean=True, with_std=False)
            scaler.mean_ = np.load(self.params('mfcc_means'))

            X_test = scaler.fit_transform(X_test_raw)

            np.save(self.params('X_test'), X_test)
            np.save(self.params('y_test'), y_test)

        if limit:
            print('Returning %d/%d of the testing data...' % (limit, X_test.shape[0]))
            X_test = X_test[:limit, :]
            y_test = y_test[:limit]

        return X_test, y_test

    def _read_labeled_wavfiles(self, root_timit_path):
        wavfiles = sorted(glob.glob(root_timit_path + '/*/*/*.WAV'))
        labels_files = sorted(glob.glob(root_timit_path + '/*/*/*.PHN'))

        X, y = [], []

        for wf, lf in zip(wavfiles, labels_files):
            for mfccs, label in self._read_labeled_wavfile(wf, lf):
                X.append(mfccs)
                y.append(label)

        # Convert phoneme strings in y_train to class numbers
        phoneme_classes = self.load_unique_phonemes_as_class_numbers()
        y = [phoneme_classes[y[i]] for i in range(len(y))]

        return np.array(X), np.array(y)

    def _read_labeled_wavfile(self, wavfile, labels_file):
        """Map each 20ms recording to a single label."""
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
                start_frame, end_frame, label = self._parse_timit_line(line)

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

    def _read_labeled_phnfiles(self, root_timit_path):
        phn_files = sorted(glob.glob(root_timit_path + '/*/*/*.PHN'))
        word_files = sorted(glob.glob(root_timit_path + '/*/*/*.WRD'))

        # Each word is mapped to a class number
        word_classes = self.load_unique_words_as_class_numbers()

        # Used to get one-hot vectors for each word; this gives its size (4893)
        num_distinct_words = len(word_classes)

        # Number of words disected into phonemes in TIMIT over 4620 files
        # (some words are duplicates; i.e. different phoneme combinations)
        num_samples = 39834

        # Max phonemes per word (in the dataset, the largest is "encyclopedias"
        # with 17... we'll go with a few more)
        num_phones_per_word = 25

        X = np.zeros((num_samples, num_phones_per_word, num_distinct_words))
        y = np.zeros((num_samples, num_distinct_words))

        """

        print(X.shape)
        print(y.shape)
        exit()

        X, y = [], []

        maxsize = 0
        fax = ""
        num_words = 0
        num_files = 0

        for pf, wf in zip(phn_files, word_files):
            num_files += 1

            for word, phones_in_word in self._read_labeled_phnfile(pf, wf):
                num_words += 1

                if len(phones_in_word) > maxsize:
                    maxsize = len(phones_in_word)
                    fax = word

        print(fax, maxsize, num_words, num_files)
        exit()
        """

    def _read_labeled_phnfile(self, phn_file, word_file):
        """Map each word to a list of phones (one phone per frame)"""
        phns = []
        with open(phn_file, 'r') as f:
            for line in f.readlines():
                start_frame, end_frame, label = self._parse_timit_line(line)

                phn_frames = end_frame - start_frame
                phns.extend([label] * phn_frames)

        with open(word_file, 'r') as f:
            for line in f.readlines():
                start_frame, end_frame, label = self._parse_timit_line(line)

                with_repeats = phns[start_frame:end_frame]
                phns = [k[0] for k in itertools.groupby(with_repeats)]

                yield label, phns

    def _parse_timit_line(self, line):
        start_frame, end_frame, label = line.split(' ')

        return int(start_frame), int(end_frame), label.strip('\n')

    def load_unique_phonemes_as_class_numbers(self):
        phonemes = {}

        with open(os.environ['PHONE_LIST_PATH'], 'r') as f:
            class_number = 0

            for ph in map(lambda p: p.strip(), f.readlines()):
                phonemes[ph] = class_number
                class_number += 1

        return phonemes

    def load_unique_words_as_class_numbers(cls):
        words = {}

        with open(os.environ['WORD_LIST_PATH'], 'r') as f:
            class_number = 0

            for word in map(lambda w: w.strip(), f.readlines()):
                words[word] = class_number
                class_number += 1

        return words
