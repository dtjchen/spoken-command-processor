import numpy as np
import scipy
import librosa


def onehot_matrix(samples_vec, num_classes):
    """
    >>> onehot_matrix(np.array([1, 0, 3]), 4)
    [[ 0.  1.  0.  0.]
     [ 1.  0.  0.  0.]
     [ 0.  0.  0.  1.]]

    >>> onehot_matrix(np.array([2, 2, 0]), 3)
    [[ 0.  0.  1.]
     [ 0.  0.  1.]
     [ 1.  0.  0.]]

    Ref: http://bit.ly/1VSKbuc
    """
    num_samples = samples_vec.shape[0]

    onehot = np.zeros(shape=(num_samples, num_classes))
    onehot[range(0, num_samples), samples_vec] = 1

    return onehot
    
def wavfile_to_mfccs(wavfile):
    sampling_rate, frames = scipy.io.wavfile.read(wavfile)

    segment_duration_ms = 20
    n_fft = int((segment_duration_ms / 1000.) * sampling_rate)

    hop_duration_ms = 10
    hop_length = int((hop_duration_ms / 1000.) * sampling_rate)

    mfcc_count = 13

    mfccs = librosa.feature.mfcc(
        y=frames,
        sr=sampling_rate,
        n_mfcc=mfcc_count,
        hop_length=hop_length,
        n_fft=n_fft
    )
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    mfccs_and_deltas = np.vstack([mfccs, mfcc_delta, mfcc_delta2])
    
    return mfccs_and_deltas, hop_length, n_fft
