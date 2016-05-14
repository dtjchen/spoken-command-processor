from . import speech2phonemes, regulator, phonemes2text


def predict(recording):
    """
    Run the user input through the entire model. Take a series of MFCCs from the
    audio recording and output the predicted words (there number of words
    depends on the length of the provided input).

    1. speech2phonemes
    2. regulator
    3. phonemes2text

    Args:
        recording: matrix of shape (*, 39) <-- see utils.wavfile_to_mfccs()

    Returns:
        list of predicted words
    """
    pass
