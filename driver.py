import sys, os

from model import speech2phonemes, phonemes2text
from model import dataset
import processor


# Suppress stderr from the output
null = open(os.devnull,'wb')
sys.stderr = null


def test_speech2phonemes():
    #speech2phonemes.train(summarize=True, data_limit=100000)
    speech2phonemes.test()

def test_phonemes2text():
    #phonemes2text.train(summarize=False, data_limit=None)
    phonemes2text.test()


if __name__ == '__main__':
    #test_speech2phonemes()
    #test_phonemes2text()
    #processor.register()

    #processor.register()
    processor.parse()
