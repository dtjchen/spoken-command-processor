import os


data_dir = os.path.join(os.environ['PROJECT_ROOT'], 'model', 'data')

def params(name, ext='npy'):
    """Get the path to any data files used by the model (weights, etc.)

    >> params('X_train')
    ~/speech-analysis/model/data/X_train.npy

    >> params('arch', 'json')
    ~/speech-analysis/model/data/arch.json
    """
    return os.path.join(data_dir, name + '.%s' % ext)
