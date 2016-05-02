from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
import numpy as np
from . import utils, dataset


def train(summarize=False, data_limit=None):
    reader = dataset.TIMITReader('speech2phonemes')
    X_train, y_train = reader.load_training_data(limit=data_limit)

    # Number of features for each sample in X_train...
    # if each 20ms corresponds to 13 MFCC coefficients + delta + delta2, then 39
    input_dim = X_train.shape[1]
    # Number of distinct classes in the dataset (number of distinct phonemes)
    output_dim = np.max(y_train) + 1
    # Model takes as input arrays of shape (*, input_dim) and outputs arrays
    # of shape (*, hidden_num)
    hidden_num = 256

    y_train_onehot = utils.onehot_matrix(y_train, output_dim)

    # Architecture of the model
    model = Sequential()

    model.add(Dense(input_dim=input_dim, output_dim=hidden_num))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=output_dim))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001))

    stats = model.fit(X_train, y_train_onehot,
        shuffle=True,
        batch_size=256,
        nb_epoch=20,
        verbose=1
    )

    save_model(model)

    if summarize:
        print(model.summary())

        import matplotlib.pyplot as plt
        plt.plot(stats.history['loss'])
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.title('Loss function for %d samples' % X_train.shape[0])
        plt.show()

def test(data_limit=None):
    model = load_model()
    X_test, y_test = dataset.load_test_data()

    out = model.predict_classes(X_test,
        batch_size=256,
        verbose=0
    )

    acc = sum(out == y_test) * 1.0 / len(out)
    print('Accuracy using %d testing samples: %f' % (X_test.shape[0], acc))

def save_model(model):
    reader = dataset.TIMITReader('speech2phonemes')

    with open(reader.params('speech2phonemes_arch', 'json'), 'w') as archf:
        archf.write(model.to_json())

    model.save_weights(
        filepath=reader.params('speech2phonemes_weights', 'h5'),
        overwrite=True
    )

def load_model():
    reader = dataset.TIMITReader('speech2phonemes')

    with open(reader.params('speech2phonemes_arch', 'json')) as arch:
        model = model_from_json(arch.read())
        model.load_weights(dataset.params('speech2phonemes_weights', 'h5'))

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001))
        return model
