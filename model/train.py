# Read the TIMIT training data

# Keras to train given a generator with the training data (list of [MFCC, label] pairs)

# Save the weights of the model (as well as the normalization values of the mean of each entry in an MFCC vector)
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Input, merge
from keras.models import Sequential
from keras.optimizers import SGD, Adadelta, Adam
import matplotlib.pyplot as P
import numpy as np
import dataset

def dec2onehot(dec, output_dim):
    num=dec.shape[0]
    ret=np.zeros((num,output_dim))
    ret[range(0,num),dec]=1
    return ret

def train():
    X_train, y_train = dataset.load_training_data()
    X_test, y_test = dataset.load_test_data()
    classes = dataset.load_unique_phonemes_as_class_numbers()
    #print(max(classes.values()))
    #print(classes)
    #exit()

    input_dim = X_train.shape[1]
    output_dim = np.max(y_train) + 1
    hidden_num = 256

    TRAINING_COUNT = 1000000
    X_train = X_train[:TRAINING_COUNT, :]
    y_train = y_train[:TRAINING_COUNT]

    y_train_out = dec2onehot(y_train, output_dim)

    model = Sequential()

    model.add(Dense(input_dim=input_dim, output_dim=hidden_num))
    model.add(Activation('sigmoid'))
    model.add(Dense(output_dim=output_dim))
    model.add(Activation('softmax'))

    #optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
    optimizer = Adam(lr=0.001)
    loss ='categorical_crossentropy'

    model.compile(loss=loss, optimizer=optimizer)
    print model.summary()

    hist = model.fit(X_train, y_train_out, shuffle=True, batch_size=256, nb_epoch=200, verbose=1)

    out = model.predict_classes(X_test, batch_size=256, verbose=0)
    #print(out[:50])
    #print(max(out))
    print(sum(out == y_test) * 1.0 / len(out))

    #P.plot(hist.history['loss'])
    #P.show()
