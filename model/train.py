# Read the TIMIT training data

# Keras to train given a generator with the training data (list of [MFCC, label] pairs)

# Save the weights of the model (as well as the normalization values of the mean of each entry in an MFCC vector)
from dataset import load_training_data, load_test_data
import sklearn


def train():
    X_train, y_train = load_training_data()

    print(X_train)
    print(y_train)

    X_test, y_test = load_test_data()

    print(X_test)
    print(y_test)
