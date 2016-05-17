# Spoken Command Processor*

Applications like Siri and Cortana allow users to specify directives by transcribing speech and mapping it to a series of known commands. For example, asking Siri "what song is this" and holding your phone close to the speaker that is playing the music will prompt it to send soundbytes of the song to its classification models.

We attempted a similar effect by implementing a mechanism that records a user's spoken command (in our case, a single word) and maps it to a directive that may be sent to any given destination. For instance, the user may record the word "lights" and instruct the service to send a message with "keeplightson" to a remote server that listens for such commands. Consequently, the user would be able to test the system at a later time by repeating the same word and expecting the server to perform the relevant task.

## Transcription Model Architecture

The transcription model is formed by two submodels: the first maps 20-ms soundbytes to individual phonemes and the second groups those phonemes into words from its dictionary (see `model/__init__.py`). The models are feed-forward neural networks built using the Keras deep learning library, which provides a high-level API to chain layers sequentially.

Once a given neural network's architecture is defined, it is trained using the ubiquitous backpropagation algorithm, which implements the chain rule backwards to incrementally modify the "neurons" so as to reduce the error of the network.

![Backpropagation](docs/img/cs231n_backprop.png)

_(Source: Stanford University's [CS231n: "Convolutional Neural Networks for Visual Recognition"](http://cs231n.github.io/optimization-2/))_

Aside from the number of neurons in the network and the architecture of the layers, the engineer must choose an optimizer. A common choice is Stochastic Gradient Descent (SGD), but there are others that converge at different rates and achieve varying degrees of accuracy depending on the model.

![Optimizers](docs/img/cs231n_optimizers.gif)

_(Source: Stanford University's [CS231n: "Convolutional Neural Networks for Visual Recognition"](http://cs231n.github.io/neural-networks-3/))_

### Speech2Phonemes

#### Architecture

The first model, "Speech2Phonemes," attempts the task of framewise phoneme classification. The process involves associating a sequence of speech frames to phoneme labels matched to those frames. Ideally, this would be a first step to achieving a speech recognition model able to recognize an arbitrary number of words by piecing together phonemes.

The model used was a multilayer perception, an artificial neural network model often used in machine learning for classification tasks. The model had one hidden layer with 256 sigmoid activation units. A dropout layer was added, which switched off a percentage of the nodes in the network to prevent the domination of a single node and to decouple the influence between nodes, slightly improving accuracy of the model. The output layer had a dimension of 61, as there were 61 phoneme classes found in the dataset. Probabilities for each class was calculated through the final softmax layer.

```
____________________________________________________________________________________________________
Layer (type)                       Output Shape        Param #     Connected to                     
====================================================================================================
dense_1 (Dense)                    (None, 256)         10240       dense_input_1[0][0]              
____________________________________________________________________________________________________
activation_1 (Activation)          (None, 256)         0           dense_1[0][0]                    
____________________________________________________________________________________________________
dropout_1 (Dropout)                (None, 256)         0           activation_1[0][0]               
____________________________________________________________________________________________________
dense_2 (Dense)                    (None, 61)          15677       dropout_1[0][0]                  
____________________________________________________________________________________________________
activation_2 (Activation)          (None, 61)          0           dense_2[0][0]                    
====================================================================================================
____________________________________________________________________________________________________

```

#### Features

In order to extract relevant information from the input speech files, we decided to use MFCC feature vectors. MFCCs are commonly used for speech recognition tasks because of its relative accuracy in revealing patterns in human speech. The Mel scale was created to more closely mimic what humans hear, as we are generally better at distinguishing between changes at lower frequencies than at higher frequencies. Expressing the speech signal as a series of vectors is also more ideal for processing the data.

Thirteen MFCC coefficients were chosen for the task, as seemed to be widely used in many implementations of speech recognition models. In addition, delta and delta-delta features (derivatives) corresponding to the thirteen MFCCs were appended to the vector to obtain additional information about the signal. These were calculated using the `librosa` library, through the `librosa.feature.mfcc` and `librosa.feature.delta` functions. The windows sizes and step sizes recommended were 25ms and 10ms, however, due to the smaller length of some uttered phones, the window size was chosen to be a bit smaller at 20ms as a compromise.

With the sampling rate at 16 kHz for all the audio files, samples were less than a millisecond. From analysis of the data, according to the transcription, some phones were smaller than the window size. In addition, the start and end times of the utterances could mean that multiple phones could be represented in a window frame. To resolve this alignment issue, we simply took the phone that occupied the majority of the frame as the label for that frame.

##### Normalization

Before feeding the 39-MFCC-long vectors into the model, each coefficient was normalized around the mean of all coefficients with the same index in the training set. Normalization provides noticeable benefits as it adjusts the ranges of the values to be "symmetric" around the origin. Other normalization techniques, such as that which normalizes around the mean and divides by the standard deviation (thereby placing most data points within three standard deviations of the mean as exemplified by the normal distribution below), were tested.

![Normal distribution](docs/img/normal_distribution.png)

Nevertheless, the most effective technique for our application was to merely normalize around the mean (no division by the standard deviation).

#### Training

As would be expected in training, a loss is calculated for each interval of training (an epoch) which should be minimized in order to obtain more accurate results. As can be seen through the loss function, this gradually decreases, and generally more epochs would result in a better trained model. Obvious constraints for training would be the time it takes to train the model, which makes MLPs slightly easy to deal with (as opposed to RNNs, and other architectures). In addition, overfitting for the training data might occur with too many epochs.

![Plot of the loss function](docs/img/speech2phonemes_loss.png)

### Phonemes2Text

#### Architecture

The second model, "Phonemes2Text", accepts a series of phonemes and attempts to classify it as any of the words used by the dataset. Like its previous counterpart, it is a feed-forward neural network characterized by a series of dense, sigmoid activation and dropout layers. The output dimension parameter of the first layer, 1500, was determined empirically to give the best results. For 20 epochs, a change in this parameter from 256 to 1500 improved the accuracy by 16.8%.

```
____________________________________________________________________________________________________
Layer (type)                       Output Shape        Param #     Connected to                     
====================================================================================================
dense_1 (Dense)                    (None, 1500)        46500       dense_input_1[0][0]              
____________________________________________________________________________________________________
activation_1 (Activation)          (None, 1500)        0           dense_1[0][0]                    
____________________________________________________________________________________________________
dropout_1 (Dropout)                (None, 1500)        0           activation_1[0][0]               
____________________________________________________________________________________________________
dense_2 (Dense)                    (None, 6102)        9159102     dropout_1[0][0]                  
____________________________________________________________________________________________________
activation_2 (Activation)          (None, 6102)        0           dense_2[0][0]                    
____________________________________________________________________________________________________
dropout_2 (Dropout)                (None, 6102)        0           activation_2[0][0]               
____________________________________________________________________________________________________
dense_3 (Dense)                    (None, 6102)        37240506    dropout_2[0][0]                  
____________________________________________________________________________________________________
activation_3 (Activation)          (None, 6102)        0           dense_3[0][0]                    
====================================================================================================
Total params: 46446108
____________________________________________________________________________________________________
```

#### Features

The phonemes are provided as a list of class numbers ranging from 0-61 (the total number of phonemes), accompanied by a one-hot vector denoting the word in a vocabulary of 6102 –see the words on `volumes/config/timit_words`. For 39826 series of phonemes –each of which corresponded to one word, the shapes of the matrices used to train the model are as follows:

```
X_train.shape = (39826, 30)
y_train.shape = (39826, 6102)
```

We make the assumption that a word will be composed of at most 30 phonemes, and right-pad the words with fewer phonemes with zeros. This seems valid, given that the longest word in the dataset contained 17 phonemes.

#### Training

The neural network was trained on a CPU (a slow process) using 50 epochs. The loss function started off at 5.6256 and, notably, decreased to 0.6436, using the Adam optimizer and a learning rate of 0.001 –a value that was greater by a factor of 10 was attempted to speed up the learning process but, unfortunately, the model did not converge (the loss function sky-rocketed).

```
Epoch 1/50
39826/39826 [==============================] - 186s - loss: 5.6256     
Epoch 2/50
39826/39826 [==============================] - 193s - loss: 4.2672     
...  
Epoch 49/50
39826/39826 [==============================] - 262s - loss: 0.6437     
Epoch 50/50
39826/39826 [==============================] - 264s - loss: 0.6436
```

### End-to-End

The two models were trained independently using data from TIMIT (1.4M and 38K samples, respectively). In order to tie the output from the first (individual phonemes) to the second (groups of phonemes from which words may be classified), a regulation scheme displayed by `model/regulator.py` was developed to remove duplicate phonemes and reduce the impact of the noise. The former algorithm would successfully trim a series e.g. `['a', 'a', 'a', 'b', 'b']` to `['a', 'b']`, and the latter assumed that a correct phoneme would appear at least "a few times" during a 20-ms period wherein one is captured for every frame.

The accuracies of the two models, trained separately, were:
- "Speech2Phonemes": 47.4%
- "Phonemes2Text": 60.7%

This means that, in the first case, a 20-ms clip has a 47.4% chance of being classified as the correct phoneme (out of 61) and, in the second, a series of phonemes has a 60.7% chance of being classified as the correct word (out of 6102). This assumption, however, is not expected to hold for the end-to-end scheme, wherein the inputs to the second model contain non-negligible levels of noise.

#### Sources

- http://andrew.gibiansky.com/blog/machine-learning/recurrent-neural-networks/
- http://www.cs.toronto.edu/~fritz/absps/RNN13.pdf
- http://karpathy.github.io/2015/05/21/rnn-effectiveness/
- https://gist.github.com/karpathy/d4dee566867f8291f086

## Implementation

### Dataset

The TIMIT dataset is an often used corpus developed by MIT, Stanford and Texas Instruments for training and testing automatic speech recognition systems. There are 6300 sentences spoken by 630 speakers (438 male, 192 female) with more than 6000 different words used. Speakers were chosen from 8 dialect regions of the United States, encompassing various geographical sections of the states. The dataset also had a suggested training/testing split, used to partition the data.

The extensive labeling for the dataset made it a favorable one to use, as both phonetic and word labels for the speech files were provided with the data. Using those labels, we were able to perform framewise labeling and use this to train and test the data.

### Keras*

As stated, [Keras](https://github.com/fchollet/keras) is a deep learning library built on top of computational libraries that provides a high-level interface to add customizable layers based on a series of hyperparameters.

> Keras is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either TensorFlow or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

It provides a rich toolset to analyze the performance of an architecture.

## Command Interpreter

To attempt to match the similarity of strings, an edit distance metric is used. One such metric is the Levenshtein distance, which measures the distance between two words as the minimum number of single-character edits (insertions, deletions and substitutions) needed to go from one string to the other. This can be efficiently implemented through a dynamic programming algorithm.

### Redis

The data of the application is stored in [Redis](http://redis.io/), a popular in-memory database that saves objects to the disk. The main data structure that is used is a hash, which maps usernames and their respective commands to a series of additional information, namely their chosen messages and ports.

The following output is from Redis' CLI:

```bash
172-16-9-92:mamigot speech-analysis$ redis-cli
127.0.0.1:6379> keys derek:*
1) "derek:ping"
2) "derek:lights"
127.0.0.1:6379> hmget derek:ping message
1) "keep alive hello"
127.0.0.1:6379> hmget derek:ping port
1) "13002"
127.0.0.1:6379>
```

### Sockets*

_[gafhasfhasfdadsfad]_

### Installation

Use a package manager (`apt-get`, `brew`, etc.) to install Redis and the system-level dependencies of PyAudio.

Install the Python-specific libraries listed in `requirements.txt` through:

```
$ pip install -r requirements.txt
```

### CLI*

Python's [Click library](http://click.pocoo.org/5/) was used to implement a command-line interface to the entire application (see `driver.py`). The commands it provides are:

```
$ python driver.py --help
```

```
$ python driver.py setlistener --port 23111
```

```
$ python driver.py register

$ python driver.py parse
```

```
$ python driver.py model speech2phonemes train --data 10000 --summarize True

$ python driver.py model speech2phonemes test

$ python driver.py model phonemes2text train --data 1000

$ python driver.py model phonemes2text test
```
