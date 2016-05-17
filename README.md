# Spoken Command Processor (miguel)

_[description of the project: its objectives and high-level functionality of this prototype]_

## Transcription Model Architecture (miguel)

_[description and architecture of the model]_

_[backpropagation]_

### Speech2Phonemes (derek)

#### Architecture

_[arbitrary parameter choices, etc.]_

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

_[why the delta MFCC features (as opposed to not delta) + the size of the resulting vectors characterizing 20ms intervals, etc.]_

In order to extract relevant information from the input speech files, we decided to use MFCC feature vectors. MFCCs are commonly used for speech recognition tasks because of its relative accuracy in revealing patterns in human speech. The Mel scale was created to more closely mimic what humans hear, as we are generally better at distinguishing between changes at lower frequencies than at higher frequencies. Expressing the speech signal as a series of vectors is also more ideal for processing the data.

Thirteen MFCC coefficients were chosen for the task, as seemed to be widely used in many implementations of speech recognition models. In addition, delta and delta-delta features (derivatives) corresponding to the thirteen MFCCs were appended to the vector to obtain additional information about the signal. These were calculated using the `librosa` library, through the `librosa.feature.mfcc` and `librosa.feature.delta` functions. The windows sizes and step sizes recommended were 25ms and 10ms, however, due to the smaller length of some uttered phones, the window size was chosen to be a bit smaller at 20ms as a compromise.

With the sampling rate at 16 kHz for all the audio files, samples were less than a millisecond. From analysis of the data, according to the transcription, some phones were smaller than the window size. In addition, the start and end times of the utterances could mean that multiple phones could be represented in a window frame. To resolve this alignment issue, we simply took the phone that occupied the majority of the frame as the label for that frame.

#### Training

_[plot of the loss function, discussion of the number of epochs]_

As would be expected in training, a loss is calculated for each interval of training (an epoch) which should be minimized in order to obtain more accurate results. As can be seen through the loss function, this gradually decreases, and generally more epochs would result in a better trained model. Obvious constraints for training would be the time it takes to train the model, which makes MLPs slightly easy to deal with (as opposed to RNNs, and other architectures). In addition, overfitting for the training data might occur with too many epochs.

### Phonemes2Text (miguel)

#### Architecture

_[arbitrary parameter choices, etc.]_

#### Features

_[also describe the bridge between Speech2Phonemes and Phonemes2Text]_

#### Training

_[plot of the loss function, discussion of the number of epochs]_

### End-to-End

_[limitations of tying the whole thing together + improvements]_

#### Sources

_[list of papers and blog posts we relied on for the implementation of the model]_

## Implementation

### Dataset (derek)

_[TIMIT and what it offers in terms of words, phonemes, wavfiles tagged frame-by-frame, sampling rate]_

The TIMIT dataset is an often used corpus developed by MIT, Stanford and Texas Instruments for training and testing automatic speech recognition systems. There are 6300 sentences spoken by 630 speakers (438 male, 192 female) with more than 6000 different words used. Speakers were chosen from 8 dialect regions of the United States, encompassing various geographical sections of the states. The dataset also had a suggested training/testing split, used to partition the data.

The extensive labeling for the dataset made it a favorable one to use, as both phonetic and word labels for the speech files were provided with the data. Using those labels, we were able to perform framewise labeling and use this to train and test the data.

### Keras

_[accessible deep learning library in terms of flexibility and the statistics it provides to gauge training, use of numpy]_

## Command Interpreter

_[description of the bells-n-whistles]_

_[edit_distance algorithm]_
To attempt to match the similarity of strings, an edit distance metric is used. One such metric is the Levenshtein distance, which measures the distance between two words as the minimum number of single-character edits (insertions, deletions and substitutions) needed to go from one string to the other. This can be efficiently implemented through a dynamic programming algorithm. 

### Sockets

### Redis (miguel)

_[database used to store and retrieve the user commands]_

### Click (miguel)

_[command-line interface]_
