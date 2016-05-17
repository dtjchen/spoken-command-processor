# Spoken Command Processor (miguel)

_[description of the project: its objectives and high-level functionality of this prototype]_

## Transcription Model Architecture (miguel)

_[description and architecture of the model]_

_[backpropagation]_

### Speech2Phonemes (derek)

#### Architecture

_[arbitrary parameter choices, etc.]_

#### Features

_[why the delta MFCC features (as opposed to not delta) + the size of the resulting vectors characterizing 20ms intervals, etc.]_

#### Training

_[plot of the loss function, discussion of the number of epochs]_

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

### Keras

_[accessible deep learning library in terms of flexibility and the statistics it provides to gauge training, use of numpy]_

## Command Interpreter

_[description of the bells-n-whistles]_

_[edit_distance algorithm]_

### Redis (miguel)

_[database used to store and retrieve the user commands]_

### Click (miguel)

_[command-line interface]_
