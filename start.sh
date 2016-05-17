#!/bin/bash

# The root of the project is the root of this script
# http://stackoverflow.com/a/246128/2708484
export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#################################
# Config. environment variables #
#################################

# Path to the trained model's parameters
export MODEL_PARAMETERS=$PROJECT_ROOT/volumes/model

# Path to the list of phones in the data
export PHONE_LIST_PATH=$PROJECT_ROOT/volumes/config/timit_phones.txt

# Path to the list of words in the data
export WORD_LIST_PATH=$PROJECT_ROOT/volumes/config/timit_words.txt

# Paths to training and testing portions of the dataset
if [ -z "$TIMIT_TRAINING_PATH" ] || [ -z "$TIMIT_TESTING_PATH" ]; then
  echo "Set env. vars: TIMIT_TRAINING_PATH and TIMIT_TESTING_PATH."
  return;
fi

export TMP_RECORDING=$PROJECT_ROOT/tmp_recording.wav

######################################
# Add relevant modules to PYTHONPATH #
######################################
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

for dir in $PROJECT_ROOT/*; do
  export PYTHONPATH=$PYTHONPATH:$dir
done
