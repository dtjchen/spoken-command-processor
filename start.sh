#!/bin/bash

PROJECT_ROOT=$PWD

######################################
# Add relevant modules to PYTHONPATH #
######################################
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

for dir in $PROJECT_ROOT/*; do
	export PYTHONPATH=$PYTHONPATH:$dir
done

##############################
# Set paths to training data #
##############################
export VOLUMES=$PWD/volumes
export VOLUMES_LABELS=$VOLUMES/labels.txt
export VOLUMES_META=$VOLUMES/meta.txt

####################################################################
# Read constants from $VOLUMES_META and set them as env. variables #
####################################################################
regex="([A-Z_]+)=([0-9]+)"

cat $VOLUMES_META | while read line; do
    if [[ $line =~ $regex ]]
    then
      key=${BASH_REMATCH[1]}
      value=${BASH_REMATCH[2]}

      export $key=$value # TODO: fix! does not work
    fi
done
