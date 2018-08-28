#!/bin/bash

DIPLOSHICDIR=$HOME/src/diploSHIC

if [ ! -d training_out ]
then
    mkdir training_out
else
    rm -f training_out/*
fi

python3 $DIPLOSHICDIR/diploSHIC.py makeTrainingSets neutral_training.txt softTraining/soft hardTraining/hard 5 0,1,2,3,4,6,7,8,9,10 training_out
    
python3 $DIPLOSHICDIR/diploSHIC.py train training_out/ training_out/ modelfile.txt

