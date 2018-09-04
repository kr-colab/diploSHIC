#!/bin/bash

NTRAINING=neutral_training.txt

if [ -e $NTRAINING ]
then
    rm -f $NTRAINING
fi
I=0
for i in simtemp/msprime*.fvec
do
    if [ $I -eq 0 ]
    then
        cat $i > $NTRAINING
    else
        # Strip the header
        tail -n +2 $i >> $NTRAINING
    fi
    I=$(($I+1))
done

HDIR=hardTraining
SDIR=softTraining

if [ ! -d $HDIR ]
then
    mkdir $HDIR
else
    rm -f $HDIR/*
fi

if [ ! -d $SDIR ]
then
    mkdir $SDIR
else
    rm -f $SDIR/*
fi

WIN=0
for sweep_pos in $(seq 0 0.1 1.0)
do
    HFILE=$HDIR/hard_$WIN.txt
    SFILE=$SDIR/soft_$WIN.txt
    I=0
    for i in simtemp/hard_$WIN.*.fvec
    do
        if [ $I -eq 0 ]
        then
            cat $i > $HFILE
        else
            tail -n +2 $i >> $HFILE
        fi
        I=$(($I+1))
    done

    for i in simtemp/soft_$WIN.*.fvec
    do
        if [ $I -eq 0 ]
        then
            cat $i > $SFILE
        else
            tail -n +2 $i >> $SFILE
        fi
        I=$(($I+1))
    done
    WIN=$((WIN+1))
done

