#!/bin/bash

DISCOALDIR=$HOME/src/discoal
DIPLOSHICDIR=$HOME/src/diploSHIC
NREPS_PER_BATCH=5
SIMCOMMMANDSFILE=sim_commands.txt
FVECCOMMANDSFILE=fvec_commands.txt
SIMTEMPDIR=simtemp

if [ -e $SIMCOMMMANDSFILE ]
then
    rm -f $SIMCOMMMANDSFILE
fi

if [ -e $FVECCOMMANDSFILE ]
then
    rm -f $FVECCOMMANDSFILE
fi

if [ ! -d $SIMTEMPDIR ]
then
    mkdir $SIMTEMPDIR
else
    # Clean out contents of our temp dir
    rm -f $SIMTEMPDIR/*
fi

# Step 1: store msprime commands in a file and
# the commands to build their feature vectors
for i in $(seq 1 1 10)
do
    echo "mspms 100 $NREPS_PER_BATCH -t 1000 -r 1000 10000 --precision 10 --random-seeds $RANDOM $RANDOM $RANDOM > $SIMTEMPDIR/msprime_out.$i.txt" >> $SIMCOMMMANDSFILE
    echo "python3 $DIPLOSHICDIR/diploSHIC.py fvecSim haploid $SIMTEMPDIR/msprime_out.$i.txt $SIMTEMPDIR/msprime_out.$i.fvec" >> $FVECCOMMANDSFILE
done

# Step 2: commands for hard and soft sweeps over a range of params 
# and the commands to build the feature vectors
WIN=0
for sweep_pos in $(seq 0 0.1 1.0)
do
    for i in $(seq 1 1 10)
    do
        HARDFILE=$SIMTEMPDIR/hard_$WIN.batch$i.discoal
        SOFTFILE=$SIMTEMPDIR/soft_$WIN.batch$i.discoal
        echo "$DISCOALDIR/discoal 100 $NREPS_PER_BATCH 10000 -ws 0 -Pa 1000 2500 -x $sweep_pos -t 1000 -r 1000 > $HARDFILE" >> $SIMCOMMMANDSFILE
        echo "$DISCOALDIR/discoal 100 $NREPS_PER_BATCH 10000 -ws 0 -Pa 1000 2500 -x $sweep_pos -Pf 0.1 0.5 -t 1000 -r 1000 > $SOFTFILE" >> $SIMCOMMMANDSFILE
        b=`basename $HARDFILE .txt`
        HARDTRAIN=$SIMTEMPDIR/$b.fvec
        b=`basename $SOFTFILE .txt`
        SOFTTRAIN=$SIMTEMPDIR/$b.fvec
        echo "python3 $DIPLOSHICDIR/diploSHIC.py fvecSim haploid $HARDFILE $HARDTRAIN" >> $FVECCOMMANDSFILE
        echo "python3 $DIPLOSHICDIR/diploSHIC.py fvecSim haploid $SOFTFILE $SOFTTRAIN" >> $FVECCOMMANDSFILE
    done
    WIN=$(($WIN+1))
done

# Step 3: let GNU parallel rip.
# Note, on an HPC system, you would
# probably execute this step as 
# parallel --jobs $CORES, where $CORES
# are the number of CPU/threads that the queuing
# system assigned to your job. 
# This command used here simply uses all available resources:
parallel < $SIMCOMMMANDSFILE
parallel < $FVECCOMMANDSFILE

# Step 4: collate all the results into a SINGLE
# feature vector file for  neutral, hard, soft
