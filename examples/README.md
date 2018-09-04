# Efficient simulation of the training data

Both the discoal step and the feature vector building steps
can be time-consuming.  The most expedient way to address 
that is to break the jobs up as much as possible.

Here, we simulate training data using msprime for the coalescent
simulations and discoal for the sweep simulations.
We will simulate 50 replicates each of neutral, soft,
and hard sweeps.  We will do the sims in batches of 5
replicates per parameter combo.  Obviously, a real-world
job would do many more replicates in total!

We'll get the job done using GNU parallel.

This directory has the following scripts:

* `efficient_simulation.sh` breaks up the simulation and feature generation jobs and executes them via GNU parallel
* `collate.sh` uses simple bash commands to bring the output from the previous steps together so that the training can be done.
* `train.sh` runs the training step.  This script exists to show that the output from the previous two steps is usable.
