# diploS/HIC
This repo contains the implementation for diploS/HIC as described in Kern and Schrider (2018), along 
with its associated support scripts. diploS/HIC uses a deep convolutional neural network to identify
hard and soft selective sweep in population genomic data. 

The workflow for analysis using diploS/HIC consists of four basic parts. 1) Generation of a training set for diploS/HIC 
using simulation. 2) diploS/HIC training and performance evaluation. 3) Calculation of dipoS/HIC feature vectors from genomic data.
4) prediction on empirical data using the trained network. The software provided here can handle the last three parts; population
genetic simulations must be performed using separate software such as discoal (https://github.com/kern-lab/discoal) 


