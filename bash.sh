#!/bin/bash

#$ -cwd
#$ -l h_vmem=8G
#$ -pe parallel-onenode 1




# name of job
# man 1 qsub
#$ -N Test1


# export environment variables to job
#$ -V


python Classifier.py
