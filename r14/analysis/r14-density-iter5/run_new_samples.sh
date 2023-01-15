#!/bin/bash
#$ -N id-new-samples
#$ -pe smp 2
#$ -r n
#$ -m ae
#$ -q long
#$ -M nwang2@nd.edu

conda activate hfcs-fffit
module load gcc/11.1.0
python id-new-samples.py
