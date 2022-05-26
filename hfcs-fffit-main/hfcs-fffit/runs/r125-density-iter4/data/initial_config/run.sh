#!/bin/bash

# Ryan DeFever
# Maginn Research Group
# University of Notre Dame
# 2020 Feb 24

conda activate nsf-hfcs
module load gcc/9.1.0
source /afs/crc.nd.edu/user/r/rdefever/software/gromacs-2020/gromacs-dp/bin/GMXRC

python create_system.py

gmx_d grompp -f em.mdp -c system.gro -p system.top -o system_em
gmx_d mdrun -v -deffnm system_em
