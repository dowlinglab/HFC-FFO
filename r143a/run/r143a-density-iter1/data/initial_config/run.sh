#!/bin/bash

conda activate FFO_FF_env
module load gcc/9.1.0
source /afs/crc.nd.edu/group/maginn/group_members/Ryan_DeFever/software/gromacs-2020/gromacs-dp/bin/GMXRC

python3 create_system.py

gmx_d grompp -f em.mdp -c system.gro -p system.top -o system_em
gmx_d mdrun -v -deffnm system_em
