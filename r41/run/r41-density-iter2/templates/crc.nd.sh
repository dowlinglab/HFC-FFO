{% extends "base_script.sh" %}
{% block header %}
#!/bin/bash
#$ -N {{ id }}
#$ -pe smp {{ np_global }}
#$ -r n
#$ -m ae
#$ -q long
#$ -M mcarlozo@nd.edu

module load gcc/9.1.0
source /afs/crc.nd.edu/group/maginn/group_members/Ryan_DeFever/software/gromacs-2020/gromacs-dp/bin/GMXRC
# export PATH=/afs/crc.nd.edu/user/m/mcarlozo/.conda/envs/hfc-ffo/bin:$PATH

{% block tasks %}
{% endblock %}
{% endblock %}

