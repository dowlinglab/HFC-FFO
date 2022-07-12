{% extends "base_script.sh" %}
{% block header %}
#!/bin/bash
#$ -N {{ id }}
#$ -pe smp {{ np_global }}
#$ -r n
#$ -m ae
#$ -q long@@maginn_q16copt
#$ -M rdefever@nd.edu

module load gcc/9.1.0
conda activate nsf-hfcs
source /afs/crc.nd.edu/group/maginn/group_members/Ryan_DeFever/software/gromacs-2020/gromacs-dp-q16copt/bin/GMXRC

{% block tasks %}
{% endblock %}
{% endblock %}

