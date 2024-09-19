{% extends "base_script.sh" %}
{% block header %}
#!/bin/bash
#$ -N {{ id }}
#$ -pe smp {{ np_global }}
#$ -r n
#$ -m ae
#$ -q long
#$ -M mcarlozo@nd.edu

export PATH=/afs/crc.nd.edu/user/m/mcarlozo/.conda/envs/hfc-ffo/bin:$PATH

{% block tasks %}
{% endblock %}
{% endblock %}

