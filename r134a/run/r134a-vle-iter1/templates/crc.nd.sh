{% extends "base_script.sh" %}
{% block header %}
#!/bin/bash
#$ -N {{ id }}
#$ -pe smp {{ np_global }}
#$ -r n
#$ -m ae
#$ -q long
#$ -M nwang2@nd.edu

conda activate hfcs-fffit

{% block tasks %}
{% endblock %}
{% endblock %}

