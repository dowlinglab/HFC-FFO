{% extends "base_script.sh" %}
{% block header %}
#!/bin/bash
#$ -N {{ id }}
#$ -pe smp {{ np_global }}
#$ -r n
#$ -m ae
#$ -q long@@maginn_q16copt
#$ -M rdefever@nd.edu

conda activate hfcs

{% block tasks %}
{% endblock %}
{% endblock %}

