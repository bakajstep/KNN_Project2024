#!/bin/bash

# Script for testing on the Metacentrum.

# Source: https://github.com/roman-janik/diploma_thesis_program/blob/main/ner/start_training_ner.sh

# Arguments:
# 1. git branch name
# 2. model name (directory)
# 3. walltime in format HH:MM:SS

BRANCHNAME=$1
MODEL=$2
JTIMEOUT=$3
SHOUR=$(echo "$JTIMEOUT" | cut -d: -f1)
STIME=$((SHOUR - 1))

qsub -v branch="$BRANCHNAME",stime="$STIME",model="$MODEL" -l walltime="$JTIMEOUT" ./ner_classification.sh
