#!/usr/bin/env bash

source /home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Envs/STATS/bin/activate

# featureDir=/tmp/tmp.IHHR7Bl2yh # Acoustic features extracted from the auditory feature pipeline using the "local" configuration (just using the training data in a train-dev split)
feautreDir=/tmp/tmp.OgrVUR6iE4   # Acoustic features extracted from the auditory feature pipeline using the "local" configuration (just using the training data and the withheld test data
rm -fr $featureDir/reports/plots/*
python3 main.py $featureDir # Analyze the vowels and the consonants
# python3 intelligibility.py $featureDir

deactivate
