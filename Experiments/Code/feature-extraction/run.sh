#!/usr/bin/env bash

# RUNNUM=$RANDOM
RUNNUM=3104 # Manually subset the training data into 80/20 train/dev
stage=1
crossVal="False"

bash ./configs/ams $RUNNUM /tmp/tmp.2CKHTckEZ3 $stage $crossVal
bash ./configs/mfcc $RUNNUM /tmp/tmp.OnCYm1VXui $stage $crossVal
bash ./configs/rasta-plp $RUNNUM /tmp/tmp.1Uu3jV6Anl $stage $crossVal
bash ./combine.sh $RUNNUM /tmp/tmp.Wy3D2NOZ86 $stage $crossVal 1
bash ./configs/compare $RUNNUM /tmp/tmp.Psc7g4V77e $stage $crossVal
bash ./configs/gemaps $RUNNUM /tmp/tmp.RFVA79Kf0X $stage $crossVal
# bash ./configs/auditory auditory-global $RUNNUM /tmp/tmp.TcRZJvM59O $stage $crossVal
bash ./configs/auditory auditory-local $RUNNUM /tmp/tmp.OgrVUR6iE4 $stage $crossVal

# Multi-process each of these input feature sets
# for config in ams mfcc rasta-plp compare; do
#     (bash ./configs/$config $RUNNUM & )
# done
