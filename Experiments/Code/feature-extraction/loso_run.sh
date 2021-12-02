#!/usr/bin/env bash

RUNNUM=21310 # Manually subset the training data into 80/20 train/dev
stage=1
crossVal="True"

bash ./configs/ams $RUNNUM /tmp/tmp.losoAMS $stage $crossVal
bash ./configs/mfcc $RUNNUM /tmp/tmp.losoMFCC $stage $crossVal
bash ./configs/rasta-plp $RUNNUM /tmp/tmp.losoRASTA $stage $crossVal
bash ./combine.sh $RUNNUM /tmp/tmp.losoCOMBINE $stage $crossVal 1
bash ./configs/compare $RUNNUM /tmp/tmp.losoCOMPARE $stage $crossVal
bash ./configs/gemaps $RUNNUM /tmp/tmp.losoGEMAPS $stage $crossVal
# bash ./configs/auditory auditory-global $RUNNUM /tmp/tmp.losoGLOBAL $stage $crossVal
bash ./configs/auditory auditory-local $RUNNUM /tmp/tmp.losoLOCAL $stage $crossVal
