#!/usr/bin/env bash

RUNNUM=1897 
stage=1
crossVal="False"

# bash ./configs/ams $RUNNUM /tmp/withheldAMS $stage $crossVal
bash ./configs/mfcc $RUNNUM /tmp/tmp.withheldMFCC $stage $crossVal
# bash ./configs/rasta-plp $RUNNUM /tmp/tmp.withheldRASTA $stage $crossVal
# bash ./combine.sh $RUNNUM /tmp/tmp.withheldCOMBINE $stage $crossVal 1
# bash ./configs/compare $RUNNUM /tmp/tmp.withheldCOMPARE $stage $crossVal
# bash ./configs/gemaps $RUNNUM /tmp/tmp.withheldGEMAPS $stage $crossVal
# bash ./configs/auditory auditory-local $RUNNUM /tmp/tmp.withheldCRAFTED $stage $crossVal
