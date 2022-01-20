#!/usr/bin/env bash

RUNNUM=1897 
stage=2
crossVal="False"

bash ./configs/ams $RUNNUM /tmp/withheldAMS $stage $crossVal
# bash ./configs/mfcc $RUNNUM /tmp/tmp.withheldMFCC $stage $crossVal
# bash ./configs/mfb $RUNNUM /tmp/tmp.withheldMFB $stage $crossVal # Mel filterbanks
# bash ./configs/rasta-plp $RUNNUM /tmp/tmp.withheldRASTA $stage $crossVal
# bash ./combine.sh $RUNNUM /tmp/tmp.withheldCOMBINE $stage $crossVal 1
# bash ./configs/compare $RUNNUM /tmp/tmp.withheldCOMPARE $stage $crossVal
# bash ./configs/gemaps $RUNNUM /tmp/tmp.withheldGEMAPS $stage $crossVal
# bash ./configs/auditory auditory-local $RUNNUM /tmp/tmp.withheldCRAFTED $stage $crossVal
