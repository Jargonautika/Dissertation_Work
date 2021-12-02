#!/usr/bin/env bash

# Let's work on cross-validation stuff first, since we know it works
RUNNUM=3104
crossVal="True"
# Stage 2 is for neural modeling
stage=2

# echo "Now working on AMS as an individual feature"
# bash ./configs/ams $RUNNUM /tmp/tmp.2CKHTckEZ3 $stage $crossVal

# echo "Now working on RASTA-PLP as an individual feature"
# bash ./configs/rasta-plp $RUNNUM /tmp/tmp.1Uu3jV6Anl $stage $crossVal

# echo "Now working on MFCC as an individual feature"
# bash ./configs/mfcc $RUNNUM /tmp/tmp.OnCYm1VXui $stage $crossVal

# echo "Now working on auditory features from QP2"
# bash ./configs/auditory $RUNNUM /tmp/tmp.TcRZJvM59O $stage $crossVal

# echo "Now working on the compare 2016 feature set"
# bash ./configs/compare $RUNNUM /tmp/tmp.Psc7g4V77e $stage $crossVal

echo "Now working on the combination of AMS, RASTA-PLP, and MFCC features"
bash ./configs/combine $RUNNUM /tmp/tmp.Wy3D2NOZ86 $stage $crossVal
