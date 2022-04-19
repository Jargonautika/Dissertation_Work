#!/bin/bash

# Iterate over the different groups of files
############################################
# Train
############################################
trainAudio=/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSSo-IS2021-data/train/progression/train/audio
for condition in decline no_decline; do
    for wav in $trainAudio/$condition/*.wav; do

        fName=$(basename $wav)
        noExt="${fName%.*}"
        out=$trainAudio/$condition/resample/$fName
        sox $wav -r 16000 -c 1 -b 16 -t wav $out

        trsfile=$trainAudio/$condition/transcripts/$noExt.txt
        outfile=$trainAudio/$condition/textgrids/$noExt.TextGrid 

        cd ../
        
        # We need to use Python 2.5 or 2.6 as per the PFA README
        /home/chasea2/.localpython/bin/python p2fa/align.py $out $trsfile $outfile
        rm -fr ./tmp/

        cd ./scripts/
        # exit 0

    done

done
exit 0

############################################
# Test
############################################
testAudio=/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSSo-IS2021-data/test/progression/test-dist/audio
for wav in $testAudio/*.wav; do

    fName=$(basename $wav)
    noExt="${fName%.*}"
    out=$testAudio/resample/$fName
    sox $wav -r 16000 -c 1 -b 16 -t wav $out

    trsfile=$testAudio/transcripts/$noExt.txt
    outfile=$testAudio/textgrids/$noExt.TextGrid 

    cd ../
    
    # We need to use Python 2.5 or 2.6 as per the PFA README
    /home/chasea2/.localpython/bin/python p2fa/align.py $out $trsfile $outfile
    rm -fr ./tmp/

    cd ./scripts/
    # exit 0

done
