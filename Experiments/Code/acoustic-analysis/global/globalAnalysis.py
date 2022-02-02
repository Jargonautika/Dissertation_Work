#!/usr/bin/env python3

import removeInterviewer
import pandas as pd
import numpy as np
import parselmouth
import shutil
import glob
import os


def getRidOfInterviewer(file, partition, condition, tmpFolder = "tmpGlobal"):

    tgDir = "/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/TextGrids/"
    id = os.path.basename(file).split('.')[0]
    if partition == "train":
        tg = os.path.join(tgDir, partition, condition, '{}.TextGrid'.format(id))

    else: 
        tg = os.path.join(tgDir, partition, '{}.TextGrid'.format(id))

    tmpFile = removeInterviewer.main(file, tg, tmpFolder)
    return tmpFile


def hertzToSemitones(x, re=1):
    
    return 12 * np.log(x / re) / np.log(2)


def getFundamentalFrequency(file):

    # Read in the file
    sound = parselmouth.Sound(file)
    # Use Praat's spectral compression model to get the pitch
    pitch = sound.to_pitch_shs()
    # Get the strongest candidate's array (0th one in Parselmouth/Praat)
    f0ArrayHertz = [z[0] for z in pitch.to_array()[0]]
    # Convert to Semitones re 1 Hz
    # Forced to replace -inf values with 0.0 due to logarithmic conversion
    f0ArraySemitones = [hertzToSemitones(x) if x != 0.0 else 0.0 for x in f0ArrayHertz]
    # Get the median f0 value
    f0 = np.median(f0ArraySemitones)
    # Get the interquartile range of the fundamental frequency for the file
    q3, q1 = np.percentile(f0ArraySemitones, [75, 25])
    iqr = q3 - q1

    return f0, iqr


def getInformation(file, which, partition, condition, destFolder):

    # Get rid of the interviewer in the long files
    if "Full" in which:
        singleSpeakerFile = getRidOfInterviewer(file, partition, condition, destFolder)

    # Get Median F0 per file
    f0, iqr = getFundamentalFrequency(singleSpeakerFile)

    # Get Intensity per file


def main(which):

    try:
        os.mkdir("tmpGlobal")
    except:
        shutil.rmtree("tmpGlobal")
        os.mkdir("tmpGlobal")

    # File for knowing in the test partition which condition each participant falls into
    testMetaData = "/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSS-IS2020-data/meta_data_test.txt"
    df = pd.read_csv(testMetaData, sep = ";", skipinitialspace = True).rename(columns=lambda x: x.strip())
    df.ID = df.ID.str.replace(' ', '')

    # Iterate over the files, maintaining access to metadata
    dataDir = "/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSS-IS2020-data/"
    for partition in ["train", "test"]:

        if partition == "train":

            for condition in ["cc", "cd"]:

                files = glob.glob(os.path.join(dataDir, partition, which, condition, "*"))
                for file in files:
                    getInformation(file, which, partition, condition)

        else:
            
            files = glob.glob(os.path.join(dataDir, partition, which, "*"))
            for file in files:
                basename = os.path.basename(file).split('.')[0]
                condition = 'cc' if df.loc[df['ID'] == basename].Label.tolist()[0] == 0 else 'cd'
                getInformation(file, which, partition, condition)

    shutil.rmtree("tmpGlobal")


if __name__ == "__main__":

    main("Full_wave_enhanced_audio")
