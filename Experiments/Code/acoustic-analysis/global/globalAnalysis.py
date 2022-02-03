#!/usr/bin/env python3

from asyncore import file_dispatcher
from lib.DSP_Tools import normaliseRMS, energy
from scipy.signal import butter, sosfilt
from lib.WAVReader import WAVReader as WR
import removeInterviewer
import articulationRate
import concatenateWords
import pandas as pd
import numpy as np
import parselmouth
import shutil
import glob
import sys
import os


def getArticulationRate(tmpFile, which, partition, condition):

    rate = articulationRate.main(tmpFile, partition, condition)

    return rate


# https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y


def getIntensity(tmpFile, which, partition, condition):

    if "Full" in which:
        # Concatenate the word segments together
        wav = WR(tmpFile)
        sig = wav.getData()
        fs = wav.getSamplingRate()

        # Get rid of silence and fillers
        sig = concatenateWords.main(tmpFile, partition, condition, sig, fs)

    else:
        wav = WR(tmpFile)
        sig = wav.getData()
        fs = wav.getSamplingRate()

    # Normalize the signal
    sig, k = normaliseRMS(sig, tarRMS = 0.075)
            
    # Bandpass filter between 1kHz and 3kHz
    bpFilteredSig = butter_bandpass_filter(sig, 1000, 3000, fs)

    # Calculate the mean energy for the file
    intensity = energy(bpFilteredSig)
    
    return intensity


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

    # # Get rid of the interviewer in the long files
    # if "Full" in which:
    #     file = getRidOfInterviewer(file, partition, condition, destFolder)

    # # Get Median F0 per file
    # f0, iqr = getFundamentalFrequency(file)

    # # Get Intensity per file
    # intensity = getIntensity(file, which, partition, condition)

    # Get articulation rate
    articulationRate = getArticulationRate(file, which, partition, condition)

    # Get pausing


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
                    getInformation(file, which, partition, condition, "tmpGlobal")

        else:
            
            files = glob.glob(os.path.join(dataDir, partition, which, "*"))
            for file in files:
                basename = os.path.basename(file).split('.')[0]
                condition = 'cc' if df.loc[df['ID'] == basename].Label.tolist()[0] == 0 else 'cd'
                getInformation(file, which, partition, condition, "tmpGlobal")

    shutil.rmtree("tmpGlobal")


if __name__ == "__main__":

    main("Full_wave_enhanced_audio") # TODO multiprocess the calculation across all files since it takes a minute
