#!/usr/bin/env python3

from asyncore import file_dispatcher
from lib.DSP_Tools import normaliseRMS, energy
from scipy.signal import butter, sosfilt
from lib.WAVReader import WAVReader as WR
from joblib import Parallel, delayed
import multiprocessing as mp
import removeInterviewer
import articulationRate
import concatenateWords
import pandas as pd
import numpy as np
import parselmouth
import pauseRate
import shutil
import glob
import sys
import os


def getPausingRate(tmpFile, partition, condition):

    rate = pauseRate.main(tmpFile, partition, condition)

    return rate


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
    id = os.path.basename(file).split('.')[0].split('-')[0]
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
    # Explicitly ignore the 0.0 values for the median calculation, otherwise you'll get some 0.0s for F0 and that's wrong
    f0ArraySemitones = [hertzToSemitones(x) for x in f0ArrayHertz if x != 0.0]
    # Get the median f0 value
    f0 = np.median(f0ArraySemitones)
    # Get the interquartile range of the fundamental frequency for the file
    q3, q1 = np.percentile(f0ArraySemitones, [75, 25])
    iqr = q3 - q1

    return f0, iqr


def getInformation(file, which, partition, condition, destFolder):

    id = os.path.basename(file).split('.')[0]

    if not isinstance(condition, str):
        basename = os.path.basename(file).split('.')[0]
        condition = 'cc' if condition.loc[condition['ID'] == basename].Label.tolist()[0] == 0 else 'cd'

    # Get rid of the interviewer in the long files
    if "Full" in which:
        file = getRidOfInterviewer(file, partition, condition, destFolder)

    # Get Median F0 per file
    f0, iqr = getFundamentalFrequency(file)

    # Get Intensity per file
    intensity = getIntensity(file, which, partition, condition)

    # Get articulation rate
    articulationRate = getArticulationRate(file, which, partition, condition)

    # Get pausing
    pausingRate = getPausingRate(file, partition, condition)

    return id, f0, iqr, intensity, articulationRate, pausingRate, condition


def main(which):

    try:
        os.mkdir("tmpGlobal")
    except:
        shutil.rmtree("tmpGlobal")
        os.mkdir("tmpGlobal")

    # Iterate over the files, maintaining access to metadata
    dataDir = "/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSS-IS2020-data/"
    bigList = list()
    for partition in ["train", "test"]:

        if partition == "train":

            for condition in ["cc", "cd"]:

                files = glob.glob(os.path.join(dataDir, partition, which, condition, "*"))
                # X = list()
                # for file in files:
                #     x = getInformation(file, which, partition, condition, "tmpGlobal")
                #     X.append(x)
                X = Parallel(n_jobs=mp.cpu_count())(delayed(getInformation)(file, which, partition, condition, "tmpGlobal") for file in files[:])

                # Get the metadata
                trainMetaData = "/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSS-IS2020-data/train/{}_meta_data.txt".format(condition)
                df = pd.read_csv(trainMetaData, sep = ";", skipinitialspace = True).rename(columns=lambda x: x.strip())
                df.ID = df.ID.str.replace(' ', '')
                for x in X:
                    id = x[0].split('-')[0]
                    row = df.loc[df['ID'] == id]
                    lilList = [id, row.age.values[0], row.gender.values[0]]
                    for i in x[1:]:
                        lilList.append(i)
                    bigList.append(lilList)

        else:

            # File for knowing in the test partition which condition each participant falls into
            testMetaData = "/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSS-IS2020-data/meta_data_test.txt"
            df = pd.read_csv(testMetaData, sep = ";", skipinitialspace = True).rename(columns=lambda x: x.strip())
            df.ID = df.ID.str.replace(' ', '')
            
            files = glob.glob(os.path.join(dataDir, partition, which, "*"))
            # X = list()
            # for file in files:
            #     x = getInformation(file, which, partition, condition, "tmpGlobal")
            #     X.append(x)
            X = Parallel(n_jobs=mp.cpu_count())(delayed(getInformation)(file, which, partition, df, "tmpGlobal") for file in files[:])
            
            for x in X:
                id = x[0].split('-')[0]
                row = df.loc[df['ID'] == id]
                lilList = [id, row.age.values[0], row.gender.values[0]]
                for i in x[1:]:
                    lilList.append(i)
                bigList.append(lilList)

    DF = pd.DataFrame(bigList, columns = ['ID', 'Age', 'Gender', 'F0', 'iqr', 'Intensity', 'ArticulationRate', 'PausingRate', 'Condition'])
    DF.to_csv('./global/GlobalMeasures_{}.csv'.format(which), index = False)

    shutil.rmtree("tmpGlobal")


if __name__ == "__main__":

    main("Full_wave_enhanced_audio")
