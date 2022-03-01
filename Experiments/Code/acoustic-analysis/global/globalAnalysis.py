#!/usr/bin/env python3

from lib.DSP_Tools import normaliseRMS, energy
from lib.WAVReader import WAVReader as WR
from lib.WAVWriter import WAVWriter as WW
from scipy.signal import firwin, lfilter
from joblib import Parallel, delayed
from scipy.fft import fft

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


def removeNaN(DF):

    # Check to make sure ther aren't any totally empty columns
    nothingList = [DF[mys].isnull().all() for mys in DF]
    # If there are any columns with no observations
    if any(nothingList):
        # Iterate over the columns
        dropList = list()
        for i, j in enumerate(nothingList):
            # Find the completely NaN columns for category
            if j:
                print("No values found for {}".format(DF.columns[i]))
                dropList.append(DF.columns[i])
        DF.drop(dropList, inplace = True, axis = 1)

    return DF


def saveWav(i):

    utt, tarRMS = i

    wav = WR(utt)
    sig = wav.getData()
    fs = wav.getSamplingRate()
    bits = wav.getBitsPerSample()
    # dur = wav.getDuration()

    kx, _ = normaliseRMS(sig, tarRMS)

    WW(utt, kx, fs, bits).write()    

    # NOTE For the full-wave enhanced files this is too aggressive right now; it's removing ~100 of the full files, negating a lot of the possible analysis.
    # if not (kx >= -1).all() or not (kx <= 1).all():
    #     os.remove(utt) # I have to remove 66 files here because they are just clipping and noise; I have gone through a handful of them and can confirm that they are just bad
    #     # print(utt, '\t', rms(sig), '\t', dur)
    # else:
    #     WW(utt, kx, fs, bits).write()


def normalizeRMS(files, tarRMS = 0.075):

    Parallel(n_jobs=int(mp.cpu_count()))(delayed(saveWav)((utt, tarRMS)) for utt in files)


def normIt(files):

    # Make some temporary copies of these files and then normalize them
    try:
        os.mkdir("filesToNormalize")
    except:
        shutil.rmtree("filesToNormalize")
        os.mkdir("filesToNormalize")

    for file in files:
        shutil.copy(file, "filesToNormalize/{}".format(os.path.basename(file)))

    # Follow the same protocol as with the machine learning stuff
    files = glob.glob(os.path.join("filesToNormalize", "*"))
    normalizeRMS(files)
    files = glob.glob(os.path.join("filesToNormalize", "*")) # Duplicated here because we throw out 66 files as noise # NOTE see comments above in normalizeRMS

    return files


def getPausingRate(tmpFile, partition, condition):

    rate = pauseRate.main(tmpFile, partition, condition)

    return rate


def getArticulationRate(tmpFile, which, partition, condition):

    rate = articulationRate.main(tmpFile, partition, condition)

    return rate


def filterbank(fs, low, high, order=1000):

    # Get the coefficients from the finite impulse response window
    bs = firwin(order+1, [low, high], pass_zero = 'bandpass', fs = fs)

    return bs


# Bandpass filter between 1kHz and 3kHz # TODO check if it should be 4kHz
def filterSignal(sig, fs, numSamples):

    # Create the filterbank
    lowpass = 1000 # Hz
    highpass = 3000 # Hz
    bs = filterbank(fs, lowpass, highpass)

    # Apply the filter
    bpFilteredSig = lfilter(bs, 1, sig, axis = 0)

    # # If you want to see the signal, use this
    # import matplotlib.pyplot as plt
    # numFFT = int(2**(np.ceil(np.log2(numSamples))))
    # ffbin = fft(bpFilteredSig, n = numFFT, axis = 0)

    # frs = np.linspace(0, (fs/2 - fs/numFFT), int(numFFT / 2))

    # plt.figure()
    # plt.plot(frs, np.abs(ffbin[:int(numFFT / 2)]))
    # plt.xlim([0, 8000])
    # plt.savefig('check.png')

    return bpFilteredSig


def getIntensity(tmpFile, which, partition, condition):

    if "Full" in which:
        # Concatenate the word segments together
        wav = WR(tmpFile)
        sig = wav.getData()
        fs = wav.getSamplingRate()
        numSamples = wav.getSampleNO()

        # Get rid of silence and fillers
        sig = concatenateWords.main(tmpFile, partition, condition, sig, fs)

    else:
        wav = WR(tmpFile)
        sig = wav.getData()
        fs = wav.getSamplingRate()
        numSamples = wav.getSampleNO()

    # Normalize the signal
    sig, k = normaliseRMS(sig, tarRMS = 0.075)

    frequencyResponse = filterSignal(sig, fs, numSamples)

    # Calculate the mean energy for the file
    intensity = energy(frequencyResponse)
    
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

    id = os.path.basename(file).split('.')[0].split('-')[0]

    if not isinstance(condition, str):
        basename = os.path.basename(file).split('.')[0].split('-')[0]
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


def main(which, task = 'numerical'):

    sexDict = {0: 'male ', 1: 'female '}

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

                files = normIt(glob.glob(os.path.join(dataDir, partition, which, condition, "*")))
                # X = list()
                # for file in files[:2]:
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

                    if task != 'categorical': # Find MMSE
                        for i in x[1:-1]:
                            lilList.append(i)
                        # Filter out that one NaN guy for MMSE
                        if np.isnan(row.mmse.values[0]):
                            continue
                        else:
                            mmse = row.mmse.values[0]
                        lilList.append(mmse)
                        bigList.append(lilList)
                    else:
                        for i in x[1:]:
                            lilList.append(i)
                        bigList.append(lilList)

                shutil.rmtree("filesToNormalize")

        else:

            # File for knowing in the test partition which condition each participant falls into
            testMetaData = "/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSS-IS2020-data/meta_data_test.txt"
            df = pd.read_csv(testMetaData, sep = ";", skipinitialspace = True).rename(columns=lambda x: x.strip())
            df.ID = df.ID.str.replace(' ', '')
            
            files = normIt(glob.glob(os.path.join(dataDir, partition, which, "*")))
            # X = list()
            # for file in files[:2]:
            #     x = getInformation(file, which, partition, condition, "tmpGlobal")
            #     X.append(x)
            X = Parallel(n_jobs=mp.cpu_count())(delayed(getInformation)(file, which, partition, df, "tmpGlobal") for file in files[:])
            
            for x in X:
                id = x[0].split('-')[0]
                row = df.loc[df['ID'] == id]
                lilList = [id, row.age.values[0], sexDict[row.gender.values[0]]]
                if task != 'categorical': # Find MMSE
                    for i in x[1:-1]:
                        lilList.append(i)
                    # Filter out that one NaN guy for MMSE
                    if np.isnan(row.mmse.values[0]):
                        continue
                    else:
                        mmse = row.mmse.values[0]
                    lilList.append(mmse)
                    bigList.append(lilList)
                else:
                    for i in x[1:]:
                        lilList.append(i)
                    bigList.append(lilList)

            shutil.rmtree("filesToNormalize")

    if task == 'categorical':
        DF = pd.DataFrame(bigList, columns = ['ID', 'Age', 'Gender', 'F0', 'iqr', 'Intensity', 'ArticulationRate', 'PausingRate', 'Condition'])
    else:
        DF = pd.DataFrame(bigList, columns = ['ID', 'Age', 'Gender', 'F0', 'iqr', 'Intensity', 'ArticulationRate', 'PausingRate', 'MMSE'])
    DF = removeNaN(DF)
    DF.to_csv('./global/GlobalMeasures_{}-{}.csv'.format(which, task), index = False)

    shutil.rmtree("tmpGlobal")


if __name__ == "__main__":

    main("Full_wave_enhanced_audio")
