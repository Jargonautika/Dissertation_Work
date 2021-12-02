#!/usr/bin/env python3

import matplotlib.pyplot as plt
import soundfile as sf
import pandas as pd
import numpy as np
import glob
import sys
import os

# The purpose of this script is primarily to get descriptive duration statistics from the ADReSS corpus
# We will be pulling:
#   (a) waveform duration by
#       (i) binary diagnosis,
#       (ii) MMSE score,
#       (iii) train/test distinction, and
#       (iv) individual


def plotIt(conditionOneDict, conditionTwoDict, title, params = False):

    fig, axs = plt.subplots(2, 1, constrained_layout=True)

    axs[0].bar(*zip(*conditionOneDict.items()))
    axs[0].set_title(title.split('-')[0])
    axs[0].set_xlabel('speaker')
    axs[0].set_ylabel('seconds')
    if params:
        axs[0].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    fig.suptitle(title, fontsize=16)

    axs[1].bar(*zip(*conditionTwoDict.items()))
    axs[1].set_title(title.split('-')[1])
    axs[1].set_xlabel('speaker')
    axs[1].set_ylabel('seconds')
    if params:
        axs[1].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    plt.savefig("../outputs/{}.pdf".format(title))


def combine(trainIndividuals, trainMMSE, trainControl, trainDiagnosed, testIndividuals, testMMSE, testControl, testDiagnosed):

    outList = list()
    outList.append("Number of individuals for training:\t{}".format(len(trainIndividuals)))
    outList.append("Number of individuals for testing:\t{}".format(len(testIndividuals)))
    outList.append("Number of MMSE scores for training (30 possible):\t{}".format(len(trainMMSE)))
    outList.append("Number of MMSE scores for testing (30 possible):\t{}".format(len(testMMSE)))
    outList.append("Number of individual waveforms in the Control condition for training:\t{}".format(len(trainControl)))
    outList.append("Number of individual waveforms in the Control condition for testing:\t{}".format(len(testControl)))
    outList.append("Number of individual waveforms in the Diagnosed condition for training:\t{}".format(len(trainDiagnosed)))
    outList.append("Number of individual waveforms in the Diagnosed condition for testing:\t{}".format(len(testDiagnosed)))
    outList.append('')
    
    controlTotal = (sum(trainControl) + sum(testControl)) / 60
    diagnosedTotal = (sum(trainDiagnosed) + sum(testDiagnosed)) / 60
    averageControl = np.mean(trainControl + testControl)
    averageDiagnosed = np.mean(trainDiagnosed + testDiagnosed)
    outList.append("Normalized speech by non-AD-diagnosed individuals, in minutes:\t\t {}".format(controlTotal))
    outList.append("Average waveform duration non-AD-diagnosed individuals, in seconds:\t {}".format(averageControl))
    outList.append("Normalized speech by AD-diagnosed individuals, in minutes:\t\t {}".format(diagnosedTotal))
    outList.append("Average waveform duration AD-diagnosed individuals, in seconds:\t\t {}".format(averageDiagnosed))
    outList.append('')

    trainTotal = (sum(trainControl) + sum(trainDiagnosed)) / 60
    testTotal = (sum(testControl) + sum(testDiagnosed)) / 60
    averageTrain = np.mean(trainControl + trainDiagnosed)
    averageTest = np.mean(testControl + testDiagnosed)
    outList.append("Normalized speech for training, in minutes:\t {}".format(trainTotal))
    outList.append("Average waveform duration training, in seconds:\t {}".format(averageTrain))
    outList.append("Normalized speech for testing, in minutes:\t {}".format(testTotal))
    outList.append("Average waveform duration testing, in seconds:\t {}".format(averageTest))
    outList.append('')

    with open('../outputs/counts.txt', 'w') as f:
        for item in outList:
            f.write("%s\n" % item)
    
    # Individuals in the train condition by individuals in the test condition
    plotIt({key:np.sum(value) for (key, value) in trainIndividuals.items()}, {key:np.sum(value) for (key, value) in testIndividuals.items()}, "TrainIND-TestIND", True)

    # Duration by MMSE for the train and test conditions
    plotIt({str(key).strip():np.sum(value) for (key, value) in trainMMSE.items()}, {str(key).strip():np.sum(value) for (key, value) in testMMSE.items()}, "TrainMMSE-TestMMSE")


def combTestFiles(condition, ID):

    individualDurList = list()
    fileList = glob.glob("/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSS-IS2020-data/test/Normalised_audio-chunks/*.wav")
    for f in fileList:
        fID = os.path.basename(f).split('-')[0]
        if ID == fID:
            mySF = sf.SoundFile(f)
            samples = len(mySF)
            fs = mySF.samplerate
            dur = samples / fs
            individualDurList.append(dur)

    return individualDurList


def getTestInformation(condition):

    df = pd.read_csv('/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSS-IS2020-data/meta_data_test.txt', sep = ';').rename(columns=lambda x: x.strip())
    
    INDIVIDUALSDICT = dict()
    MMSEDICT = dict()
    allList = list()
    for i, row in df.iterrows():
        ID, AGE, GENDER, LABEL, MMSE = row
        if (condition == 'cc' and int(LABEL) == 0) or (condition == 'cd' and int(LABEL) == 1):
            durList = combTestFiles(condition, ID.strip())
            assert ID.strip() not in INDIVIDUALSDICT, "You have a duplicate person somehow?"
            INDIVIDUALSDICT[ID.strip()] = durList
            if MMSE not in MMSEDICT:
                MMSEDICT[MMSE] = list()
            for dur in durList:
                MMSEDICT[MMSE].append(dur)
                allList.append(dur)

    return INDIVIDUALSDICT, MMSEDICT, allList


def test():

    allPeople = dict()
    allMMSE = dict()
    ccDurs, cdDurs = list(), list()
    for condition in ['cc', 'cd']:
        individuals, mmse, allCondition = getTestInformation(condition)
        for i in individuals:
            allPeople[i] = individuals[i]
        for m in mmse:
            if m not in allMMSE:
                allMMSE[m] = list()
            for i in mmse[m]:
                allMMSE[m].append(i)
        for c in allCondition:
            if condition == 'cc':
                ccDurs.append(c)
            else:
                cdDurs.append(c)
    return allPeople, allMMSE, ccDurs, cdDurs


def combTrainFiles(condition, ID):

    individualDurList = list()
    fileList = glob.glob("/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSS-IS2020-data/train/Normalised_audio-chunks/{}/*.wav".format(condition))
    for f in fileList:
        fID = os.path.basename(f).split('-')[0]
        if ID == fID:
            mySF = sf.SoundFile(f)
            samples = len(mySF)
            fs = mySF.samplerate
            dur = samples / fs
            individualDurList.append(dur)

    return individualDurList


def getTrainInformation(condition):

    df = pd.read_csv('/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSS-IS2020-data/train/{}_meta_data.txt'.format(condition), sep = ';').rename(columns=lambda x: x.strip())
    
    INDIVIDUALSDICT = dict()
    MMSEDICT = dict()
    allList = list()
    for i, row in df.iterrows():
        ID, AGE, GENDER, MMSE = row
        durList = combTrainFiles(condition, ID.strip())
        assert ID.strip() not in INDIVIDUALSDICT, "You have a duplicate person somehow?"
        INDIVIDUALSDICT[ID.strip()] = durList
        if MMSE not in MMSEDICT:
            MMSEDICT[MMSE] = list()
        else:
            print(MMSE)
        for dur in durList:
            MMSEDICT[MMSE].append(dur)
            allList.append(dur)

    return INDIVIDUALSDICT, MMSEDICT, allList


def train():

    allPeople = dict()
    allMMSE = dict()
    ccDurs, cdDurs = list(), list()
    for condition in ['cc', 'cd']:

        individuals, mmse, allCondition = getTrainInformation(condition)
        for i in individuals:
            allPeople[i] = individuals[i]
        for m in mmse:
            if m not in allMMSE:
                allMMSE[m] = list()
            for i in mmse[m]:
                allMMSE[m].append(i)
        for c in allCondition:
            if condition == 'cc':
                ccDurs.append(c)
            else:
                cdDurs.append(c)
    return allPeople, allMMSE, ccDurs, cdDurs


def main():

    a, b, c, d = train()
    e, f, g, h = test()
    combine(a, b, c, d, e, f, g, h)


if __name__ == "__main__":
    
    main()
