#!/usr/bin/env python3

from ObjectiveIntelligibilityMetrics import Intelligibility
from extractor_local import Extractor
from joblib import Parallel, delayed
import multiprocessing as mp
import pandas as pd
import numpy as np
import argparse
import pickle
import shutil
import glob
import os


def locateBigWav(wav):

    id = "{}.wav".format(wav.split('-')[0].split('/')[-1])
    startDir = "/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSS-IS2020-data/Full_wave_enhanced_audio_rms_normalized"
    for root, dirs, files in os.walk(startDir, topdown=False):
        if id in files:
            return os.path.join(root, files[files.index(id)])


def locateTextGrid(wav):

    id = "{}.TextGrid".format(wav.split('-')[0].split('/')[-1])
    startDir = "/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/TextGrids"
    for root, dirs, files in os.walk(startDir, topdown=False):
        if id in files:
            return os.path.join(root, files[files.index(id)])


def multiProcNoVal(i):

    wav, destFolder, noValues, names = i

    base = os.path.basename(os.path.splitext(wav)[0].strip())
    location = os.path.join(destFolder, base + '.csv')
    df = pd.read_csv(location, names = names)
    for noV in noValues:
        df.drop(noV, 1, inplace = True)
    df.to_csv(location, index = False, header = False) # Save over the extracted file. 


# For the case where the dev/withheld data needs to have the same columns as the training data
def matchMissing(wavs, destFolder, names):

    with open(os.path.join('/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Code/PYTHON/AUDITORY', 'noValuesTrain.pkl'), 'rb') as f:
        noValues = pickle.load(f)

    # Pull out the offending columns one by one; parallelize this because it's slow
    Parallel(n_jobs = int(mp.cpu_count() / 2))(delayed(multiProcNoVal)((wav, destFolder, noValues, names)) for wav in wavs[:])
    # Update names after we've ditched some columns
    for noV in noValues:
        names.remove(noV)

    # Sanity Check
    # Because what happens to the imputer when there are columns in the dev data which are completely empty
    # but which in the test data actually have SOME values? 
    segmentalFeatureCSVs = list()
    for wav in wavs[:]:
        base = os.path.basename(os.path.splitext(wav)[0].strip())
        location = os.path.join(destFolder, base + '.csv')
        segmentalFeatureCSVs.append(location)

    checkDF = pd.concat([pd.read_csv(x, names = names, index_col = False) for x in segmentalFeatureCSVs])
    for column in checkDF:
        if np.isnan(checkDF[column]).all():
            pass
            # print("There are columns in the dev frame which do not have data, but which do have data in the training frame.")
            # print("When you get to imputation, you'll need to set those dev columns to 0 and then impute so that the column is not deleted. It's a hack, but I think it should work and keep the analysis consistent. ")
            # raise AssertionError

    return names, noValues


def checkMissing(wavs, destFolder, names):

    # Update the names here, if there are any. This has downstream implications for the imputation of missing values
    # because our imputer will delete columns for which there are no values at all. 
    segmentalFeatureCSVs = list()
    for wav in wavs[:]:
        base = os.path.basename(os.path.splitext(wav)[0].strip())
        location = os.path.join(destFolder, base + '.csv')
        segmentalFeatureCSVs.append(location)

    checkDF = pd.concat([pd.read_csv(x, names = names, index_col = False) for x in segmentalFeatureCSVs])
    noValues = list()
    for column in checkDF:
        if np.isnan(checkDF[column]).all():
            noValues.append(column)

    # Pull out the offending columns one by one; parallelize this because it's slow
    Parallel(n_jobs = int(mp.cpu_count() / 2))(delayed(multiProcNoVal)((wav, destFolder, noValues, names)) for wav in wavs[:])
    # Update names after we've ditched some columns
    for noV in noValues:
        names.remove(noV)

    with open(os.path.join('/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Code/PYTHON/AUDITORY', 'noValuesTrain.pkl'), 'wb') as f:
        pickle.dump(noValues, f)
    
    with open(os.path.join('/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Code/PYTHON/AUDITORY', 'realNames.pkl'), 'wb') as f:
        pickle.dump(names, f)

    return names, noValues


def multiProcessingCall(i):

    wav, _, destFolder = i
    base = os.path.basename(os.path.splitext(wav)[0].strip())
    out = os.path.join(destFolder, base + '.csv')
    tg = locateTextGrid(wav)
    bw = locateBigWav(wav)
    ext = Extractor(wav, tg, bw)
    matrix = ext.returnMatrix()
    matrix_labels = ext.matrix_labels

    # Save the input feature vectors for acoustic modeling, machine learning, and neural networking
    df = pd.DataFrame(matrix)
    dfT = df.T # This does indeed need to be transposed, as with MFCCs
    dfT.to_csv(out, index = False, header = False)

    # Save the consonantal and vocalic observations for acoustic analysis
    # Plosives
    cDF = pd.DataFrame(ext.plosiveObservations, columns =     ["Speaker", "Waveform", "Condition", "Segment", "Voice Onset Time", "Center_of_Gravity", "Kurtosis", 
                                                               "Skewness", "Standard_Deviation",])
    cDF.to_csv("{}/../../acoustics/{}_plosives.csv".format(destFolder, os.path.basename(ext.name).split('.')[-2]), index = False)

    # Fricatives
    cDF = pd.DataFrame(ext.fricativeObservations, columns =   ["Speaker", "Waveform", "Condition", "Segment", 
                                                               "Center_of_Gravity", "Kurtosis", 
                                                               "Skewness", "Standard_Deviation", 
                                                               "Duration"])
    cDF.to_csv("{}/../../acoustics/{}_fricatives.csv".format(destFolder, os.path.basename(ext.name).split('.')[-2]), index = False)

    # Approximants
    cDF = pd.DataFrame(ext.approximantObservations, columns =  ["Speaker", "Waveform", "Condition", "Segment",
                                                                "F0_01", "F0_02", "F0_03", "F0_04", "F0_05",
                                                                "F1_01", "F2_01", "F3_01",
                                                                "F1_02", "F2_02", "F3_02",
                                                                "F1_03", "F2_03", "F3_03",
                                                                "F1_04", "F2_04", "F3_04",
                                                                "F1_05", "F2_05", "F3_05",
                                                                "Duration"])
    cDF.to_csv("{}/../../acoustics/{}_approximants.csv".format(destFolder, os.path.basename(ext.name).split('.')[-2]), index = False)

    vDF = pd.DataFrame(ext.vocalicObservations, columns = ["Speaker", "Waveform", "Condition", "Segment",
                                                           "F0_01", "F0_02", "F0_03", "F0_04", "F0_05",
                                                           "F1_01", "F2_01", "F3_01",
                                                           "F1_02", "F2_02", "F3_02",
                                                           "F1_03", "F2_03", "F3_03",
                                                           "F1_04", "F2_04", "F3_04",
                                                           "F1_05", "F2_05", "F3_05",
                                                           "Duration"])
    vDF.to_csv("{}/../../acoustics/{}_vowels.csv".format(destFolder, os.path.basename(ext.name).split('.')[-2]), index = False)
    return matrix_labels

def extract(exp_dir, which):

    wavFolder = os.path.join(exp_dir, 'data', which, 'wav')
    destFolder = os.path.join(exp_dir, 'data', which, 'csv')
    wavs = glob.glob(os.path.join(wavFolder, '*.wav'))

    # IMPORTANT - Because we're calling MATLAB functions, we can't multiprocess everything. So we do MATLAB outside the for loop and that's linear time. 
    # for wav in wavs[:]: # For debugging
    #     multiProcessingCall((wav, wavFolder, destFolder))
    matrix_labels_all = Parallel(n_jobs = int(mp.cpu_count() / 1))(delayed(multiProcessingCall)((wav, wavFolder, destFolder)) for wav in wavs[:])
    # return
    
    # We need to check if there are any features for which no values were extracted at all, across all of the training material
    names = matrix_labels_all[0]
    if os.path.basename(which) == 'train':
        names, noValues = checkMissing(wavs, destFolder, names)
    else: # Because dev always comes after, we can just use what we got from train and load it up
        names, noValues = matchMissing(wavs, destFolder, names)

    # Each of these dbSNRs is, for its noise type, roughly associated with perceptual intelligibility at 
    # roughly 70%, 50%, and 30% levels.
    levels  = [70, 50, 30]
    smnSNRs = [-7, -9, -11]
    ssnSNRs = [-2, -4, -6]
    for level, dBSNR_smn, dBSNR_ssn in zip(levels, smnSNRs, ssnSNRs):

        smnDWGPs, ssnDWGPs, smnSIIs, ssnSIIs, smnSTIs, ssnSTIs = Intelligibility(wavFolder, dBSNR_smn, dBSNR_ssn).returnValues()

        for wav in wavs[:]:

            base = os.path.basename(os.path.splitext(wav)[0].strip())
            out1 = os.path.join(destFolder, base + '.csv')
            out2 = os.path.join(destFolder, str(level), base + '.csv')

            df = pd.read_csv(out1, names = names)

            # Check to make sure we haven't already added in the OIMs
            assert df.shape == (1, len(names)), "We've already added in OIMs or else something has gone wrong."
            
            df['DWGP-SMN'] = smnDWGPs[wav]
            df['DWGP-SSN'] = ssnDWGPs[wav]
            df['SII-SMN']  = smnSIIs[wav]
            df['SII-SSN']  = ssnSIIs[wav]
            df['STI-SMN']  = smnSTIs[wav]
            df['STI-SSN']  = ssnSTIs[wav]

            df.to_csv(out2, index_label = False, index = False, header = False)
            # os.remove(out1) # This is a temporary file put there during OIM calculation but I can't remove it in this loop so just don't bother. 


def extractGlobal(exp_dir, which):

    wavFolder = os.path.join(exp_dir, 'data', which, 'wav')
    destFolder = os.path.join(exp_dir, 'data', which, 'csv')

    speakerList, wavs = list(), list()
    allWavs = glob.glob(os.path.join(wavFolder, '*.wav'))
    bigWavFolder = os.path.join(wavFolder, 'bigWaveforms')
    # Set up for DWGP calculation from big waveforms
    if os.path.isdir(bigWavFolder):
        shutil.rmtree(bigWavFolder)
    os.mkdir(bigWavFolder)

    for wav in allWavs:
        speaker = os.path.basename(wav).split('-')[0]
        if speaker not in speakerList:
            speakerList.append(speaker)
            wavs.append(wav)
            try:
                bigFile = "/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSS-IS2020-data/Full_wave_enhanced_audio_rms_normalized/train/{}.wav".format(speaker)
                shutil.copyfile(bigFile, os.path.join(bigWavFolder, os.path.basename(wav)))
            except:
                bigFile = "/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSS-IS2020-data/Full_wave_enhanced_audio_rms_normalized/dev/{}.wav".format(speaker)
                shutil.copyfile(bigFile, os.path.join(bigWavFolder, os.path.basename(wav)))

        else:
            os.remove(wav) # Get rid of everything except a single random utterance per speaker (as an index for them)
                           # This sets us up better to deal with the DWGP call later

    # IMPORTANT - Because we're calling MATLAB functions, we can't multiprocess everything. So we do MATLAB outside the for loop and that's linear time. 
    Parallel(n_jobs = int(mp.cpu_count() / 2))(delayed(multiProcessingCall)((wav, wavFolder, destFolder)) for wav in wavs)
    SMNs, SSNs = Intelligibility(bigWavFolder).returnDWGP()

    for wav in wavs:

        smn = SMNs[wav]
        ssn = SSNs[wav]
        base = os.path.basename(os.path.splitext(wav)[0].strip())
        out = os.path.join(destFolder, base + '.csv')

        df = pd.read_csv(out)

        # Check to make sure we haven't already added in DWGP
        assert df.shape == (1, 515), "We've already added in DWGP or else something has gone wrong."
            
        df['DWGP-SMN'] = smn
        df['DWGP-SSN'] = ssn
        df.to_csv(out, index_label = False, index = False, header = False)


def main():

    parser = argparse.ArgumentParser(description='Description of part of pipeline.')
    parser.add_argument('exp_dir', nargs='?', type=str, help='Temporary experiment directory.', default = '/tmp/tmp.OgrVUR6iE4')
    parser.add_argument('train_dev', nargs='?', type=str, help='Whether we are doing training or development extraction at the moment.', default = '/tmp/tmp.OgrVUR6iE4/data/train')
    parser.add_argument('scope', nargs = '?', type = str, help = "speaker level global or utterance level local", default = 'auditory-local')

    args = parser.parse_args()
    if args.scope == "auditory-global":
        extractGlobal(args.exp_dir, args.train_dev)
    else:
        extract(args.exp_dir, args.train_dev)


if __name__ == "__main__":

    main()
