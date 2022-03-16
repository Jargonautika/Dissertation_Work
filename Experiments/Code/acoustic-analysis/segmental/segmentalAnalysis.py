#!/usr/bin/env python3

from re import L
from joblib import Parallel, delayed
import phonemeCategoryMeasures
import multiprocessing as mp
import vowelSpaceMeasures
import pandas as pd
import numpy as np
import shutil
import glob
import sys
import os

sys.path.insert(1, './global')
from globalAnalysis import normIt


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


def getPhonemicCategoryInformation(file, sig, which, partition, condition, destFolder, gender = 0):

    id = os.path.basename(file).split('.')[0].split('-')[0]

    if not isinstance(condition, str):
        basename = os.path.basename(file).split('.')[0].split('-')[0]
        condition = 'cc' if condition.loc[condition['ID'] == basename].Label.tolist()[0] == 0 else 'cd'

    ##################################
    # Phoneme Category Distinctions
    ##################################
    plosiveVOTMeans, fricativeCOGMeans, vowelsERBMeans, vowelsDURMeans = phonemeCategoryMeasures.main(file, sig, gender)
    return id, plosiveVOTMeans, fricativeCOGMeans, vowelsERBMeans, vowelsDURMeans, condition

    # Model 2 Features
    P_VOT, B_VOT, T_VOT, D_VOT, K_VOT, G_VOT = plosiveVOTMeans
    F_COG, V_COG, S_COG, Z_COG, SH_COG, ZH_COG, TH_COG, DH_COG = fricativeCOGMeans
    IY_ERB, IH_ERB, UW_ERB, UH_ERB, AA_ERB, AE_ERB = vowelsERBMeans
    IY_DUR, IH_DUR, UW_DUR, UH_DUR, AA_DUR, AE_DUR = vowelsDURMeans

    # Model 3 Features
    P_B_BCD, T_D_BCD, K_G_BCD = pD[0] 
    P_B_WCD, T_D_WCD, K_G_WCD = pD[1]
    P_B_CO, T_D_CO, K_G_CO = pD[2]
    P_B_CD, T_D_CD, K_G_CD = pD[3]

    # Model 4 Features
    F_V_BCD, S_Z_BCD, SH_ZH_BCD, TH_DH_BCD = fD[0] 
    F_V_WCD, S_Z_WCD, SH_ZH_WCD, TH_DH_WCD = fD[1]
    F_V_CO, S_Z_CO, SH_ZH_CO, TH_DH_CO = fD[2]
    F_V_CD, S_Z_CD, SH_ZH_CD, TH_DH_CD = fD[3]

    # Model 5 Features
    IY_IH_BCD_ERB, UW_UH_BCD_ERB, AA_AE_BCD_ERB = vERBD[0] 
    IY_IH_WCD_ERB, UW_UH_WCD_ERB, AA_AE_WCD_ERB = vERBD[1]
    IY_IH_CO_ERB, UW_UH_CO_ERB, AA_AE_CO_ERB = vERBD[2]
    IY_IH_CD_ERB, UW_UH_CD_ERB, AA_AE_CD_ERB = vERBD[3]

    # Model 6 Features
    IY_IH_BCD_DUR, UW_UH_BCD_DUR, AA_AE_BCD_DUR = vDurD[0] 
    IY_IH_WCD_DUR, UW_UH_WCD_DUR, AA_AE_WCD_DUR = vDurD[1]
    IY_IH_CO_DUR, UW_UH_CO_DUR, AA_AE_CO_DUR = vDurD[2]
    IY_IH_CD_DUR, UW_UH_CD_DUR, AA_AE_CD_DUR = vDurD[3]

    return  id, \
            P_VOT, B_VOT, T_VOT, D_VOT, K_VOT, G_VOT, \
            F_COG, V_COG, S_COG, Z_COG, SH_COG, ZH_COG, TH_COG, DH_COG, \
            IY_ERB, IH_ERB, UW_ERB, UH_ERB, AA_ERB, AE_ERB, \
            IY_DUR, IH_DUR, UW_DUR, UH_DUR, AA_DUR, AE_DUR, \
            P_B_BCD, T_D_BCD, K_G_BCD, \
            P_B_WCD, T_D_WCD, K_G_WCD, \
            P_B_CO, T_D_CO, K_G_CO, \
            P_B_CD, T_D_CD, K_G_CD, \
            F_V_BCD, S_Z_BCD, SH_ZH_BCD, TH_DH_BCD, \
            F_V_WCD, S_Z_WCD, SH_ZH_WCD, TH_DH_WCD, \
            F_V_CO, S_Z_CO, SH_ZH_CO, TH_DH_CO, \
            F_V_CD, S_Z_CD, SH_ZH_CD, TH_DH_CD, \
            IY_IH_BCD_ERB, UW_UH_BCD_ERB, AA_AE_BCD_ERB, \
            IY_IH_WCD_ERB, UW_UH_WCD_ERB, AA_AE_WCD_ERB, \
            IY_IH_CO_ERB, UW_UH_CO_ERB, AA_AE_CO_ERB, \
            IY_IH_CD_ERB, UW_UH_CD_ERB, AA_AE_CD_ERB, \
            IY_IH_BCD_DUR, UW_UH_BCD_DUR, AA_AE_BCD_DUR, \
            IY_IH_WCD_DUR, UW_UH_WCD_DUR, AA_AE_WCD_DUR, \
            IY_IH_CO_DUR, UW_UH_CO_DUR, AA_AE_CO_DUR, \
            IY_IH_CD_DUR, UW_UH_CD_DUR, AA_AE_CD_DUR, \
            condition


def getVowelSpaceInformation(file, sig, which, partition, condition, destFolder, gender = 0):

    id = os.path.basename(file).split('.')[0].split('-')[0]

    if not isinstance(condition, str):
        basename = os.path.basename(file).split('.')[0].split('-')[0]
        condition = 'cc' if condition.loc[condition['ID'] == basename].Label.tolist()[0] == 0 else 'cd'

    ##################################
    # Vowel Space Measures
    ##################################
    vowelRate, vA2D, vA3D, F1, F2, F3 = vowelSpaceMeasures.main(file, sig) # float, list, list, list, list, list

    return id, vowelRate, vA2D, vA3D, F1, F2, F3, condition


def saveOut(categoricalList, numericalList, saveName, which):

    for task, taskList, target in [('categorical', categoricalList, 'Condition'), ('numerical', numericalList, 'MMSE')]:
        
        # We have just one model using these segmental measures related to vowel space
        if saveName == "Vowel_Space":

            # Segmental Model #1 (Vowel Space Degradations)
            specifics = ["Vowel_Rate", "Vowel_Area_2D", "Vowel_Area_3D", "F1_Range", "F2_Range", "F3_Range"]

            columns = ['ID', 'Age', 'Gender'] + specifics + [target]
            # IDK why it's happening so here's a weird hack to fix it
            if task == 'categorical':
                taskList = [[x for i, x in enumerate(tList) if i != (len(tList) - 1)] for tList in taskList]
            else:
                taskList = [[x for i, x in enumerate(tList) if i != (len(tList) - 2)] for tList in taskList]
            DF = pd.DataFrame(taskList, columns = columns)
            DF = removeNaN(DF)
            DF.to_csv('./segmental/SegmentalMeasures_{}-{}-{}.csv'.format(which, task, saveName), index = False)
        
        # WE have five models using these other segmental features related to phoneme category distinctions
        else:

            # Segmental Model #2 (Phonemic Contrast Degradations) # 26 values
            specifics2 = (["P_VOT", "B_VOT", "T_VOT", "D_VOT", "K_VOT", "G_VOT", 
                        "F_COG", "V_COG", "S_COG", "Z_COG", "SH_COG", "ZH_COG", "TH_COG", "DH_COG", 
                        "IY_ERB", "IH_ERB", "UW_ERB", "UH_ERB", "AA_ERB", "AE_ERB",
                        "IY_DUR", "IH_DUR", "UW_DUR", "UH_DUR", "AA_DUR", "AE_DUR"],
                        3,
                        29)

            # Segmental Model #3 (Plosive Category Distinction Degradations) # 12 values
            specifics3 = (["P_B_BCD", "T_D_BCD", "K_G_BCD",
                        "P_B_WCD", "T_D_WCD", "K_G_WCD",
                        "P_B_CO", "T_D_CO", "K_G_CO", 
                        "P_B_CD", "T_D_CD", "K_G_CD"],
                        29,
                        41)

            # Segmental Model #4 (Fricative Category Distinction Degradations) # 16 values
            specifics4 = (["F_V_BCD", "S_Z_BCD", "SH_ZH_BCD", "TH_DH_BCD", 
                            "F_V_WCD", "S_Z_WCD", "SH_ZH_WCD", "TH_DH_WCD",
                            "F_V_CO", "S_Z_CO", "SH_ZH_CO", "TH_DH_CO",
                            "F_V_CD", "S_Z_CD", "SH_ZH_CD", "TH_DH_CD"],
                            41,
                            57)

            # Segmental Model #5 (Vowel ERB Category Distinction Degradations) # 12 values
            specifics5 = (["IY_IH_BCD_ERB", "UW_UH_BCD_ERB", "AA_AE_BCD_ERB",
                            "IY_IH_WCD_ERB", "UW_UH_WCD_ERB", "AA_AE_WCD_ERB",
                            "IY_IH_CO_ERB", "UW_UH_CO_ERB", "AA_AE_CO_ERB",
                            "IY_IH_CD_ERB", "UW_UH_CD_ERB", "AA_AE_CD_ERB"],
                            57,
                            69)

            # Segmental Model #6 (Vowel DUR Category Distinction Degradations) # 12 values
            specifics6 = (["IY_IH_BCD_DUR", "UW_UH_BCD_DUR", "AA_AE_BCD_DUR",
                            "IY_IH_WCD_DUR", "UW_UH_WCD_DUR", "AA_AE_WCD_DUR",
                            "IY_IH_CO_DUR", "UW_UH_CO_DUR", "AA_AE_CO_DUR",
                            "IY_IH_CD_DUR", "UW_UH_CD_DUR", "AA_AE_CD_DUR"],
                            69,
                            81)

            for (specifics, start, end), modelName in zip([specifics2, specifics3, specifics4, specifics5, specifics6], ['phonetic_contrasts', 'plosive_categories', 'fricative_categories', 'vowel_erb_categories', 'vowel_dur_categories']):
                columns = ['ID', 'Age', 'Gender'] + specifics + [target]
                DF = pd.DataFrame([taskList[i][:3] + taskList[i][start:end] + [taskList[i][-1]] for i in range(len(taskList))], columns = columns)
                DF = removeNaN(DF)            
                DF.to_csv('./segmental/SegmentalMeasures_{}-{}-{}-{}.csv'.format(which, task, saveName, modelName), index = False)

    shutil.rmtree("tmpGlobal")


def categoryDiscriminability(first: list, second: list):

    return (np.nanmean(first) - np.nanmean(second)) * np.sqrt(2) / np.sqrt((np.std(first) ** 2) + (np.std(second) ** 2))


def categoryOverlap(first: list, second: list):

    if len(first) > 0 and len(second) > 0:
        return np.max(second) - np.min(first)
    else:
        return 0 # Sometimes we don't have a particular phoneme produced for a speaker


def withinCategoryDispersion(first: list, second: list):

    return np.nanmean([np.std(first), np.std(second)])


def betweenCategoryDistance(first: list, second: list):

    return np.nanmean(first) - np.nanmean(second)


def distinctions(plosives, fricatives, vowelsERB, vowelsDur):

    plosivePairs = [('P', 'B'), ('T', 'D'), ('K', 'G')]                     # Voiceless and then voiced
    fricativePairs = [('F', 'V'), ('S', 'Z'), ('SH', 'ZH'), ('TH', 'DH')]   # Voiceless and then voiced
    vowelPairs = [('IY', 'IH'), ('UW', 'UH'), ('AA', 'AE')]                 # Tense and then lax

    plosiveBCD, plosiveWCD, plosiveCO, plosiveCD = list(), list(), list(), list()
    for voiceless, voiced in plosivePairs:
        first, second = plosives[voiceless], plosives[voiced]
        # Calculate Between-Category Distance
        plosiveBCD.append(betweenCategoryDistance(first, second))
        # Calculate Within-Category Dispersion
        plosiveWCD.append(withinCategoryDispersion(first, second))
        # Calculate Category Overlap
        plosiveCO.append(categoryOverlap(first, second))
        # Calculate Category Discriminability
        plosiveCD.append(categoryDiscriminability(first, second))

    fricativeBCD, fricativeWCD, fricativeCO, fricativeCD = list(), list(), list(), list()
    for voiceless, voiced in fricativePairs:
        first, second = fricatives[voiceless], fricatives[voiced]
        fricativeBCD.append(betweenCategoryDistance(first, second))
        fricativeWCD.append(withinCategoryDispersion(first, second))
        fricativeCO.append(categoryOverlap(first, second))
        fricativeCD.append(categoryDiscriminability(first, second))

    vowelERBBCD, vowelERBWCD, vowelERBCO, vowelERBCD = list(), list(), list(), list()
    for tense, lax in vowelPairs:
        first, second = vowelsERB[tense], vowelsERB[lax]
        vowelERBBCD.append(betweenCategoryDistance(first, second))
        vowelERBWCD.append(withinCategoryDispersion(first, second))
        vowelERBCO.append(categoryOverlap(first, second))
        vowelERBCD.append(categoryDiscriminability(first, second))

    vowelDurBCD, vowelDurWCD, vowelDurCO, vowelDurCD = list(), list(), list(), list()
    for tense, lax in vowelPairs:
        first, second = vowelsDur[tense], vowelsDur[lax]
        vowelDurBCD.append(betweenCategoryDistance(first, second))
        vowelDurWCD.append(withinCategoryDispersion(first, second))
        vowelDurCO.append(categoryOverlap(first, second))
        vowelDurCD.append(categoryDiscriminability(first, second))

    return [plosiveBCD, plosiveWCD, plosiveCO, plosiveCD], \
           [fricativeBCD, fricativeWCD, fricativeCO, fricativeCD], \
           [vowelERBBCD, vowelERBWCD, vowelERBCO, vowelERBCD], \
           [vowelDurBCD, vowelDurWCD, vowelDurCO, vowelDurCD]


def removeOutliers(values, sexDict, saveName): # 

    categoricalList, numericalList = list(), list()
    # We have just one model using these segmental measures related to vowel space
    if saveName == "Vowel_Space":

        # vowelArea2D, vowelArea3D, f1Range, f2Range, f3Range
        allValuesList = [[], [], [], [], []]
        for X, _, _ in values:
            # Flatten out all values from all speakers
            for x in X:
                for i in range(5): # str, float, list, list, list, list, list, str
                    for j in x[i + 2]: # 2 because the first two are ID and vowelRate
                        allValuesList[i].append(j)

        stdsList = list()
        for valuesList in allValuesList:
            myMean = np.nanmean(valuesList)
            std = np.nanstd(valuesList)

            stdsList.append((myMean - 2 * std, myMean + 2 * std))

        # Second pass to now use the means and stds we just calculated
        for X, df, condition in values:
            for x in X:

                id = x[0].split('-')[0]
                row = df.loc[df['ID'] == id]
                if condition != 'test':
                    gender = row.gender.values[0]
                else: 
                    gender = sexDict[row.gender.values[0]]
                lilList = [id, row.age.values[0], gender, x[1]] # ID, Age, Gender, vowelRate

                for i in range(5):
                    lilList.append(np.nanmean([j for j in x[i+2] if stdsList[i][0] < j < stdsList[i][1]]))

                categoricalList.append(lilList)
                categoricalList[-1].append(x[-1]) # Condition 'cc' or 'cd'

                # Filter out that one NaN guy for MMSE
                if np.isnan(row.mmse.values[0]):
                    continue
                else:
                    numericalList.append(lilList)
                    numericalList[-1].append(row.mmse.values[0])

    else:

        allValuesList = list()
        for i in range(26):
            allValuesList.append([])
        for X, _, _ in values:
            for _, plosiveVOTDict, fricativeCOGDict, vowelsERBDict, vowelsDURDict, _ in X:
                i = 0
                for plosive in ['P', 'B', 'T', 'D', 'K', 'G']:
                    for j in plosiveVOTDict[plosive]:
                        allValuesList[i].append(j)
                    i += 1
                for fricative in ['F', 'V', 'S', 'Z', 'SH', 'ZH', 'TH', 'DH']:
                    for j in fricativeCOGDict[fricative]:
                        allValuesList[i].append(j)
                    i += 1
                for vowel in ['IY', 'IH', 'UW', 'UH', 'AA', 'AE']:
                    for j in vowelsERBDict[vowel]:
                        allValuesList[i].append(j)
                    i += 1
                for vowel in ['IY', 'IH', 'UW', 'UH', 'AA', 'AE']:
                    for j in vowelsDURDict[vowel]:
                        allValuesList[i].append(j)
                    i += 1

                assert i == 26, "You're doing values wrong here."

        stdsList = list()
        for valuesList in allValuesList:
            myMean = np.nanmean(valuesList)
            std = np.nanstd(valuesList)

            stdsList.append((myMean - 2 * std, myMean + 2 * std))

        for X, df, condition in values:
            for x, plosiveVOTDict, fricativeCOGDict, vowelsERBDict, vowelsDURDict, y in X:
                
                id = x.split('-')[0]
                row = df.loc[df['ID'] == id]
                if condition != 'test':
                    gender = row.gender.values[0]
                else: 
                    gender = sexDict[row.gender.values[0]]
                lilList = [id, row.age.values[0], gender] # ID, Age, Gender

                i = 0
                pD = dict()
                for plosive in ['P', 'B', 'T', 'D', 'K', 'G']:
                    valList = [j for j in plosiveVOTDict[plosive] if stdsList[i][0] < j < stdsList[i][1]]
                    pD[plosive] = valList
                    lilList.append(np.nanmean(valList))
                    i += 1

                fD = dict()
                for fricative in ['F', 'V', 'S', 'Z', 'SH', 'ZH', 'TH', 'DH']:
                    valList = [j for j in fricativeCOGDict[fricative] if stdsList[i][0] < j < stdsList[i][1]]
                    fD[fricative] = valList
                    lilList.append(np.nanmean(valList))
                    i += 1

                vERBD = dict()
                for vowel in ['IY', 'IH', 'UW', 'UH', 'AA', 'AE']:
                    valList = [j for j in vowelsERBDict[vowel] if stdsList[i][0] < j < stdsList[i][1]]
                    vERBD[vowel] = valList
                    lilList.append(np.nanmean(valList))
                    i += 1

                vDurD = dict()
                for vowel in ['IY', 'IH', 'UW', 'UH', 'AA', 'AE']:
                    valList = [j for j in vowelsDURDict[vowel] if stdsList[i][0] < j < stdsList[i][1]]
                    vDurD[vowel] = valList
                    lilList.append(np.nanmean(valList))
                    i += 1

                pD, fD, vERBD, vDurD = distinctions(pD, fD, vERBD, vDurD) # 4 lists of 4 lists each (BCD, CO, etc); the sublists contain 3 or 4 float values related to, for example with plosives, [('P', 'B'), ('T', 'D'), ('K', 'G')] contrasts

                for j in range(4): # ['BCD', 'WCD', 'CO', 'CD']
                    for k, _ in enumerate([('P', 'B'), ('T', 'D'), ('K', 'G')]):
                        lilList.append(pD[j][k])
                        i += 1

                for j in range(4): # ['BCD', 'WCD', 'CO', 'CD']
                    for k, _ in enumerate([('F', 'V'), ('S', 'Z'), ('SH', 'ZH'), ('TH', 'DH')]):
                        lilList.append(fD[j][k])
                        i += 1

                for j in range(4): # ['BCD', 'WCD', 'CO', 'CD']
                    for k, _ in enumerate([('IY', 'IH'), ('UW', 'UH'), ('AA', 'AE')] ):
                        lilList.append(vERBD[j][k])
                        i += 1

                for j in range(4): # ['BCD', 'WCD', 'CO', 'CD']
                    for k, _ in enumerate([('IY', 'IH'), ('UW', 'UH'), ('AA', 'AE')] ):
                        lilList.append(vDurD[j][k])
                        i += 1

                categoricalList.append(lilList)
                categoricalList[-1].append(y)

                # Filter out that one NaN guy for MMSE
                if np.isnan(row.mmse.values[0]):
                    continue
                else:
                    numericalList.append(lilList)
                    numericalList[-1].append(row.mmse.values[0])

    shutil.rmtree("filesToNormalize")
    return categoricalList, numericalList
    

def getValues(which, myFunction, saveName, sexDict, reverseSexDict):

    # Iterate over the files, maintaining access to metadata
    dataDir = "/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSS-IS2020-data/"
    valuesList = list()
    for partition in ["train", "test"]:

        if partition == "train":

            for condition in ["cc", "cd"]:

                # Get the metadata
                trainMetaData = "/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSS-IS2020-data/train/{}_meta_data.txt".format(condition)
                df = pd.read_csv(trainMetaData, sep = ";", skipinitialspace = True).rename(columns=lambda x: x.strip())
                df.ID = df.ID.str.replace(' ', '')

                signals, files = normIt(glob.glob(os.path.join(dataDir, partition, which, condition, "*")))
                # X = list()
                # for i in range(len(signals))[:]:
                #     x = myFunction(files[i], signals[i], which, partition, condition, "tmpGlobal", reverseSexDict[df[df.ID == os.path.basename(files[i]).split('.')[0]].gender.values[0]])
                #     X.append(x)
                X = Parallel(n_jobs=mp.cpu_count())(delayed(myFunction)(files[i], signals[i], which, partition, condition, "tmpGlobal", reverseSexDict[df[df.ID == os.path.basename(files[i]).split('.')[0]].gender.values[0]]) for i in range(len(signals))[:])
                valuesList.append((X, df, condition))

        else:

            # File for knowing in the test partition which condition each participant falls into
            testMetaData = "/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSS-IS2020-data/meta_data_test.txt"
            df = pd.read_csv(testMetaData, sep = ";", skipinitialspace = True).rename(columns=lambda x: x.strip())
            df.ID = df.ID.str.replace(' ', '')
            
            signals, files = normIt(glob.glob(os.path.join(dataDir, partition, which, "*")))
            # X = list()
            # for sig, file in zip(signals, files):
            #     x = myFunction(file, sig, which, partition, condition, "tmpGlobal")
            #     X.append(x)
            X = Parallel(n_jobs=mp.cpu_count())(delayed(myFunction)(files[i], signals[i], which, partition, df, "tmpGlobal", df[df.ID == os.path.basename(files[i]).split('.')[0]].gender.values[0]) for i in range(len(files))[:])
            valuesList.append((X, df, 'test')) 

    return valuesList   


def setUp(function):

    if function:
        myFunction = getVowelSpaceInformation # str, float, list, list, list, list, list, str
        saveName = "Vowel_Space"
    else:
        myFunction = getPhonemicCategoryInformation # dict, dict, dict, dict, list, list, list, list
        saveName = "Phoneme_Category"

    sexDict = {0: 'male ', 1: 'female '}
    reverseSexDict = {v: k for k, v in sexDict.items()}

    try:
        os.mkdir("tmpGlobal")
    except:
        shutil.rmtree("tmpGlobal")
        os.mkdir("tmpGlobal")

    return myFunction, saveName, sexDict, reverseSexDict


def main(which, function = True):

    myFunction, saveName, sexDict, reverseSexDict = setUp(function)

    values = getValues(which, myFunction, saveName, sexDict, reverseSexDict)

    catList, numList = removeOutliers(values, sexDict, saveName)

    saveOut(catList, numList, saveName, which)


if __name__ == "__main__":

    main(which = "Full_wave_enhanced_audio")
    # main("Full_wave_enhanced_audio", False)
