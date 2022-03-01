#!/usr/bin/env python3

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


def getPhonemicCategoryInformation(file, which, partition, condition, destFolder):

    id = os.path.basename(file).split('.')[0].split('-')[0]

    if not isinstance(condition, str):
        basename = os.path.basename(file).split('.')[0].split('-')[0]
        condition = 'cc' if condition.loc[condition['ID'] == basename].Label.tolist()[0] == 0 else 'cd'

    ##################################
    # Phoneme Category Distinctions
    ##################################
    plosiveVOTMeans, fricativeCOGMeans, vowelsERBMeans, vowelsDURMeans, pD, fD, vERBD, vDurD = phonemeCategoryMeasures.main(file)

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


def getVowelSpaceInformation(file, which, partition, condition, destFolder):

    id = os.path.basename(file).split('.')[0].split('-')[0]

    if not isinstance(condition, str):
        basename = os.path.basename(file).split('.')[0].split('-')[0]
        condition = 'cc' if condition.loc[condition['ID'] == basename].Label.tolist()[0] == 0 else 'cd'

    ##################################
    # Vowel Space Measures
    ##################################
    vowelRate, vA2D, vA3D, F1, F2, F3 = vowelSpaceMeasures.main(file)

    return id, vowelRate, vA2D, vA3D, F1, F2, F3, condition


def main(which, task = 'numerical', function = True):

    if function:
        myFunction = getVowelSpaceInformation
        saveName = "Vowel_Space"
    else:
        myFunction = getPhonemicCategoryInformation
        saveName = "Phoneme_Category"

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
                X = Parallel(n_jobs=mp.cpu_count())(delayed(myFunction)(file, which, partition, condition, "tmpGlobal") for file in files[:])

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
            X = Parallel(n_jobs=mp.cpu_count())(delayed(myFunction)(file, which, partition, df, "tmpGlobal") for file in files[:])
            
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

    # We have just one model using these segmental measures related to vowel space
    if saveName == "Vowel_Space":
        if task == 'categorical':
            taskList = ['Condition']
        else:
            taskList = ['MMSE']

        # Segmental Model #1 (Vowel Space Degradations)
        specifics = ["Vowel_Rate", "Vowel_Area_2D", "Vowel_Area_3D", "F1_Range", "F2_Range", "F3_Range"]

        columns = ['ID', 'Age', 'Gender'] + specifics + taskList
        DF = pd.DataFrame(bigList, columns = columns)
        DF = removeNaN(DF)
        DF.to_csv('./segmental/SegmentalMeasures_{}-{}-{}.csv'.format(which, task, saveName), index = False)
    
    # WE have five models using these other segmental features related to phoneme category distinctions
    else:
        if task == 'categorical':
            taskList = ['Condition']
        else:
            taskList = ['MMSE']

        # Segmental Model #2 (Phonetic Contrast Degradations)
        specifics2 = (["P_VOT", "B_VOT", "T_VOT", "D_VOT", "K_VOT", "G_VOT", 
                      "F_COG", "V_COG", "S_COG", "Z_COG", "SH_COG", "ZH_COG", "TH_COG", "DH_COG", 
                      "IY_ERB", "IH_ERB", "UW_ERB", "UH_ERB", "AA_ERB", "AE_ERB",
                      "IY_DUR", "IH_DUR", "UW_DUR", "UH_DUR", "AA_DUR", "AE_DUR"],
                      3,
                      29)

        # Segmental Model #3 (Plosive Category Distinction Degradations)
        specifics3 = (["P_B_BCD", "T_D_BCD", "K_G_BCD",
                       "P_B_WCD", "T_D_WCD", "K_G_WCD",
                       "P_B_CO", "T_D_CO", "K_G_CO", 
                       "P_B_CD", "T_D_CD", "K_G_CD"],
                       29,
                       41)

        # Segmental Model #4 (Fricative Category Distinction Degradations)
        specifics4 = (["F_V_BCD", "S_Z_BCD", "SH_ZH_BCD", "TH_DH_BCD", 
                        "F_V_WCD", "S_Z_WCD", "SH_ZH_WCD", "TH_DH_WCD",
                        "F_V_CO", "S_Z_CO", "SH_ZH_CO", "TH_DH_CO",
                        "F_V_CD", "S_Z_CD", "SH_ZH_CD", "TH_DH_CD"],
                        41,
                        57)

        # Segmental Model #5 (Vowel ERB Category Distinction Degradations)
        specifics5 = (["IY_IH_BCD_ERB", "UW_UH_BCD_ERB", "AA_AE_BCD_ERB",
                        "IY_IH_WCD_ERB", "UW_UH_WCD_ERB", "AA_AE_WCD_ERB",
                        "IY_IH_CO_ERB", "UW_UH_CO_ERB", "AA_AE_CO_ERB",
                        "IY_IH_CD_ERB", "UW_UH_CD_ERB", "AA_AE_CD_ERB"],
                        57,
                        69)

        # Segmental Model #6 (Vowel DUR Category Distinction Degradations)
        specifics6 = (["IY_IH_BCD_DUR", "UW_UH_BCD_DUR", "AA_AE_BCD_DUR",
                        "IY_IH_WCD_DUR", "UW_UH_WCD_DUR", "AA_AE_WCD_DUR",
                        "IY_IH_CO_DUR", "UW_UH_CO_DUR", "AA_AE_CO_DUR",
                        "IY_IH_CD_DUR", "UW_UH_CD_DUR", "AA_AE_CD_DUR"],
                        69,
                        81)

        for (specifics, start, end), modelName in zip([specifics2, specifics3, specifics4, specifics5, specifics6], ['phonetic_contrasts', 'plosive_categories', 'fricative_categories', 'vowel_erb_categories', 'vowel_dur_categories']):
            columns = ['ID', 'Age', 'Gender'] + specifics + taskList
            DF = pd.DataFrame([bigList[i][:3] + bigList[i][start:end] + [bigList[i][-1]] for i in range(len(bigList))], columns = columns)
            DF = removeNaN(DF)            
            DF.to_csv('./segmental/SegmentalMeasures_{}-{}-{}-{}.csv'.format(which, task, saveName, modelName), index = False)

    shutil.rmtree("tmpGlobal")


if __name__ == "__main__":

    main("Full_wave_enhanced_audio", "categorical", False)
