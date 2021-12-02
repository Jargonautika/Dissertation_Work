#!/usr/bin/env python3

import os
import glob
import vowels
import argparse
import consonants
import numpy as np
import pandas as pd
import multiprocessing as mp
from joblib import Parallel, delayed


def consolidateFiles(z = 2.58, exp_dir = '/tmp/tmp.IHHR7Bl2yh'):

    for grouping in ['consonant', 'vowel', 'vowelSpace']:
    # for grouping in ['vowel']: # Something is wrong with the vowels. We've got a lot of duplicate in there

        files = glob.glob('{}/data/acoustics/{}/{}/*'.format(exp_dir, str(z), grouping))
        DF = pd.concat([pd.read_csv(file) for file in files], ignore_index = True)
        # DF = DF.dropna()
        DF.to_csv('{}/data/acoustics/{}/{}/{}_consolidated.csv'.format(exp_dir, str(z), grouping, str(z)), index = False)


def groupExploration(X, Y, exp_dir, z = 2.58):
    
    # 24 of these measures return as non-significantly different from a normal distribution by the Shapiro-Wilk test of normalcy
    df = pd.DataFrame(X + Y, columns = ['Segment_Type', 'Segment', 'Condition', 'Measure', 'Number_of_Data_Points', 'Median', 'Mean', 'Standard_Error', 'Variance', 'Standard_Error', 'Coefficiant_of_Variance', 'Skewness', 'Skewness.2SE', 'Kurtosis', 'Kurtosis.2SE', 'Shapiro-Wilk_Statistic', 'Wilk_pValue', 'Levene', 'Levene_pValue'])
    df.to_csv('{}/data/acoustics/groupExploration_{}.csv'.format(exp_dir, str(z)), index = False)


def analyzeVowels(filename, vPlotFilename, exp_dir, z = 2.58):

    df = pd.read_csv(filename)
    vPlotDF = pd.read_csv(vPlotFilename)
    return vowels.main(df, vPlotDF, exp_dir, z)


def analyzeConsonants(filename, exp_dir, z = 2.58):

    df = pd.read_csv(filename)
    return consonants.main(df, exp_dir, z)


# Vestigial code; not used right now
def logicalFilter(filename, name):

    # Read in the files (delete this later)
    df = pd.read_csv(filename)

    # https://asa.scitation.org/doi/pdf/10.1121/1.411872 page 6; rounded to the nearest tens place (diphthongs don't factor in here)
    maleVowelDict = {'IY': (340, 2300),
                        'IH': (430, 2000),
                        'EY': (480, 2090),
                        'EH': (580, 1800), 
                        'AE': (590, 1950),
                        'AA': (770, 1330), 
                        'AO': (650, 1000), 
                        'OW': (500, 910),
                        'UH': (470, 1120),
                        'UW': (380, 1000), 
                        'AH': (620, 1200),
                        'ER': (470, 1380),
                        'AX': (500, 1500), # From Shosted's class (schwa)
                        }

    femaleVowelDict = {'IY': (440, 2760),
                        'IH': (480, 2370),
                        'EY': (540, 2530),
                        'EH': (730, 2060), 
                        'AE': (220, 2350),
                        'AA': (940, 1550), 
                        'AO': (780, 1140), 
                        'OW': (560, 1040),
                        'UH': (520, 1230),
                        'UW': (460, 1100), 
                        'AH': (750, 1430),
                        'ER': (520, 1590),
                        'AX': (600, 1800), # From Shosted's class (schwa) (double check with Ladefoged 1996) (https://is.muni.cz/th/ndch7/Markova_Schwa.pdf page 86)
                        }

    maleConsonantDict = {'B', 
                         'P', 
                         'M', 
                         'W', 
                         'WH',
                         'F', 
                         'V', 
                         'T', 
                         'TH', 
                         'DH', 
                         'CH', 
                         'JH', 
                         'S', 
                         'Z', 
                         'D', 
                         'N', 
                         'R', 
                         'DX', 
                         'SH', 
                         'ZH', 
                         'L', 
                         'Y', 
                         'K', 
                         'G', 
                         'NX', 
                         'NG', 
                         'H', 
                         'Q'
                         }  

    femaleConsonantDict = {'B', 
                           'P', 
                           'M', 
                           'W', 
                           'WH',
                           'F', 
                           'V', 
                           'T', 
                           'TH', 
                           'DH', 
                           'CH', 
                           'JH', 
                           'S', 
                           'Z', 
                           'D', 
                           'N', 
                           'R', 
                           'DX', 
                           'SH', 
                           'ZH', 
                           'L', 
                           'Y', 
                           'K', 
                           'G', 
                           'NX', 
                           'NG', 
                           'H', 
                           'Q'
                         } 

    # I may be doing things incorrectly with consonants. We may want to break down the 
    # different places and/or manners of articulation and extract appropriate measures from
    # each since approximants should be treated more like a vowel while for fricatives the 
    # spectral moments may actually matter                
    if name == 'consonants':

        pass

    elif name == 'vowels':

        together = list()
        male = df[df['gender'] == 0]
        female = df[df['gender'] == 1]

        # Treat each gendered subset separately
        for gender, genderedDict in [(male, maleVowelDict), 
                                     (female, femaleVowelDict)]:

            # Iterate over all of the vowels in the corpus
            for vowel in genderedDict:
                subDF = gender[gender['Segment'] == vowel]

                if subDF.shape[0] == 0:
                    continue
                
                # Treat F1 and F2 separately
                F1DF = subDF[subDF['Measure'] == 'F1']
                F2DF = subDF[subDF['Measure'] == 'F2']
                F1, F2 = genderedDict[vowel]
    
                # Iteratively remove the outliers
                for miniDF, meanValue in [(F1DF, F1), (F2DF, F2)]:
                    X = miniDF.sort_values(by = ['Value'])
                    indices = X.index.tolist()

                    # I'm honestly not sure if this is the best way to do this. It means that if
                    # the overall mean of the dataframe is higher than what I want, I only delete the 
                    # high values, while the low values are left there. So for male /IY/ I delete ~300
                    # values above 600 Hz to get down to a mean of ~356 where I want ~340, but I still have
                    # a bunch of low values of ~100 Hz. Those are still wrong, but combined they work out to
                    # a good average. Maybe what I need to do is two passes?

                    # One-pass attempt:
                    # if np.mean(X['Value']) > meanValue:
                    #     side = -1 # We need to get rid of the too-big outliers
                    # else:
                    #     side = 0 # We need to get rid of the too-small outliers
                    # print(X.shape)
                    # while not (meanValue * 0.95) < np.mean(X['Value']) < (meanValue * 1.05):
                    #     X = X.drop(indices[side])
                    #     del indices[side]
                    # print(X.shape)
                    # together.append(X)

                    # Two-pass attempt:
                    reserve = X.copy() # Just in case we filter everything else out later
                    if np.mean(X['Value']) > meanValue:
                        side = -1 # We need to get rid of the too-big outliers
                    else:
                        side = 0 # We need to get rid of the too-small outliers
                    while not (meanValue * 0.5) < np.mean(X['Value']) < (meanValue * 1.5): # Set a wider window here and narrow down later
                        X = X.drop(indices[side])
                        del indices[side]

                    print(vowel, X.shape[0], reserve.shape[0])
                    sides = [-1, 0] # Switch back and forth
                    side = sides[side] # Flip sides before beginning narrow pass
                    try:
                        while not (meanValue * 0.75) < np.mean(X['Value']) < (meanValue * 1.25): # Set a narrower window here now
                            X = X.drop(indices[side])
                            del indices[side]
                            side = sides[side] # Flip back to the other side of outliers, working iteratively towards the middle

                        print(vowel, X.shape[0], reserve.shape[0])
                        print()
                        together.append(X)
                    except:
                        together.append(reserve)
                        print()

        return pd.concat(together)
        # 0.8;1.2   - 0.9;1.1   - (267307, 10) > (77596, 10) (29% preserved) - Too aggressive
        # 0.75;1.25 - 0.8;1.2 - (267307, 10) > (80116, 10) (30% preserved) - Too aggressive
        # 0.5;1.5 - 
    else:

        pass


def prepConsolidation(exp_dir, consonantsFile, vowelsFile, vowelPlotFile):

    cdDF = pd.read_csv('/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSS-IS2020-data/train/cd_meta_data.txt', sep = ';').rename(columns=lambda x: x.strip())
    cdDF['Label'] = [1 for i in range(cdDF.shape[0])]
    cdDF['gender'] = cdDF['gender'].str.replace('female', '1')
    cdDF['gender'] = cdDF['gender'].str.replace('male', '0').astype(int)
    ccDF = pd.read_csv('/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSS-IS2020-data/train/cc_meta_data.txt', sep = ';').rename(columns=lambda x: x.strip())
    ccDF['Label'] = [0 for i in range(ccDF.shape[0])]
    ccDF['gender'] = ccDF['gender'].str.replace('female', '0')
    ccDF['gender'] = ccDF['gender'].str.replace('male', '0').astype(int)

    # For the testing data frame 0 means Male and 1 means Female; 0 means CC and 1 means CD
    teDF = pd.read_csv('/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSS-IS2020-data/meta_data_test.txt', sep = ';').rename(columns=lambda x: x.strip())
    # teDF['gender'] = teDF['gender'].astype(str).str.replace('0', 'male')
    # teDF['gender'] = teDF['gender'].astype(str).str.replace('1', 'female')
    # teDF['Label'] = teDF['Label'].astype(str).str.replace('0', 'cc')
    # teDF['Label'] = teDF['Label'].astype(str).str.replace('1', 'cd')

    X = pd.concat([ccDF, cdDF, teDF])
    X['ID'] = X['ID'].str.strip()

    # Concatenate all of the speaker-level observation files together
    C = pd.concat(pd.read_csv(x) for x in glob.glob(os.path.join(exp_dir, 'data', 'acoustics', '*consonants*')))
    V = pd.concat(pd.read_csv(x) for x in glob.glob(os.path.join(exp_dir, 'data', 'acoustics', '*vowels*')))

    # Convert wide data to long format
    C = pd.melt(C, id_vars = C.columns.tolist()[:4], value_vars = C.columns.tolist()[4:], var_name = 'Measure', value_name = 'Value').dropna()

    # This is not so easily done for vowels. Here we go by hand. 
    newVList, vPlotList = list(), list()
    for _, j in V.iterrows():
        for measure in ['F0', 'F1', 'F2', 'F3']:
            for timeStamp in ['01', '02', '03', '04', '05']:
                vPlotList.append([j['Speaker'], j['Waveform'], j['Condition'], j['Segment'], timeStamp, j['F1_{}'.format(timeStamp)], j['F2_{}'.format(timeStamp)]]) # For vowel plots
                newVList.append([j['Speaker'], j['Waveform'], j['Condition'], j['Segment'], measure, timeStamp, j['{}_{}'.format(measure, timeStamp)]])
        newVList.append([j['Speaker'], j['Waveform'], j['Condition'], j['Segment'], 'Duration', '00', j['Duration']])

        # Calculate the average across the timestamps for a given observation
        F1Avg = np.nanmean([j['F1_{}'.format(t)] for t in ['01', '02', '03', '04', '05']])
        F2Avg = np.nanmean([j['F2_{}'.format(t)] for t in ['01', '02', '03', '04', '05']])
        vPlotList.append([j['Speaker'], j['Waveform'], j['Condition'], j['Segment'], '06', F1Avg, F2Avg]) # Also or vowel plots; 06 here means average
    
    V = pd.DataFrame(newVList, columns = ['Speaker', 'Waveform', 'Condition', 'Segment', 'Measure', 'Rep', 'Value']).dropna()
    VPlot = pd.DataFrame(vPlotList, columns = ['Speaker', 'Waveform', 'Condition', 'Segment', 'Rep', 'F1', 'F2']).dropna()

    for df in [C, V, VPlot]: # Add in demographic data
        age, gender, mmse = list(), list(), list()
        speakerList = df['Speaker'].tolist()
        for speaker in speakerList:
            tmpDF = X[X['ID'] == speaker]
            age.append(tmpDF['age'].tolist()[0])
            gender.append(tmpDF['gender'].tolist()[0])
            mmse.append(tmpDF['mmse'].tolist()[0])

        df.insert(1, 'mmse', mmse)
        df.insert(1, 'gender', gender)
        df.insert(1, 'age', age)

        df['Condition'] = df['Condition'].str.replace('cc', '0')
        df['Condition'] = df['Condition'].str.replace('cd', '1').astype(int)

    # Add in a logical filter here to root out extranneous values # Don't do this; comment at bottom
    # C = logicalFilter(C, 'consonants')
    # V = logicalFilter(V, 'vowels')
    # VPlot = logicalFilter(VPlot, 'VPlot')

    # Save out the files
    C.to_csv(consonantsFile, index = False)
    V.to_csv(vowelsFile, index = False)
    VPlot.to_csv(vowelPlotFile, index = False)


def main(z = 2.58):

    # Run this with the STATS virtual environment
    parser = argparse.ArgumentParser(description='Run acoustic analyses on the consonants and vowels for all speakers by condition')
    parser.add_argument('exp_dir', nargs='?', type=str, help='Temporary experiment directory.', default = '/tmp/tmp.OgrVUR6iE4')
    args = parser.parse_args()

    consonantsFile = os.path.join(args.exp_dir, 'data', 'acoustics', 'C_consolidated.csv')
    vowelsFile = os.path.join(args.exp_dir, 'data', 'acoustics', 'V_consolidated.csv')
    vowelPlotFile = os.path.join(args.exp_dir, 'data', 'acoustics', 'Vowel_Space.csv')

    if not os.path.isfile(consonantsFile):
        prepConsolidation(args.exp_dir, consonantsFile, vowelsFile, vowelPlotFile)

    # Check if we've tried this zScore before
    outputPath = '{}/reports/plots/histograms/{}'.format(args.exp_dir, str(z))
    if not os.path.isdir(outputPath):
        os.mkdir(outputPath)
        os.mkdir('{}/data/acoustics/{}'.format(args.exp_dir, str(z)))
        os.mkdir('{}/reports/plots/vowelSpace/{}'.format(args.exp_dir, str(z)))

    # Remove histograms and vowel plots since the names are pseudo-randomly 
    # joined together and we'll end up with lots of duplicates otherwise
    else:
        histFiles = glob.glob('{}/reports/plots/histograms/{}/*'.format(args.exp_dir, z))
        for f in histFiles:
            os.remove(f)
        vowelSpaceFiles = glob.glob('{}/reports/plots/vowelSpace/{}/*'.format(args.exp_dir, z))
        for f in vowelSpaceFiles:
            os.remove(f)

    # X = analyzeConsonants(consonantsFile, args.exp_dir, z)
    Y = analyzeVowels(vowelsFile, vowelPlotFile, args.exp_dir, z)

    # groupExploration(X, Y, args.exp_dir, z)


if __name__ == "__main__":

    # Run an experiment where we try a bunch of z score values to see how the R code deals with the filtering
    # http://www.ltcconline.net/greenl/courses/201/estimation/smallConfLevelTable.htm
    zScoreList =  [1.04, 1.15, 1.28, 1.44, 1.645, 1.75, 1.96, 2.05, 2.33, 2.58]
    # Parallel(n_jobs=mp.cpu_count()/2)(delayed(main)(z) for z in zScoreList) # I don't think this is actually set up to be able to run these in parallel
    # for z in zScoreList:
    #     print('Now working on z {}'.format(z))
    #     main(z)
    main(1.44)

    # Prepare for analyses in R Studio
    # for z in zScoreList:
    #     consolidateFiles(z = z)

    # This approach below may be ill-advised since there's no rationale for throwing out arbitrary values 
        # based on closeness to expected formant values

    # for filename, name in [('/tmp/tmp.IHHR7Bl2yh/data/acoustics/C_consolidated.csv', 'consonants'),
    #                        ('/tmp/tmp.IHHR7Bl2yh/data/acoustics/V_consolidated.csv', 'vowels'),
    #                        ('/tmp/tmp.IHHR7Bl2yh/data/acoustics/Vowel_Space.csv', 'VPlot')]:
    #     logicalFilter(filename, name)
