#!/usr/bin/env python3

from joblib import Parallel, delayed
from numpy.random import seed
import multiprocessing as mp
from scipy import stats
import pandas as pd
import numpy as np
import glob
import os

seed(42)

# I can't remember if I already have some code for this stuff so I'm starting over. 
# This script is going to take in all of the extracted measurements from:
    # eGemaps
    # Compare
    # Auditory
# and run ANOVA-style significance testing on whatever rejects the null hypothesis 
# for Levene's test for Homogeneity of Variance


# An ANOVA compares the means of more than 2 groups; ergo I shouldn't actually do it because I just have two groups. 
# I could actually run an ANOVA test on the MMSE data, but that's not for now
# Tutorial for future use: https://www.reneshbedre.com/blog/anova.html
def anovaCall(df, leveneList):

    pass


# First we have to determine which of the 'center' variations for the Levene's test we need to use
def determineCenter(cc, cd):

    ###############################
    # Check if normally distributed
    ###############################
    a, b = False, False
    ccStat, ccP = stats.shapiro(cc)
    if ccP > 0.05: # Sample is Gaussian (normal); We should be running parametric statistical methods with these data
        a = True
    cdStat, cdP = stats.shapiro(cd)
    if cdP > 0.5:
        b = True

    # Both need to be normally distributed to use 'mean' (Recommended for symmetric, moderate-tailed distributions)     
    if a and b:
        center = 'mean'
        return center, ccStat, ccP, cdStat, cdP
    # Now that we know it's not normally distributed, do we have a skewed distribution or a heavily-tailed distribution?
    else:
        ###############################
        # Check for skewness
        ###############################
        c, d = False, False
        ccStatSkew, ccPSkew = stats.skewtest(cc)
        if ccPSkew > 0.05: # Sample rejects the null hypothesis that the skewness of the observations that the sample was drawn from is the same as that of a corresponding normal (Gaussian) distribution
            c = True
        cdStatSkew, cdPSkew = stats.skewtest(cd)
        if cdPSkew > 0.5:
            d = True

        ###############################
        # Check for kurtosis
        ###############################
        e, f = False, False
        ccStatKurt, ccPKurt = stats.kurtosistest(cc)
        if ccPKurt > 0.05: # Sample rejects the null hypothesis that the kurtosis of the observations are normal
            e = True
        cdStatKurt, cdPKurt = stats.kurtosistest(cd)
        if cdPKurt > 0.5:
            f = True

        # NOTE
        # Skewness is a quantification of how mucha  distribution is pushed left or right; a measure of asymmetry in the distribution
            # If the data are skewed, we need to use the median of the data for the theoretical center
        # Kurtosis quantifies how much of the distribution is in the tail. 
            # If the data are kurtotic, we need to use a "trimmed" center

        # I think the way to do this is to pick whichever array is strongest in terms of p-value for rejecting the null hypothesis and 
        # use that to determine 'median' vs 'trimmed'. The logic of this is tenuous, but it seems a good way to run a bunch of these
        strongest = np.argmax([ccPSkew, cdPSkew, ccPKurt, cdPKurt])
        if strongest == 0 or strongest == 1:
            center = 'median'
            return center, ccStatSkew, ccPSkew, cdStatSkew, cdPSkew
        elif strongest == 2 or strongest == 3:
            center = 'trimmed'
            return center, ccStatKurt, ccPKurt, cdStatKurt, cdPKurt


def multiProcLeveneHOV(i):

    cc, cd = i
    # Figure out the center strategy for the test
    center, ccStat, ccP, cdStat, cdP = determineCenter(cc, cd) # I don't know if I really need any of this information saved out
                                                               # All it really indicates is how the decision was arrived at for the Levene strategy. 

    # Run the test
    statistic, pvalue = stats.levene(cc, cd, center = center)
    return center, statistic, pvalue


def leveneHOV(df):

    ccDF = df[df['Condition'] == 'cc']
    cdDF = df[df['Condition'] == 'cd']

    X = Parallel(n_jobs=mp.cpu_count())(delayed(multiProcLeveneHOV)((ccDF[feature].tolist(), cdDF[feature].tolist())) for feature in df.columns[:-1])
    leveneCenterList, leveneStatisticList, levenePValueList = map(list, zip(*X))
    df = pd.DataFrame(list(zip(leveneCenterList, leveneStatisticList, levenePValueList)), columns = ['Levene Center Strategy', 'Levene Test Statistic', 'Levene p-Value'])
    return df


def multiProctTest(i):

    cc, cd, feature = i
    statistic, pvalue = stats.ttest_ind(cc, cd, equal_var = False, nan_policy = 'raise') # Equal_var means Welch, not student's t-test
    return feature, statistic, pvalue


# Since we cannot guarantee that the size of the arrays will be equal (we have balanced participants here, but not 
# balanced number of utterances), we have to perform the Welch t-test to see if there is a significant difference
def tTestInd(df):

    ccDF = df[df['Condition'] == 'cc']
    cdDF = df[df['Condition'] == 'cd']

    X = Parallel(n_jobs=mp.cpu_count())(delayed(multiProctTest)((ccDF[feature].tolist(), cdDF[feature].tolist(), feature)) for feature in df.columns[:-1])
    featureList, welchStatisticList, welchPValueList = map(list, zip(*X))
    df = pd.DataFrame(list(zip(featureList, welchStatisticList, welchPValueList)), columns = ['Feature', 'Welch Test Statistic', 'Welch p-Value'])
    return df


# our hand-crafted auditory features aren't extractable in the same way because we also 
# want the OIM features
def getAuditory(exp_dir, names):

    dfs = list()
    for ddir in ['train', 'dev']:
        info = pd.read_csv(os.path.join(exp_dir, ddir, 'df.csv'), sep = '\t')
        info['ID'] = info['ID'].str.strip()
        csvs = glob.glob(os.path.join(exp_dir, ddir, 'csv', '*.csv'))
        for csv in csvs:
            df = pd.read_csv(csv, names = names)
            name = csv.split('/')[-1].split('-')[0]
            # Get condition
            condition = info.loc[info['ID'] == name]['Label'].tolist()[0]
            
            # Get intelligibility metrics at 70, 50, 30% intelligibility levels
            for level in ['70', '50', '30']:
                levelCSV = csv.replace('/csv/', '/csv/{}/'.format(level))
                levelDF = pd.read_csv(levelCSV, names = names +   ['SMN DWGP {}'.format(level),
                                                                   'SSN DWGP {}'.format(level),
                                                                   'SMN SII {}'.format(level),
                                                                   'SSN SII {}'.format(level),
                                                                   'SMN STI {}'.format(level),
                                                                   'SSN STI {}'.format(level),
                                        ])
                df = pd.concat([df, levelDF.iloc[:,-6:]], axis = 1)
            df['Condition'] = condition
            dfs.append(df)
    DF = pd.concat(dfs)
    return DF


# Make it go brrrr
def multiProcessingCall(i):

    csv, condition, names = i

    df = pd.read_csv(csv, names = names)
    df['Condition'] = condition
    return df


# gemaps and compare are both readable using the same sort of pipeline
def get(exp_dir, names):

    dfs = list()
    for ddir in ['train', 'dev']:
        info = pd.read_csv(os.path.join(exp_dir, ddir, 'df.csv'), sep = '\t')
        info['ID'] = info['ID'].str.strip()
        csvs = glob.glob(os.path.join(exp_dir, ddir, 'csv', '*.csv'))
        conditions = [info.loc[info['ID'] == csv.split('/')[-1].split('-')[0]]['Label'].tolist()[0] for csv in csvs]

        # This is too slow not multi-processed
        X = Parallel(n_jobs=mp.cpu_count())(delayed(multiProcessingCall)((csv, condition, names)) for csv, condition in list(zip(csvs[:], conditions[:])))
        # for csv, condition in list(zip(csvs[:5], conditions[:5])):
        #     df = multiProcessingCall((csv, condition, names))
        for df in X:
            dfs.append(df)
    DF = pd.concat(dfs)
    return DF


def getData(name):

    if name == "egemaps":
        exp_dir = "/tmp/tmp.RFVA79Kf0X/data"
        if os.path.isfile(os.path.join(exp_dir, 'features_all.csv')):
            df = pd.read_csv(os.path.join(exp_dir, 'features_all.csv'))
        else:
            names = ['F0semitoneFrom27.5Hz_sma3nz_amean', 'F0semitoneFrom27.5Hz_sma3nz_stddevNorm', 'F0semitoneFrom27.5Hz_sma3nz_percentile20.0', 'F0semitoneFrom27.5Hz_sma3nz_percentile50.0', 'F0semitoneFrom27.5Hz_sma3nz_percentile80.0', 'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2', 'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope', 'F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope', 'F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope', 'F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope', 'loudness_sma3_amean', 'loudness_sma3_stddevNorm', 'loudness_sma3_percentile20.0', 'loudness_sma3_percentile50.0', 'loudness_sma3_percentile80.0', 'loudness_sma3_pctlrange0-2', 'loudness_sma3_meanRisingSlope', 'loudness_sma3_stddevRisingSlope', 'loudness_sma3_meanFallingSlope', 'loudness_sma3_stddevFallingSlope', 'spectralFlux_sma3_amean', 'spectralFlux_sma3_stddevNorm', 'mfcc1_sma3_amean', 'mfcc1_sma3_stddevNorm', 'mfcc2_sma3_amean', 'mfcc2_sma3_stddevNorm', 'mfcc3_sma3_amean', 'mfcc3_sma3_stddevNorm', 'mfcc4_sma3_amean', 'mfcc4_sma3_stddevNorm', 'jitterLocal_sma3nz_amean', 'jitterLocal_sma3nz_stddevNorm', 'shimmerLocaldB_sma3nz_amean', 'shimmerLocaldB_sma3nz_stddevNorm', 'HNRdBACF_sma3nz_amean', 'HNRdBACF_sma3nz_stddevNorm', 'logRelF0-H1-H2_sma3nz_amean', 'logRelF0-H1-H2_sma3nz_stddevNorm', 'logRelF0-H1-A3_sma3nz_amean', 'logRelF0-H1-A3_sma3nz_stddevNorm', 'F1frequency_sma3nz_amean', 'F1frequency_sma3nz_stddevNorm', 'F1bandwidth_sma3nz_amean', 'F1bandwidth_sma3nz_stddevNorm', 'F1amplitudeLogRelF0_sma3nz_amean', 'F1amplitudeLogRelF0_sma3nz_stddevNorm', 'F2frequency_sma3nz_amean', 'F2frequency_sma3nz_stddevNorm', 'F2bandwidth_sma3nz_amean', 'F2bandwidth_sma3nz_stddevNorm', 'F2amplitudeLogRelF0_sma3nz_amean', 'F2amplitudeLogRelF0_sma3nz_stddevNorm', 'F3frequency_sma3nz_amean', 'F3frequency_sma3nz_stddevNorm', 'F3bandwidth_sma3nz_amean', 'F3bandwidth_sma3nz_stddevNorm', 'F3amplitudeLogRelF0_sma3nz_amean', 'F3amplitudeLogRelF0_sma3nz_stddevNorm', 'alphaRatioV_sma3nz_amean', 'alphaRatioV_sma3nz_stddevNorm', 'hammarbergIndexV_sma3nz_amean', 'hammarbergIndexV_sma3nz_stddevNorm', 'slopeV0-500_sma3nz_amean', 'slopeV0-500_sma3nz_stddevNorm', 'slopeV500-1500_sma3nz_amean', 'slopeV500-1500_sma3nz_stddevNorm', 'spectralFluxV_sma3nz_amean', 'spectralFluxV_sma3nz_stddevNorm', 'mfcc1V_sma3nz_amean', 'mfcc1V_sma3nz_stddevNorm', 'mfcc2V_sma3nz_amean', 'mfcc2V_sma3nz_stddevNorm', 'mfcc3V_sma3nz_amean', 'mfcc3V_sma3nz_stddevNorm', 'mfcc4V_sma3nz_amean', 'mfcc4V_sma3nz_stddevNorm', 'alphaRatioUV_sma3nz_amean', 'hammarbergIndexUV_sma3nz_amean', 'slopeUV0-500_sma3nz_amean', 'slopeUV500-1500_sma3nz_amean', 'spectralFluxUV_sma3nz_amean', 'loudnessPeaksPerSec', 'VoicedSegmentsPerSec', 'MeanVoicedSegmentLengthSec', 'StddevVoicedSegmentLengthSec', 'MeanUnvoicedSegmentLength', 'StddevUnvoicedSegmentLength', 'equivalentSoundLevel_dBp']
            df = get(exp_dir, names)
            df.to_csv(os.path.join(exp_dir, 'features_all.csv'), index = False)

    elif name == "compare":
        exp_dir = "/tmp/tmp.Psc7g4V77e/data"
        if os.path.isfile(os.path.join(exp_dir, 'features_all.csv')):
            df = pd.read_csv(os.path.join(exp_dir, 'features_all.csv'))
        else:
            import pickle
            with open('/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Code/PYTHON/openSMILE/compare_feature_names.pkl', 'rb') as f:
                names = pickle.load(f)
            df = get(exp_dir, names)
            df.to_csv(os.path.join(exp_dir, 'features_all.csv'), index = False)

    elif name == "auditory":
        exp_dir = "/tmp/tmp.OgrVUR6iE4/data"
        if os.path.isfile(os.path.join(exp_dir, 'features_all.csv')):
            df = pd.read_csv(os.path.join(exp_dir, 'features_all.csv'))
        else:
            names = ['Avg. Word Dur.', 'Avg. Sil. Dur.', 'Dynamic Range', 'Energy', 'Intensity', 'ZCR', 'Root Mean Square', 'Sound Pressure Level', 'Consonant Vowel Ratio', 'B_CoG', 'B_Kur', 'B_Ske', 'B_Std', 'CH_CoG', 'CH_Kur', 'CH_Ske', 'CH_Std', 'D_CoG', 'D_Kur', 'D_Ske', 'D_Std', 'DH_CoG', 'DH_Kur', 'DH_Ske', 'DH_Std', 'CX_CoG', 'CX_Kur', 'CX_Ske', 'CX_Std', 'EL_CoG', 'EL_Kur', 'EL_Ske', 'EL_Std', 'EM_CoG', 'EM_Kur', 'EM_Ske', 'EM_Std', 'EN_CoG', 'EN_Kur', 'EN_Ske', 'EN_Std', 'F_CoG', 'F_Kur', 'F_Ske', 'F_Std', 'G_CoG', 'G_Kur', 'G_Ske', 'G_Std', 'H_CoG', 'H_Kur', 'H_Ske', 'H_Std', 'JH_CoG', 'JH_Kur', 'JH_Ske', 'JH_Std', 'K_CoG', 'K_Kur', 'K_Ske', 'K_Std', 'L_CoG', 'L_Kur', 'L_Ske', 'L_Std', 'M_CoG', 'M_Kur', 'M_Ske', 'M_Std', 'N_CoG', 'N_Kur', 'N_Ske', 'N_Std', 'NX_CoG', 'NX_Kur', 'NX_Ske', 'NX_Std', 'NG_CoG', 'NG_Kur', 'NG_Ske', 'NG_Std', 'P_CoG', 'P_Kur', 'P_Ske', 'P_Std', 'Q_CoG', 'Q_Kur', 'Q_Ske', 'Q_Std', 'R_CoG', 'R_Kur', 'R_Ske', 'R_Std', 'S_CoG', 'S_Kur', 'S_Ske', 'S_Std', 'SH_CoG', 'SH_Kur', 'SH_Ske', 'SH_Std', 'T_CoG', 'T_Kur', 'T_Ske', 'T_Std', 'TH_CoG', 'TH_Kur', 'TH_Ske', 'TH_Std', 'V_CoG', 'V_Kur', 'V_Ske', 'V_Std', 'W_CoG', 'W_Kur', 'W_Ske', 'W_Std', 'WH_CoG', 'WH_Kur', 'WH_Ske', 'WH_Std', 'Y_CoG', 'Y_Kur', 'Y_Ske', 'Y_Std', 'Z_CoG', 'Z_Kur', 'Z_Ske', 'Z_Std', 'ZH_CoG', 'ZH_Kur', 'ZH_Ske', 'ZH_Std', 'Avg. Cons. Dur.', 'AA-F0_1', 'AA-F0_2', 'AA-F0_3', 'AA-F0_4', 'AA-F0_5', 'AA-F1_1', 'AA-F2_1', 'AA-F3_1', 'AA-F1_2', 'AA-F2_2', 'AA-F3_2', 'AA-F1_3', 'AA-F2_3', 'AA-F3_3', 'AA-F1_4', 'AA-F2_4', 'AA-F3_4', 'AA-F1_5', 'AA-F2_5', 'AA-F3_5', 'AE-F0_1', 'AE-F0_2', 'AE-F0_3', 'AE-F0_4', 'AE-F0_5', 'AE-F1_1', 'AE-F2_1', 'AE-F3_1', 'AE-F1_2', 'AE-F2_2', 'AE-F3_2', 'AE-F1_3', 'AE-F2_3', 'AE-F3_3', 'AE-F1_4', 'AE-F2_4', 'AE-F3_4', 'AE-F1_5', 'AE-F2_5', 'AE-F3_5', 'AH-F0_1', 'AH-F0_2', 'AH-F0_3', 'AH-F0_4', 'AH-F0_5', 'AH-F1_1', 'AH-F2_1', 'AH-F3_1', 'AH-F1_2', 'AH-F2_2', 'AH-F3_2', 'AH-F1_3', 'AH-F2_3', 'AH-F3_3', 'AH-F1_4', 'AH-F2_4', 'AH-F3_4', 'AH-F1_5', 'AH-F2_5', 'AH-F3_5', 'AO-F0_1', 'AO-F0_2', 'AO-F0_3', 'AO-F0_4', 'AO-F0_5', 'AO-F1_1', 'AO-F2_1', 'AO-F3_1', 'AO-F1_2', 'AO-F2_2', 'AO-F3_2', 'AO-F1_3', 'AO-F2_3', 'AO-F3_3', 'AO-F1_4', 'AO-F2_4', 'AO-F3_4', 'AO-F1_5', 'AO-F2_5', 'AO-F3_5', 'AW-F0_1', 'AW-F0_2', 'AW-F0_3', 'AW-F0_4', 'AW-F0_5', 'AW-F1_1', 'AW-F2_1', 'AW-F3_1', 'AW-F1_2', 'AW-F2_2', 'AW-F3_2', 'AW-F1_3', 'AW-F2_3', 'AW-F3_3', 'AW-F1_4', 'AW-F2_4', 'AW-F3_4', 'AW-F1_5', 'AW-F2_5', 'AW-F3_5', 'AX-F0_1', 'AX-F0_2', 'AX-F0_3', 'AX-F0_4', 'AX-F0_5', 'AX-F1_1', 'AX-F2_1', 'AX-F3_1', 'AX-F1_2', 'AX-F2_2', 'AX-F3_2', 'AX-F1_3', 'AX-F2_3', 'AX-F3_3', 'AX-F1_4', 'AX-F2_4', 'AX-F3_4', 'AX-F1_5', 'AX-F2_5', 'AX-F3_5', 'AXR-F0_1', 'AXR-F0_2', 'AXR-F0_3', 'AXR-F0_4', 'AXR-F0_5', 'AXR-F1_1', 'AXR-F2_1', 'AXR-F3_1', 'AXR-F1_2', 'AXR-F2_2', 'AXR-F3_2', 'AXR-F1_3', 'AXR-F2_3', 'AXR-F3_3', 'AXR-F1_4', 'AXR-F2_4', 'AXR-F3_4', 'AXR-F1_5', 'AXR-F2_5', 'AXR-F3_5', 'AY-F0_1', 'AY-F0_2', 'AY-F0_3', 'AY-F0_4', 'AY-F0_5', 'AY-F1_1', 'AY-F2_1', 'AY-F3_1', 'AY-F1_2', 'AY-F2_2', 'AY-F3_2', 'AY-F1_3', 'AY-F2_3', 'AY-F3_3', 'AY-F1_4', 'AY-F2_4', 'AY-F3_4', 'AY-F1_5', 'AY-F2_5', 'AY-F3_5', 'EH-F0_1', 'EH-F0_2', 'EH-F0_3', 'EH-F0_4', 'EH-F0_5', 'EH-F1_1', 'EH-F2_1', 'EH-F3_1', 'EH-F1_2', 'EH-F2_2', 'EH-F3_2', 'EH-F1_3', 'EH-F2_3', 'EH-F3_3', 'EH-F1_4', 'EH-F2_4', 'EH-F3_4', 'EH-F1_5', 'EH-F2_5', 'EH-F3_5', 'ER-F0_1', 'ER-F0_2', 'ER-F0_3', 'ER-F0_4', 'ER-F0_5', 'ER-F1_1', 'ER-F2_1', 'ER-F3_1', 'ER-F1_2', 'ER-F2_2', 'ER-F3_2', 'ER-F1_3', 'ER-F2_3', 'ER-F3_3', 'ER-F1_4', 'ER-F2_4', 'ER-F3_4', 'ER-F1_5', 'ER-F2_5', 'ER-F3_5', 'EY-F0_1', 'EY-F0_2', 'EY-F0_3', 'EY-F0_4', 'EY-F0_5', 'EY-F1_1', 'EY-F2_1', 'EY-F3_1', 'EY-F1_2', 'EY-F2_2', 'EY-F3_2', 'EY-F1_3', 'EY-F2_3', 'EY-F3_3', 'EY-F1_4', 'EY-F2_4', 'EY-F3_4', 'EY-F1_5', 'EY-F2_5', 'EY-F3_5', 'IH-F0_1', 'IH-F0_2', 'IH-F0_3', 'IH-F0_4', 'IH-F0_5', 'IH-F1_1', 'IH-F2_1', 'IH-F3_1', 'IH-F1_2', 'IH-F2_2', 'IH-F3_2', 'IH-F1_3', 'IH-F2_3', 'IH-F3_3', 'IH-F1_4', 'IH-F2_4', 'IH-F3_4', 'IH-F1_5', 'IH-F2_5', 'IH-F3_5', 'IX-F0_1', 'IX-F0_2', 'IX-F0_3', 'IX-F0_4', 'IX-F0_5', 'IX-F1_1', 'IX-F2_1', 'IX-F3_1', 'IX-F1_2', 'IX-F2_2', 'IX-F3_2', 'IX-F1_3', 'IX-F2_3', 'IX-F3_3', 'IX-F1_4', 'IX-F2_4', 'IX-F3_4', 'IX-F1_5', 'IX-F2_5', 'IX-F3_5', 'IY-F0_1', 'IY-F0_2', 'IY-F0_3', 'IY-F0_4', 'IY-F0_5', 'IY-F1_1', 'IY-F2_1', 'IY-F3_1', 'IY-F1_2', 'IY-F2_2', 'IY-F3_2', 'IY-F1_3', 'IY-F2_3', 'IY-F3_3', 'IY-F1_4', 'IY-F2_4', 'IY-F3_4', 'IY-F1_5', 'IY-F2_5', 'IY-F3_5', 'OW-F0_1', 'OW-F0_2', 'OW-F0_3', 'OW-F0_4', 'OW-F0_5', 'OW-F1_1', 'OW-F2_1', 'OW-F3_1', 'OW-F1_2', 'OW-F2_2', 'OW-F3_2', 'OW-F1_3', 'OW-F2_3', 'OW-F3_3', 'OW-F1_4', 'OW-F2_4', 'OW-F3_4', 'OW-F1_5', 'OW-F2_5', 'OW-F3_5', 'OY-F0_1', 'OY-F0_2', 'OY-F0_3', 'OY-F0_4', 'OY-F0_5', 'OY-F1_1', 'OY-F2_1', 'OY-F3_1', 'OY-F1_2', 'OY-F2_2', 'OY-F3_2', 'OY-F1_3', 'OY-F2_3', 'OY-F3_3', 'OY-F1_4', 'OY-F2_4', 'OY-F3_4', 'OY-F1_5', 'OY-F2_5', 'OY-F3_5', 'UH-F0_1', 'UH-F0_2', 'UH-F0_3', 'UH-F0_4', 'UH-F0_5', 'UH-F1_1', 'UH-F2_1', 'UH-F3_1', 'UH-F1_2', 'UH-F2_2', 'UH-F3_2', 'UH-F1_3', 'UH-F2_3', 'UH-F3_3', 'UH-F1_4', 'UH-F2_4', 'UH-F3_4', 'UH-F1_5', 'UH-F2_5', 'UH-F3_5', 'UW-F0_1', 'UW-F0_2', 'UW-F0_3', 'UW-F0_4', 'UW-F0_5', 'UW-F1_1', 'UW-F2_1', 'UW-F3_1', 'UW-F1_2', 'UW-F2_2', 'UW-F3_2', 'UW-F1_3', 'UW-F2_3', 'UW-F3_3', 'UW-F1_4', 'UW-F2_4', 'UW-F3_4', 'UW-F1_5', 'UW-F2_5', 'UW-F3_5', 'UX-F0_1', 'UX-F0_2', 'UX-F0_3', 'UX-F0_4', 'UX-F0_5', 'UX-F1_1', 'UX-F2_1', 'UX-F3_1', 'UX-F1_2', 'UX-F2_2', 'UX-F3_2', 'UX-F1_3', 'UX-F2_3', 'UX-F3_3', 'UX-F1_4', 'UX-F2_4', 'UX-F3_4', 'UX-F1_5', 'UX-F2_5', 'UX-F3_5', 'Avg. Voca. Dur.']
            df = getAuditory(exp_dir, names)
            df.to_csv(os.path.join(exp_dir, 'features_all.csv'), index = False)

    return df, exp_dir


def main(name):

    df, exp_dir = getData(name)

    # We can run a Welch's t-test on arrays of varying sizes because it doesn't assume homogeneity of variance
    output = tTestInd(df)

    # In addition to Welch's t-test, we can determine if we are dealing with samples that equate to roughly equal variances
    levene = leveneHOV(df)
    output = pd.concat([output, levene], axis = 1)

    # We can get at an Analysis of Variance (ANOVA) for *some* of our features
    # here (if there are more than two groups we're interested in), but only if we can 
    # reject the null hypothesis surrounding Levene's test related to 
    # determining if we have homogeneity of variance between the groups of observations
    # anova = anovaCall(df, output['Levene p-Value'].to_list())

    # Save something out for each of the three
    output.to_csv('{}.csv'.format(name), index = False)

if __name__ == "__main__":

    for extractedFeatureSet in ['egemaps', 'compare', 'auditory']:
        main(extractedFeatureSet)
    # main('egemaps')
    # main('compare')
    # main('auditory')
