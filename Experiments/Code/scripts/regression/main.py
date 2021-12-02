#!/usr/bin/env python3

import os
import sys
import glob
import joblib
import argparse
import warnings
import numpy as np
import pandas as pd
import huberRegression
import linearLeastSquares
import ordinaryLeastSquares

from matplotlib import pyplot as plt
from consolidator import Consolidator
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold


def writeContributionReport(contributions, featureNames, exp_dir, experiment, baseline):

    df = pd.DataFrame(list(zip(contributions, featureNames)), columns = ['Contributions', 'Feature Names'])
    df.to_csv(os.path.join(exp_dir, 'reports', 'feature-contribution-{}-baseRMSE={}.csv'.format(experiment, baseline)), index = False)


def writeReport(regressors, exp_dir, experiment):

    # Part of cross-validation; sue me
    groups = [list(row) for row in zip(*reversed(regressors))]

    scoreList = list()
    for group in groups:
        labels, scores = Consolidator(group)._get_means()
        scoreList.append(scores)

    df = pd.DataFrame(scoreList, columns = labels)
    df.to_csv(os.path.join(exp_dir, 'reports', '{}-regressors.csv'.format(experiment)), index = False)

    print(df)

def getFeatureNames(algorithm):

    if algorithm == "gemaps":
        return ['F0semitoneFrom27.5Hz_sma3nz_amean', 'F0semitoneFrom27.5Hz_sma3nz_stddevNorm', 'F0semitoneFrom27.5Hz_sma3nz_percentile20.0', 'F0semitoneFrom27.5Hz_sma3nz_percentile50.0', 'F0semitoneFrom27.5Hz_sma3nz_percentile80.0', 'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2', 'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope', 'F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope', 'F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope', 'F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope', 'loudness_sma3_amean', 'loudness_sma3_stddevNorm', 'loudness_sma3_percentile20.0', 'loudness_sma3_percentile50.0', 'loudness_sma3_percentile80.0', 'loudness_sma3_pctlrange0-2', 'loudness_sma3_meanRisingSlope', 'loudness_sma3_stddevRisingSlope', 'loudness_sma3_meanFallingSlope', 'loudness_sma3_stddevFallingSlope', 'spectralFlux_sma3_amean', 'spectralFlux_sma3_stddevNorm', 'mfcc1_sma3_amean', 'mfcc1_sma3_stddevNorm', 'mfcc2_sma3_amean', 'mfcc2_sma3_stddevNorm', 'mfcc3_sma3_amean', 'mfcc3_sma3_stddevNorm', 'mfcc4_sma3_amean', 'mfcc4_sma3_stddevNorm', 'jitterLocal_sma3nz_amean', 'jitterLocal_sma3nz_stddevNorm', 'shimmerLocaldB_sma3nz_amean', 'shimmerLocaldB_sma3nz_stddevNorm', 'HNRdBACF_sma3nz_amean', 'HNRdBACF_sma3nz_stddevNorm', 'logRelF0-H1-H2_sma3nz_amean', 'logRelF0-H1-H2_sma3nz_stddevNorm', 'logRelF0-H1-A3_sma3nz_amean', 'logRelF0-H1-A3_sma3nz_stddevNorm', 'F1frequency_sma3nz_amean', 'F1frequency_sma3nz_stddevNorm', 'F1bandwidth_sma3nz_amean', 'F1bandwidth_sma3nz_stddevNorm', 'F1amplitudeLogRelF0_sma3nz_amean', 'F1amplitudeLogRelF0_sma3nz_stddevNorm', 'F2frequency_sma3nz_amean', 'F2frequency_sma3nz_stddevNorm', 'F2bandwidth_sma3nz_amean', 'F2bandwidth_sma3nz_stddevNorm', 'F2amplitudeLogRelF0_sma3nz_amean', 'F2amplitudeLogRelF0_sma3nz_stddevNorm', 'F3frequency_sma3nz_amean', 'F3frequency_sma3nz_stddevNorm', 'F3bandwidth_sma3nz_amean', 'F3bandwidth_sma3nz_stddevNorm', 'F3amplitudeLogRelF0_sma3nz_amean', 'F3amplitudeLogRelF0_sma3nz_stddevNorm', 'alphaRatioV_sma3nz_amean', 'alphaRatioV_sma3nz_stddevNorm', 'hammarbergIndexV_sma3nz_amean', 'hammarbergIndexV_sma3nz_stddevNorm', 'slopeV0-500_sma3nz_amean', 'slopeV0-500_sma3nz_stddevNorm', 'slopeV500-1500_sma3nz_amean', 'slopeV500-1500_sma3nz_stddevNorm', 'spectralFluxV_sma3nz_amean', 'spectralFluxV_sma3nz_stddevNorm', 'mfcc1V_sma3nz_amean', 'mfcc1V_sma3nz_stddevNorm', 'mfcc2V_sma3nz_amean', 'mfcc2V_sma3nz_stddevNorm', 'mfcc3V_sma3nz_amean', 'mfcc3V_sma3nz_stddevNorm', 'mfcc4V_sma3nz_amean', 'mfcc4V_sma3nz_stddevNorm', 'alphaRatioUV_sma3nz_amean', 'hammarbergIndexUV_sma3nz_amean', 'slopeUV0-500_sma3nz_amean', 'slopeUV500-1500_sma3nz_amean', 'spectralFluxUV_sma3nz_amean', 'loudnessPeaksPerSec', 'VoicedSegmentsPerSec', 'MeanVoicedSegmentLengthSec', 'StddevVoicedSegmentLengthSec', 'MeanUnvoicedSegmentLength', 'StddevUnvoicedSegmentLength', 'equivalentSoundLevel_dBp']
    elif algorithm == "auditory-local":
        return ['Avg. Word Dur.', 'Avg. Sil. Dur.', 'Dynamic Range', 'Energy', 'Intensity', 'ZCR', 'Root Mean Square', 'Sound Pressure Level', 'Consonant Vowel Ratio', 'B_CoG', 'B_Kur', 'B_Ske', 'B_Std', 'CH_CoG', 'CH_Kur', 'CH_Ske', 'CH_Std', 'D_CoG', 'D_Kur', 'D_Ske', 'D_Std', 'DH_CoG', 'DH_Kur', 'DH_Ske', 'DH_Std', 'CX_CoG', 'CX_Kur', 'CX_Ske', 'CX_Std', 'EL_CoG', 'EL_Kur', 'EL_Ske', 'EL_Std', 'EM_CoG', 'EM_Kur', 'EM_Ske', 'EM_Std', 'EN_CoG', 'EN_Kur', 'EN_Ske', 'EN_Std', 'F_CoG', 'F_Kur', 'F_Ske', 'F_Std', 'G_CoG', 'G_Kur', 'G_Ske', 'G_Std', 'H_CoG', 'H_Kur', 'H_Ske', 'H_Std', 'JH_CoG', 'JH_Kur', 'JH_Ske', 'JH_Std', 'K_CoG', 'K_Kur', 'K_Ske', 'K_Std', 'L_CoG', 'L_Kur', 'L_Ske', 'L_Std', 'M_CoG', 'M_Kur', 'M_Ske', 'M_Std', 'N_CoG', 'N_Kur', 'N_Ske', 'N_Std', 'NX_CoG', 'NX_Kur', 'NX_Ske', 'NX_Std', 'NG_CoG', 'NG_Kur', 'NG_Ske', 'NG_Std', 'P_CoG', 'P_Kur', 'P_Ske', 'P_Std', 'Q_CoG', 'Q_Kur', 'Q_Ske', 'Q_Std', 'R_CoG', 'R_Kur', 'R_Ske', 'R_Std', 'S_CoG', 'S_Kur', 'S_Ske', 'S_Std', 'SH_CoG', 'SH_Kur', 'SH_Ske', 'SH_Std', 'T_CoG', 'T_Kur', 'T_Ske', 'T_Std', 'TH_CoG', 'TH_Kur', 'TH_Ske', 'TH_Std', 'V_CoG', 'V_Kur', 'V_Ske', 'V_Std', 'W_CoG', 'W_Kur', 'W_Ske', 'W_Std', 'WH_CoG', 'WH_Kur', 'WH_Ske', 'WH_Std', 'Y_CoG', 'Y_Kur', 'Y_Ske', 'Y_Std', 'Z_CoG', 'Z_Kur', 'Z_Ske', 'Z_Std', 'ZH_CoG', 'ZH_Kur', 'ZH_Ske', 'ZH_Std', 'Avg. Cons. Dur.', 'AA-F0_1', 'AA-F0_2', 'AA-F0_3', 'AA-F0_4', 'AA-F0_5', 'AA-F1_1', 'AA-F2_1', 'AA-F3_1', 'AA-F1_2', 'AA-F2_2', 'AA-F3_2', 'AA-F1_3', 'AA-F2_3', 'AA-F3_3', 'AA-F1_4', 'AA-F2_4', 'AA-F3_4', 'AA-F1_5', 'AA-F2_5', 'AA-F3_5', 'AE-F0_1', 'AE-F0_2', 'AE-F0_3', 'AE-F0_4', 'AE-F0_5', 'AE-F1_1', 'AE-F2_1', 'AE-F3_1', 'AE-F1_2', 'AE-F2_2', 'AE-F3_2', 'AE-F1_3', 'AE-F2_3', 'AE-F3_3', 'AE-F1_4', 'AE-F2_4', 'AE-F3_4', 'AE-F1_5', 'AE-F2_5', 'AE-F3_5', 'AH-F0_1', 'AH-F0_2', 'AH-F0_3', 'AH-F0_4', 'AH-F0_5', 'AH-F1_1', 'AH-F2_1', 'AH-F3_1', 'AH-F1_2', 'AH-F2_2', 'AH-F3_2', 'AH-F1_3', 'AH-F2_3', 'AH-F3_3', 'AH-F1_4', 'AH-F2_4', 'AH-F3_4', 'AH-F1_5', 'AH-F2_5', 'AH-F3_5', 'AO-F0_1', 'AO-F0_2', 'AO-F0_3', 'AO-F0_4', 'AO-F0_5', 'AO-F1_1', 'AO-F2_1', 'AO-F3_1', 'AO-F1_2', 'AO-F2_2', 'AO-F3_2', 'AO-F1_3', 'AO-F2_3', 'AO-F3_3', 'AO-F1_4', 'AO-F2_4', 'AO-F3_4', 'AO-F1_5', 'AO-F2_5', 'AO-F3_5', 'AW-F0_1', 'AW-F0_2', 'AW-F0_3', 'AW-F0_4', 'AW-F0_5', 'AW-F1_1', 'AW-F2_1', 'AW-F3_1', 'AW-F1_2', 'AW-F2_2', 'AW-F3_2', 'AW-F1_3', 'AW-F2_3', 'AW-F3_3', 'AW-F1_4', 'AW-F2_4', 'AW-F3_4', 'AW-F1_5', 'AW-F2_5', 'AW-F3_5', 'AX-F0_1', 'AX-F0_2', 'AX-F0_3', 'AX-F0_4', 'AX-F0_5', 'AX-F1_1', 'AX-F2_1', 'AX-F3_1', 'AX-F1_2', 'AX-F2_2', 'AX-F3_2', 'AX-F1_3', 'AX-F2_3', 'AX-F3_3', 'AX-F1_4', 'AX-F2_4', 'AX-F3_4', 'AX-F1_5', 'AX-F2_5', 'AX-F3_5', 'AXR-F0_1', 'AXR-F0_2', 'AXR-F0_3', 'AXR-F0_4', 'AXR-F0_5', 'AXR-F1_1', 'AXR-F2_1', 'AXR-F3_1', 'AXR-F1_2', 'AXR-F2_2', 'AXR-F3_2', 'AXR-F1_3', 'AXR-F2_3', 'AXR-F3_3', 'AXR-F1_4', 'AXR-F2_4', 'AXR-F3_4', 'AXR-F1_5', 'AXR-F2_5', 'AXR-F3_5', 'AY-F0_1', 'AY-F0_2', 'AY-F0_3', 'AY-F0_4', 'AY-F0_5', 'AY-F1_1', 'AY-F2_1', 'AY-F3_1', 'AY-F1_2', 'AY-F2_2', 'AY-F3_2', 'AY-F1_3', 'AY-F2_3', 'AY-F3_3', 'AY-F1_4', 'AY-F2_4', 'AY-F3_4', 'AY-F1_5', 'AY-F2_5', 'AY-F3_5', 'EH-F0_1', 'EH-F0_2', 'EH-F0_3', 'EH-F0_4', 'EH-F0_5', 'EH-F1_1', 'EH-F2_1', 'EH-F3_1', 'EH-F1_2', 'EH-F2_2', 'EH-F3_2', 'EH-F1_3', 'EH-F2_3', 'EH-F3_3', 'EH-F1_4', 'EH-F2_4', 'EH-F3_4', 'EH-F1_5', 'EH-F2_5', 'EH-F3_5', 'ER-F0_1', 'ER-F0_2', 'ER-F0_3', 'ER-F0_4', 'ER-F0_5', 'ER-F1_1', 'ER-F2_1', 'ER-F3_1', 'ER-F1_2', 'ER-F2_2', 'ER-F3_2', 'ER-F1_3', 'ER-F2_3', 'ER-F3_3', 'ER-F1_4', 'ER-F2_4', 'ER-F3_4', 'ER-F1_5', 'ER-F2_5', 'ER-F3_5', 'EY-F0_1', 'EY-F0_2', 'EY-F0_3', 'EY-F0_4', 'EY-F0_5', 'EY-F1_1', 'EY-F2_1', 'EY-F3_1', 'EY-F1_2', 'EY-F2_2', 'EY-F3_2', 'EY-F1_3', 'EY-F2_3', 'EY-F3_3', 'EY-F1_4', 'EY-F2_4', 'EY-F3_4', 'EY-F1_5', 'EY-F2_5', 'EY-F3_5', 'IH-F0_1', 'IH-F0_2', 'IH-F0_3', 'IH-F0_4', 'IH-F0_5', 'IH-F1_1', 'IH-F2_1', 'IH-F3_1', 'IH-F1_2', 'IH-F2_2', 'IH-F3_2', 'IH-F1_3', 'IH-F2_3', 'IH-F3_3', 'IH-F1_4', 'IH-F2_4', 'IH-F3_4', 'IH-F1_5', 'IH-F2_5', 'IH-F3_5', 'IX-F0_1', 'IX-F0_2', 'IX-F0_3', 'IX-F0_4', 'IX-F0_5', 'IX-F1_1', 'IX-F2_1', 'IX-F3_1', 'IX-F1_2', 'IX-F2_2', 'IX-F3_2', 'IX-F1_3', 'IX-F2_3', 'IX-F3_3', 'IX-F1_4', 'IX-F2_4', 'IX-F3_4', 'IX-F1_5', 'IX-F2_5', 'IX-F3_5', 'IY-F0_1', 'IY-F0_2', 'IY-F0_3', 'IY-F0_4', 'IY-F0_5', 'IY-F1_1', 'IY-F2_1', 'IY-F3_1', 'IY-F1_2', 'IY-F2_2', 'IY-F3_2', 'IY-F1_3', 'IY-F2_3', 'IY-F3_3', 'IY-F1_4', 'IY-F2_4', 'IY-F3_4', 'IY-F1_5', 'IY-F2_5', 'IY-F3_5', 'OW-F0_1', 'OW-F0_2', 'OW-F0_3', 'OW-F0_4', 'OW-F0_5', 'OW-F1_1', 'OW-F2_1', 'OW-F3_1', 'OW-F1_2', 'OW-F2_2', 'OW-F3_2', 'OW-F1_3', 'OW-F2_3', 'OW-F3_3', 'OW-F1_4', 'OW-F2_4', 'OW-F3_4', 'OW-F1_5', 'OW-F2_5', 'OW-F3_5', 'OY-F0_1', 'OY-F0_2', 'OY-F0_3', 'OY-F0_4', 'OY-F0_5', 'OY-F1_1', 'OY-F2_1', 'OY-F3_1', 'OY-F1_2', 'OY-F2_2', 'OY-F3_2', 'OY-F1_3', 'OY-F2_3', 'OY-F3_3', 'OY-F1_4', 'OY-F2_4', 'OY-F3_4', 'OY-F1_5', 'OY-F2_5', 'OY-F3_5', 'UH-F0_1', 'UH-F0_2', 'UH-F0_3', 'UH-F0_4', 'UH-F0_5', 'UH-F1_1', 'UH-F2_1', 'UH-F3_1', 'UH-F1_2', 'UH-F2_2', 'UH-F3_2', 'UH-F1_3', 'UH-F2_3', 'UH-F3_3', 'UH-F1_4', 'UH-F2_4', 'UH-F3_4', 'UH-F1_5', 'UH-F2_5', 'UH-F3_5', 'UW-F0_1', 'UW-F0_2', 'UW-F0_3', 'UW-F0_4', 'UW-F0_5', 'UW-F1_1', 'UW-F2_1', 'UW-F3_1', 'UW-F1_2', 'UW-F2_2', 'UW-F3_2', 'UW-F1_3', 'UW-F2_3', 'UW-F3_3', 'UW-F1_4', 'UW-F2_4', 'UW-F3_4', 'UW-F1_5', 'UW-F2_5', 'UW-F3_5', 'UX-F0_1', 'UX-F0_2', 'UX-F0_3', 'UX-F0_4', 'UX-F0_5', 'UX-F1_1', 'UX-F2_1', 'UX-F3_1', 'UX-F1_2', 'UX-F2_2', 'UX-F3_2', 'UX-F1_3', 'UX-F2_3', 'UX-F3_3', 'UX-F1_4', 'UX-F2_4', 'UX-F3_4', 'UX-F1_5', 'UX-F2_5', 'UX-F3_5', 'Avg. Voca. Dur.']
    elif algorithm == "compare":
        import pickle
        with open('/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Code/PYTHON/openSMILE/compare_feature_names.pkl', 'wb') as f:
            return pickle.load(f)
        
def featContrib(exp_dir, experiment, X, y, X_test, y_test, devSpeakerDict, random_state, byFrame, RUNNUM):

    # Get the baseline results using all features
    module = linearLeastSquares
    baseReg, _ = module.main(X, y, X_test, y_test, devSpeakerDict, byFrame) # These shapes are wrong TODO
    print(baseReg.rmse) # 8.2

    # Get the feature names
    featureNames = getFeatureNames(experiment)

    # Save out the contributions
    contributions = list()
    loo = LeaveOneOut()
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    for train_index, _ in loo.split(X.T):

        X_train, X_test_mini = X.T[train_index].T, X_test.T[train_index].T

        reg, _ = module.main(X_train, y, X_test_mini, y_test, devSpeakerDict, byFrame)
        contributions.append(reg.rmse - baseReg.rmse) # 

    writeContributionReport(contributions, featureNames, exp_dir, experiment, baseline = baseReg.rmse)


def withheldVal(exp_dir, experiment, X, y, X_test, y_test, devSpeakerDict, byFrame, RUNNUM):

    regressors = list()
    for module, name in [(linearLeastSquares, "Linear-Least-Squares"),
                         (ordinaryLeastSquares, "Ordinary-Least-Squares"),
                         (huberRegression, "Huber")
                        ]:

        print(name)
        reg, model = module.main(X, y, X_test, y_test, devSpeakerDict, byFrame)
        regressors.append(reg)

        plt.plot(y_test, reg.pred, 'o')
        m, b = np.polyfit(y_test, reg.pred, 1)
        plt.plot(y_test, m*np.array(y_test) + b)
        savePoint = os.path.join(exp_dir, "reports", "plots", "linearRegressionPlots", "LinearRegressionPlot_{}_{}.png".format(experiment, name))
        plt.xlabel('True Labels')
        plt.ylabel('Predictions')
        plt.savefig(savePoint)
        plt.clf()

        # https://stackoverflow.com/a/11169797/13177304
        filename = "{}/models/regressors/{}-{}-{}.pkl".format(exp_dir, RUNNUM, experiment, name)
        _ = joblib.dump(model, filename, compress=9)

    writeReport(regressors, exp_dir, experiment)


def crossVal(exp_dir, experiment, X, y, X_test, y_test, devSpeakerDict, random_state, byFrame, RUNNUM):

    kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = random_state)

    all_regressors = list()
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    split = 0
    for train_index, _ in kf.split(X, y):

        X_train, y_train = X[train_index], np.array(y)[train_index]

        regressors = list()
        for module, name in [(linearLeastSquares, "Linear-Least-Squares"),
                             (ordinaryLeastSquares, "Ordinary-Least-Squares"),
                             (huberRegression, "Huber")
                            ]:

            print(name)
            reg, model = module.main(X_train, y_train, X_test, y_test, devSpeakerDict, byFrame)
            regressors.append(reg)

            plt.plot(y_test, reg.pred, 'o')
            m, b = np.polyfit(y_test, reg.pred, 1)
            plt.plot(y_test, m*np.array(y_test) + b)
            savePoint = os.path.join(exp_dir, "reports", "plots", "linearRegressionPlots", "LinearRegressionPlot_{}_{}.png".format(experiment, name))
            plt.xlabel('True Labels')
            plt.ylabel('Predictions')
            plt.savefig(savePoint)
            plt.clf()

            # https://stackoverflow.com/a/11169797/13177304
            filename = "{}/models/regressors/{}-{}-{}-{}.pkl".format(exp_dir, RUNNUM, experiment, name, split)
            _ = joblib.dump(model, filename, compress=9)

        all_regressors.append(regressors)
        split += 1

    writeReport(all_regressors, exp_dir, experiment)


def subsetDataFrame(df, windowCount, n=4):

    start = 0
    end = (n * 2) + 1
    myLists = list()
    while end <= windowCount:
        myLists.append(list(range(start, end)))
        start += 1
        end += 1

    for subList in myLists:
        miniDF = df[subList]
        yield miniDF.to_numpy().flatten()
        

def averageDataFrame(df, windowCount, n=4):

    start = 0
    end = (n * 2) + 1
    myLists = list()
    while end <= windowCount:
        myLists.append(list(range(start, end)))
        start += 1
        end += 1

    for subList in myLists:
        miniDF = df[subList]
        averagedColumn = miniDF.mean(axis=1)
        yield averagedColumn.to_numpy()


def readFiles(X, byFrame, experiment):

    labels, utteranceList = X
    y, speakerDict, returnVectors = list(), dict(), list()
    for label, utterance in zip(labels, utteranceList):
        speaker = utterance.split('/')[-1].split('-')[0]
        if speaker not in speakerDict:
            speakerDict[speaker] = [label, list()]
        df = pd.read_csv(utterance, header = None)

        if not byFrame:
            if experiment == 'compare' or experiment == 'gemaps': # OPENSMILE (duration-independent)

                vector = df.to_numpy()
                y.append(label)
                returnVectors.append(vector)
                speakerDict[speaker][1].append(vector)

            else: # SEGMENTAL/AUDITORY stuff

                vector = df.to_numpy().tolist()[0]
                # Don't forget the Intelligibility Metrics!
                for intLevel in [70, 50, 30]:
                    intDF = pd.read_csv(utterance.replace('/csv/', '/csv/{}/'.format(str(intLevel))), header = None)
                    intVec = intDF.to_numpy()
                    intVals = intVec[:,-6:].tolist()[0]
                    vector += intVals

                y.append(label)
                vector = np.array(vector).reshape(1, -1)
                returnVectors.append(vector)
                speakerDict[speaker][1].append(vector)

        else:
            # Consider an input dataframe of shape (375,150) with AMS features
            if experiment == "raw": # Raw frames as input feature fectors (150)
                for _, colSeries in df.items():
                    vector = colSeries.to_numpy()
                    y.append(label)
                    returnVectors.append(vector)
                    speakerDict[speaker][1].append(vector)

            elif experiment == "averaged": # Averaged frames as input feature vectors (142)
                windowCount = df.shape[1]
                if windowCount > 8:
                    vectors = list(averageDataFrame(df, windowCount))
                    for vector in vectors:
                        y.append(label)
                        returnVectors.append(vector)
                        speakerDict[speaker][1].append(vector)

            elif experiment == "flattened": # Flattened matrices 9 x 375 as ifv (142)
                windowCount = df.shape[1]
                if windowCount > 8:
                    vectors = list(subsetDataFrame(df, windowCount))
                    for vector in vectors:
                        y.append(label)
                        returnVectors.append(vector)
                        speakerDict[speaker][1].append(vector)

            elif experiment == "averaged_and_flattened": # Average frames and then flattened 9 x 375 ?? Maybe this fails miserably (134)
                windowCount = df.shape[1]
                if windowCount > 8:
                    vectors = list(averageDataFrame(df, windowCount))
                    miniDF = pd.DataFrame(vectors).T
                    miniWindowCount = miniDF.shape[1]
                    if miniWindowCount > 8:
                        vectors = list(subsetDataFrame(miniDF, miniWindowCount))
                        for vector in vectors:
                            y.append(label)
                            returnVectors.append(vector)
                            speakerDict[speaker][1].append(vector)
        
    return y, np.vstack(returnVectors), speakerDict


def getLabels(exp, data, which, loocv_not_withheld):

    y, files = list(), list()

    allUtterances = glob.glob(os.path.join(exp, 'data', which, 'csv/*.csv'))
    if loocv_not_withheld == 'True': # Cross-validation from training only
        metaDF = pd.concat([pd.read_csv(os.path.join(data, '{}_meta_data.txt'.format(condition)), sep = ';').rename(columns=lambda x: x.strip()) for condition in ['cc', 'cd']])
    else: # Use the withheld test set
        if which == "train":
            metaDF = pd.concat([pd.read_csv(os.path.join(data, '{}_meta_data.txt'.format(condition)), sep = ';').rename(columns=lambda x: x.strip()) for condition in ['cc', 'cd']])
        else:
            metaDF = pd.concat([pd.read_csv(os.path.join(data, '../meta_data_test.txt'.format(condition)), sep = ';').rename(columns=lambda x: x.strip()) for condition in ['cc', 'cd']])
    metaDF['ID'] = metaDF['ID'].str.strip()
    metaDF = metaDF.drop_duplicates(subset = 'ID')
    metaDF.set_index('ID', inplace = True)

    for utterance in allUtterances:
        id = os.path.basename(utterance).split('.')[0].split('_')[0].split('-')[0]
        row = metaDF.loc[id]
        score = row['mmse']
        # We may have NaN values for the scores, so make sure it's an int
        if isinstance(score, np.int) or isinstance(score, np.int64):
            y.append(score)
            files.append(utterance)
        elif isinstance(score, str):
            actualScore = score.strip()
            if actualScore != 'NA':
                y.append(int(score))
                files.append(utterance)
            else:
                #The score is NaN; skip it
                pass

    return y, files


def getData(exp, data, byFrame, experiment, loocv_not_withheld):

    # Vectorizing the inputs takes forever, so let's save these out in case we have to do it again
    if os.path.isfile('{}/vectors/regressors/{}-devSpeakerDict.pkl'.format(exp, experiment)):
        y_train = joblib.load('{}/vectors/regressors/{}-y_train.pkl'.format(exp, experiment))
        X_train = joblib.load('{}/vectors/regressors/{}-X_train.pkl'.format(exp, experiment))
        trainSpeakerDict = joblib.load('{}/vectors/regressors/{}-trainSpeakerDict.pkl'.format(exp, experiment))

        y_test = joblib.load('{}/vectors/regressors/{}-y_test.pkl'.format(exp, experiment))
        X_test = joblib.load('{}/vectors/regressors/{}-X_test.pkl'.format(exp, experiment))
        devSpeakerDict = joblib.load('{}/vectors/regressors/{}-devSpeakerDict.pkl'.format(exp, experiment))

        scaler = joblib.load('{}/vectors/regressors/{}-scaler.pkl'.format(exp, experiment))

    else:

        y_train, X_train, trainSpeakerDict = readFiles(getLabels(exp, data, 'train', loocv_not_withheld), byFrame, experiment)
        y_test, X_test, devSpeakerDict = readFiles(getLabels(exp, data, 'dev', loocv_not_withheld), byFrame, experiment)

        # Separate out the control condition from the test condition
        # https://www.alz.org/alzheimers-dementia/diagnosis/medical_tests
            # tl;dr: < 25 indicates mild dementia; < 21 indicates moderate dementia; < 13 indicates severe dementia
        controlLabelsTrain = [True if i >= 25 else False for i in y_train]
        controlLabelsTest = [True if i >= 25 else False for i in y_test]
        controlTrainSubset = X_train[controlLabelsTrain]
        controlTestSubset = X_test[controlLabelsTest]

        diagnosedLabelsTrain = [True if i < 25 else False for i in y_train]
        diagnosedLabelsTest = [True if i < 25 else False for i in y_test]
        diagnosedTrainSubset = X_train[diagnosedLabelsTrain]
        diagnosedTestSubset = X_test[diagnosedLabelsTest]

        # Scale the column-wise variance of the control condition, applying that to the train and test data
        # This stands in contrast to how I was doing it before, by just scaling the column-wise variance of the 
        # train and test data all in one go. Because we assume that the variance of the test condition (diagnosed)
        # will be greater than the control condition, this new way won't flatten out that test condition variance and 
        # we should see more obvious differences. 
        scaler = StandardScaler()
        scaler.fit(controlTrainSubset)
        controlTrainScaled = scaler.transform(controlTrainSubset)
        controlTestScaled = scaler.transform(controlTestSubset)
        diagnosedTrainScaled = scaler.transform(diagnosedTrainSubset)
        diagnosedTestScaled = scaler.transform(diagnosedTestSubset)

        for speaker in trainSpeakerDict:
            _, vectorsList = trainSpeakerDict[speaker]
            trainSpeakerDict[speaker][1] = scaler.transform(np.vstack(vectorsList))

        for speaker in devSpeakerDict:
            _, vectorsList = devSpeakerDict[speaker]
            devSpeakerDict[speaker][1] = scaler.transform(np.vstack(vectorsList))

        # If there are missing values, impute them
        # We are also now doing this by creating a separate imputer for the control and test conditions
        # because we shouldn't be using the same mean to complete missing values for the diagnosed participants
        # as what we are using for the typically-aging participants.
        if np.isnan(controlTrainScaled).any() or np.isnan(diagnosedTrainScaled).any() or np.isnan(controlTestScaled).any() or np.isnan(diagnosedTestScaled).any():

            # We need to impute the values based on the control/diagnosed division
            # NOTE this is actually an area where we can do a lot of work messing around with things. 
                # Stuff I've thought to try: 
                # Try to first impute values by speaker so that if they have a (few) realization(s) of /e/
                    # in (one) utterance(s) then I can impute their values. But if they just never used a particular
                    # segment, then I impute the value based on the group's
                # Try the iterativeimputer
                # Try the KNN imputer (probably makes more sense than the iterative imputer because it's column-wise)

            # Create the control and diagnosed imputers
            conImputer = SimpleImputer(missing_values = np.nan, strategy='mean', add_indicator = True)
            diaImputer = SimpleImputer(missing_values = np.nan, strategy='mean', add_indicator = True)

            # Fit the imputers to the training data
            conImputer.fit(controlTrainScaled)
            diaImputer.fit(diagnosedTrainScaled)

            # Transform the train and test data for the control condition
            controlTrainScaledImputed = conImputer.transform(controlTrainScaled)[:,:controlTrainScaled.shape[1]]
            diagnosedTrainScaledImputed = diaImputer.transform(diagnosedTrainScaled)[:,:diagnosedTrainScaled.shape[1]]
            # controlTestScaledImputed = conImputer.transform(controlTestScaled) # This is cheating

            # Create and fit a impNEW imputer based on ALL of the training data
            X_train_scaled_and_imputed = np.vstack((controlTrainScaledImputed, diagnosedTrainScaledImputed))
            y_train_recreated = np.array(y_train)[controlLabelsTrain].tolist() + np.array(y_train)[diagnosedLabelsTrain].tolist()
            trainImputer = SimpleImputer(missing_values = np.nan, strategy='mean').fit(X_train_scaled_and_imputed)

            # It may be the case that there are values in the training data which never arise in the testing data
            # In this case, we need to leave the columns consistent across train and dev frames; set things to 0 instead
            # of np.nan so that the column doesn't disappear.
            transposedControlTestScaled = controlTestScaled.T
            cols = controlTestScaled.shape[1]
            for col in range(cols):
                if np.isnan(transposedControlTestScaled[col]).all():
                    transposedControlTestScaled[col] = np.zeros(transposedControlTestScaled[col].shape)
            controlTestScaled = transposedControlTestScaled.T

            transposedDiagnosedTestScaled = diagnosedTestScaled.T
            cols = diagnosedTestScaled.shape[1]
            for col in range(cols):
                if np.isnan(transposedDiagnosedTestScaled[col]).all():
                    transposedDiagnosedTestScaled[col] = np.zeros(transposedDiagnosedTestScaled[col].shape)
            diagnosedTestScaled = transposedDiagnosedTestScaled.T

            # Impute all missing values from the test data using the imputer fit to all the training data
            controlTestScaledImputed = trainImputer.transform(controlTestScaled)
            diagnosedTestScaledImputed = trainImputer.transform(diagnosedTestScaled)

            X_test_scaled_and_imputed = np.vstack((controlTestScaledImputed, diagnosedTestScaledImputed))
            y_test_recreated = np.array(y_test)[controlLabelsTest].tolist() + np.array(y_test)[diagnosedLabelsTest].tolist()

            # Shuffle all of the data points and the labels with them by random permutation
            idxTrain = np.random.permutation(X_train.shape[0])
            X_train, y_train = X_train_scaled_and_imputed[idxTrain], np.array(y_train_recreated)[idxTrain].tolist()
            idxTest = np.random.permutation(X_test.shape[0])
            X_test, y_test = X_test_scaled_and_imputed[idxTest], np.array(y_test_recreated)[idxTest].tolist()

            for speaker in trainSpeakerDict:
                mmse, vectorsList = trainSpeakerDict[speaker]
                if mmse >= 25:
                    trainSpeakerDict[speaker][1] = conImputer.transform(np.vstack(vectorsList))[:,:controlTrainScaled.shape[1]]
                else:
                    trainSpeakerDict[speaker][1] = diaImputer.transform(np.vstack(vectorsList))[:,:diagnosedTrainScaled.shape[1]]

            for speaker in devSpeakerDict:
                devSpeakerDict[speaker][1] = trainImputer.transform(np.array(devSpeakerDict[speaker][1]))

        else:
            
            for speaker in devSpeakerDict:
                devSpeakerDict[speaker][1] = np.array(devSpeakerDict[speaker][1])

        z = joblib.dump(y_train, '{}/vectors/regressors/{}-y_train.pkl'.format(exp, experiment), compress=9)
        z = joblib.dump(X_train, '{}/vectors/regressors/{}-X_train.pkl'.format(exp, experiment), compress=9)
        z = joblib.dump(trainSpeakerDict, '{}/vectors/regressors/{}-trainSpeakerDict.pkl'.format(exp, experiment), compress=9)

        z = joblib.dump(y_test, '{}/vectors/regressors/{}-y_test.pkl'.format(exp, experiment), compress=9)
        z = joblib.dump(X_test, '{}/vectors/regressors/{}-X_test.pkl'.format(exp, experiment), compress=9)
        z = joblib.dump(devSpeakerDict, '{}/vectors/regressors/{}-devSpeakerDict.pkl'.format(exp, experiment), compress=9)

        z = joblib.dump(scaler, '{}/vectors/regressors/{}-scaler.pkl'.format(exp, experiment), compress=9)

    # print('Objects dumped for regression')
    # return None, None, None, None, None # Useful for when I actually don't want to run the ML models, just get the data in the right shape. 

    if byFrame:
        kf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 1)
        trainSplitter = list(kf.split(X_train, y_train))
        train_index = trainSplitter[0][1]
        testSplitter = list(kf.split(X_test, y_test))
        test_index = testSplitter[0][1]

        X_train, y_train = X_train[train_index], list(np.array(y_train)[train_index])
        X_test, y_test = X_test[test_index], list(np.array(y_test)[test_index])

    return X_train, y_train, X_test, y_test, devSpeakerDict


def makeCalls(exp_dir, data_dir, random_state, byFrame, RUNNUM, experiment, loocv_not_withheld, feature_contribution):

    X, y, X_test, y_test, devSpeakerDict = getData(exp_dir, data_dir, byFrame, experiment, loocv_not_withheld)
    # if X is None and y is None and X_test is None and y_test is None and devSpeakerDict is None:
    #     return

    if loocv_not_withheld:
        crossVal(exp_dir, experiment, X, y, X_test, y_test, devSpeakerDict, random_state, byFrame, RUNNUM)
    else:
        withheldVal(exp_dir, experiment, X, y, X_test, y_test, devSpeakerDict, byFrame, RUNNUM)

    # if feature_contribution:
    #     featContrib(exp_dir, experiment, X, y, X_test, y_test, devSpeakerDict, random_state, byFrame, RUNNUM)


def main():

    parser = argparse.ArgumentParser(description='Call all of the regression algorithms and consolidate a global report.')
    parser.add_argument('exp_dir', nargs='?', type=str, help='Temporary experiment directory.', default='/tmp/tmp.OgrVUR6iE4')
    parser.add_argument('data_dir', nargs='?', type=str, help='Location for the map for the scalar score (e.g. MMSE).', default='../../Data/ADReSS-IS2020-data/train')
    parser.add_argument('random_state', nargs='?', type=int, help='Affects the ordering of the indices, which controls the randomness of each fold in KFold validation.', default=1)
    parser.add_argument('by_frame', nargs='?', type=str, help='True if we need to run this by frame or False if we are using COMPARE or something else distilled.', default="False")
    parser.add_argument('run_num', nargs='?', type=str, help='Which runthrough we are on.', default='14102')
    parser.add_argument('loocv_not_withheld', nargs='?', type=str, help='If True, we will do 5 fold leave-one-out cross validation; if False, we are training on all the training data and testing on the withheld test data ', default='False')
    parser.add_argument('algorithm', nargs='?', type=str, help='Which input feature type we are using', default='auditory-local')
    parser.add_argument('feature_contribution', nargs='?', type=str, help='Whether to use a leave one out approach for calculating feature contribution', default='False')

    args = parser.parse_args()
    
    if args.by_frame == "True":
        for experiment in ['raw', 'averaged', 'flattened']: #, 'averaged_and_flattened']:
            print('Now working on {}'.format(experiment))
            makeCalls(args.exp_dir, args.data_dir, args.random_state, True, args.run_num, experiment, args.loocv_not_withheld, args.feature_contribution)
    else:
        makeCalls(args.exp_dir, args.data_dir, args.random_state, False, args.run_num, args.algorithm, args.loocv_not_withheld, args.feature_contribution)


if __name__ == "__main__":

    main()
