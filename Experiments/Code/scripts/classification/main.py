#!/usr/bin/env python3

import os
import sys
import glob
import joblib
import argparse
import numpy as np
import pandas as pd
import seaborn as sns

import gaussianNB
import decisionTree 
import randomForest
import nearestNeighbors
import linearSupportVector
import linearDiscriminantAnalysis

from tqdm import tqdm
from matplotlib import pyplot as plt
from consolidator import Consolidator
from sklearn.impute import SimpleImputer
from cf_matrix import make_confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold


def writeReport(classifiers, exp_dir, experiment):

    # Part of cross-validation; sue me
    groups = [list(row) for row in zip(*reversed(classifiers))]

    scoreList = list()
    for group in groups:
        labels, scores = Consolidator(group)._get_means()
        scoreList.append(scores)

    df = pd.DataFrame(scoreList, columns = labels)
    df.to_csv(os.path.join(exp_dir, 'reports', '{}-classifiers.csv'.format(experiment)), index = False)

    print(df)
    print()


def latexConfusionMatrixFormatting(exp_dir, experiment, confusionMatrix, name):

    # str1 = """\\noindent\n\\renewcommand\\arraystretch{1.5}\n\setlength\\tabcolsep{0pt}\n\\begin{tabular}{c >{\\bfseries}r @{\hspace{0.7em}}c @{\hspace{0.4em}}c @{\hspace{0.7em}}l}\n\multirow{10}{*}{\\rotatebox{90}{\parbox{1.1cm}{\\bfseries\centering actual value}}} &\n& \multicolumn{2}{c}{\\bfseries Prediction outcome} & \\\\\n& & \\bfseries p & \\bfseries n & \\bfseries total \\\\\n& p$'$ & \MyBox{"""
    # str2 = str(confusionMatrix[0][0])
    # str3 = """} & \MyBox{"""
    # str4 = str(confusionMatrix[0][1])
    # str5 = """}& P$'$ \\\\[2.4em]\n& n$'$ & \MyBox{"""
    # str6 = str(confusionMatrix[1][0]) 
    # str7 = """}& \MyBox{"""
    # str8 = str(confusionMatrix[1][1])
    # str9 = """}& N$'$ \\\\\n  & total & P & N &\n\end{tabular}"""

    # myStr = ''.join([str1, str2, str3, str4, str5, str6, str7, str8, str9])

    # with open(os.path.join(exp_dir, "reports", "plots", "confusionMatrices", "ConfusionMatrix_{}.lat".format(name)), 'w') as f:
    #     f.write(myStr)

    labels = ['True Pos', 'False Neg', 'False Pos', 'True Neg']
    categories = ['cc', 'cd']
    savePoint = os.path.join(exp_dir, "reports", "plots", "confusionMatrices", "ConfusionMatrix_{}_{}.png".format(experiment, name))
    make_confusion_matrix(np.flip(confusionMatrix), group_names = labels, categories = categories, cmap = 'Blues', savePoint = savePoint)


def withheldVal(exp_dir, experiment, X, y, X_test, y_test, devSpeakerDict, le, RUNNUM):

    classifiers = list()
    for module, name in [
        (nearestNeighbors, "Nearest-Neighbors"),
        # (linearSupportVector, "Linear-Support-Vector"),
        (decisionTree, "Decision-Tree"), #
        (randomForest, "Random-Forest"),
        (gaussianNB, "Gaussian-Naive-Bayes"), #
        (linearDiscriminantAnalysis, "Linear-Discriminant-Analysis")
        ]:

        print(name)    
        clf, model = module.main(X, y, X_test, y_test, le.classes_, devSpeakerDict)
        classifiers.append(clf)
        latexConfusionMatrixFormatting(exp_dir, experiment, clf.confusionMatrix, name)

        # https://stackoverflow.com/a/11169797/13177304
        filename = "{}/models/classifiers/{}-{}-{}.pkl".format(exp_dir, RUNNUM, experiment, name)
        _ = joblib.dump(model, filename, compress=9)

    writeReport(classifiers, exp_dir, experiment)


def crossVal(exp_dir, experiment, X, y, X_test, y_test, devSpeakerDict, random_state, le, RUNNUM):

    kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = random_state)
    
    all_classifiers = list()
    split = 0
    for train_index, test_index in kf.split(X, y):

        X_train, y_train = X[train_index], y[train_index]

        classifiers = list()
        for module, name in [
            (nearestNeighbors, "Nearest-Neighbors"),
            # (linearSupportVector, "Linear-Support-Vector"),
            (decisionTree, "Decision-Tree"), #
            (randomForest, "Random-Forest"),
            (gaussianNB, "Gaussian-Naive-Bayes"), #
            (linearDiscriminantAnalysis, "Linear-Discriminant-Analysis")
            ]:

            print(name)    
            clf, model = module.main(X_train, y_train, X_test, y_test, le.classes_, devSpeakerDict)
            classifiers.append(clf)
            latexConfusionMatrixFormatting(exp_dir, experiment, clf.confusionMatrix, name)

            # https://stackoverflow.com/a/11169797/13177304
            filename = "{}/models/classifiers/{}-{}-{}-{}.pkl".format(exp_dir, RUNNUM, experiment, name, split)
            _ = joblib.dump(model, filename, compress=9)

        all_classifiers.append(classifiers)
        split += 1

    writeReport(all_classifiers, exp_dir, experiment)


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
    # counter = 0
    for label, utterance in zip(labels, utteranceList):
        # counter += 1
        # if counter > 10:
        #     break
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


def getLabels(exp, data, which):

    y, files = list(), list()
    allUtterances = glob.glob(os.path.join(exp, 'data', which, 'csv/*.csv'))

    for utterance in allUtterances:
        # I wrote this blind without testing, so check it if things get weird
        label = os.path.basename(utterance).split('.')[0].split('_')[1]
        y.append(label)
        files.append(utterance)

    return y, files


def getData(exp, data, byFrame, experiment):

    # Vectorizing the inputs takes forever, so let's save these out in case we have to do it again
    if os.path.isfile('{}/vectors/classifiers/{}-devSpeakerDict.pkl'.format(exp, experiment)):
        y_train = joblib.load('{}/vectors/classifiers/{}-y_train.pkl'.format(exp, experiment))
        X_train = joblib.load('{}/vectors/classifiers/{}-X_train.pkl'.format(exp, experiment))
        trainSpeakerDict = joblib.load('{}/vectors/classifiers/{}-trainSpeakerDict.pkl'.format(exp, experiment))

        y_test = joblib.load('{}/vectors/classifiers/{}-y_test.pkl'.format(exp, experiment))
        X_test = joblib.load('{}/vectors/classifiers/{}-X_test.pkl'.format(exp, experiment))
        devSpeakerDict = joblib.load('{}/vectors/classifiers/{}-devSpeakerDict.pkl'.format(exp, experiment))

        scaler = joblib.load('{}/vectors/classifiers/{}-scaler.pkl'.format(exp, experiment))

    else:

        y_train, X_train, trainSpeakerDict = readFiles(getLabels(exp, data, 'train'), byFrame, experiment)
        y_test, X_test, devSpeakerDict = readFiles(getLabels(exp, data, 'dev'), byFrame, experiment)

        # Separate out the control condition from the test condition
        controlLabelsTrain = [True if i == 'cc' else False for i in y_train]
        controlLabelsTest = [True if i == 'cc' else False for i in y_test]
        controlTrainSubset = X_train[controlLabelsTrain]
        controlTestSubset = X_test[controlLabelsTest]

        diagnosedLabelsTrain = [True if i == 'cd' else False for i in y_train]
        diagnosedLabelsTest = [True if i == 'cd' else False for i in y_test]
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
            from sklearn.impute import SimpleImputer # Sometimes this doesn't load right? idk
            conImputer = SimpleImputer(missing_values = np.nan, strategy='mean', add_indicator = True)
            diaImputer = SimpleImputer(missing_values = np.nan, strategy='mean', add_indicator = True)

            # Fit the imputers to the training data
            conImputer.fit(controlTrainScaled)
            diaImputer.fit(diagnosedTrainScaled)

            # Transform the train and test data for the control condition
            controlTrainScaledImputed = conImputer.transform(controlTrainScaled)[:,:controlTrainScaled.shape[1]]
            diagnosedTrainScaledImputed = diaImputer.transform(diagnosedTrainScaled)[:,:diagnosedTrainScaled.shape[1]]
            # controlTestScaledImputed = conImputer.transform(controlTestScaled) # This is cheating

            # Create and fit a NEW imputer based on ALL of the training data
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
                label, vectorsList = trainSpeakerDict[speaker]
                if label == 'cc':
                    trainSpeakerDict[speaker][1] = conImputer.transform(np.vstack(vectorsList))[:,:controlTrainScaled.shape[1]]
                else:
                    trainSpeakerDict[speaker][1] = diaImputer.transform(np.vstack(vectorsList))[:,:diagnosedTrainScaled.shape[1]]

            for speaker in devSpeakerDict:
                devSpeakerDict[speaker][1] = trainImputer.transform(np.array(devSpeakerDict[speaker][1]))

        else:
            
            for speaker in devSpeakerDict:
                devSpeakerDict[speaker][1] = np.array(devSpeakerDict[speaker][1])

        z = joblib.dump(y_train, '{}/vectors/classifiers/{}-y_train.pkl'.format(exp, experiment), compress=9)
        z = joblib.dump(X_train, '{}/vectors/classifiers/{}-X_train.pkl'.format(exp, experiment), compress=9)
        z = joblib.dump(trainSpeakerDict, '{}/vectors/classifiers/{}-trainSpeakerDict.pkl'.format(exp, experiment), compress=9)

        z = joblib.dump(y_test, '{}/vectors/classifiers/{}-y_test.pkl'.format(exp, experiment), compress=9)
        z = joblib.dump(X_test, '{}/vectors/classifiers/{}-X_test.pkl'.format(exp, experiment), compress=9)
        z = joblib.dump(devSpeakerDict, '{}/vectors/classifiers/{}-devSpeakerDict.pkl'.format(exp, experiment), compress=9)

        z = joblib.dump(scaler, '{}/vectors/classifiers/{}-scaler.pkl'.format(exp, experiment), compress=9)

    # Encode from categorical to numerical
    le = LabelEncoder()
    le.fit(y_train)

    # print('Objects dumped for classification')
    # return None, None, None, None, None, None

    # The frame-level features take SO LONG to train and test, so we're going to quantize them down to 10% and see if that helps much. 
    if byFrame:
        kf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 1)
        trainSplitter = list(kf.split(X_train, y_train))
        train_index = trainSplitter[0][1]
        testSplitter = list(kf.split(X_test, y_test))
        test_index = testSplitter[0][1]

        X_train, y_train = X_train[train_index], list(np.array(y_train)[train_index])
        X_test, y_test = X_test[test_index], list(np.array(y_test)[test_index])

    return X_train, le.transform(y_train), X_test, le.transform(y_test), le, devSpeakerDict


def makeCalls(exp_dir, data_dir, random_state, byFrame, RUNNUM, experiment, loocv_not_withheld):

    X, y, X_test, y_test, le, devSpeakerDict = getData(exp_dir, data_dir, byFrame, experiment)
    # if not X and not y and not X_test and not y_test and not le and not devSpeakerDict:
    #     return

    if loocv_not_withheld:
        crossVal(exp_dir, experiment, X, y, X_test, y_test, devSpeakerDict, random_state, le, RUNNUM)
    else:
        withheldVal(exp_dir, experiment, X, y, X_test, y_test, devSpeakerDict, le, RUNNUM)


def main():

    parser = argparse.ArgumentParser(description='Call all of the classification algorithms and consolidate a global report.')
    parser.add_argument('exp_dir', nargs='?', type=str, help='Temporary experiment directory.', default='/tmp/tmp.2CKHTckEZ3')
    parser.add_argument('data_dir', nargs='?', type=str, help='Location for the map for the scalar score (e.g. MMSE).', default='../../Data/ADReSS-IS2020-data/train')
    parser.add_argument('random_state', nargs='?', type=int, help='Affects the ordering of the indices, which controls the randomness of each fold in KFold validation', default=1)
    parser.add_argument('by_frame', nargs='?', type=str, help='True if we need to run this by frame or False if we are using COMPARE or something else distilled.', default="True")
    parser.add_argument('run_num', nargs='?', type=str, help='Which runthrough we are on.', default='3104')
    parser.add_argument('loocv_not_withheld', nargs='?', type=str, help='If True, we will do 5 fold leave-one-out cross validation; if False, we are training on all the training data and testing on the withheld test data ', default='False')
    parser.add_argument('algorithm', nargs='?', type=str, help='Which input feature type we are using', default='ams')

    args = parser.parse_args()
    if args.by_frame == "True":
        for experiment in ['raw', 'averaged', 'flattened']: # 'averaged_and_flattened']:
            print('Now working on {}'.format(experiment))
            makeCalls(args.exp_dir, args.data_dir, args.random_state, True, args.run_num, experiment, args.loocv_not_withheld)
    else:
        makeCalls(args.exp_dir, args.data_dir, args.random_state, False, args.run_num, args.algorithm, args.loocv_not_withheld)


if __name__ == "__main__":

    main()

