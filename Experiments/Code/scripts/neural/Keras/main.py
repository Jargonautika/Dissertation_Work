#!/usr/bin/env python3

# Make it reproducible
from numpy.lib.type_check import nan_to_num
from numpy.random import seed
seed(42)
import tensorflow as tf
tf.random.set_seed(42)

import os
import sys
import pickle
import joblib
import argparse
import numpy as np

from keras import backend as K
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import MissingIndicator
from feedForwardSequential import feedForward
from sklearn.preprocessing import LabelEncoder


def plotIt(model, exp_dir, experiment, algorithm, title = ""):

    fig, axs = plt.subplots(2)
    fig.suptitle(title + ', {}-{}'.format(algorithm, experiment))
    axs[0].plot(model.history.history['binary_accuracy'], 'r-', label = 'binary accuracy')
    axs[0].plot(model.history.history['loss'], 'b-', label = 'loss')
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    axs[1].plot(model.history.history['val_binary_accuracy'], 'r-', label = 'validation binary accuracy') # TODO this is plotting incorrectly
    axs[1].plot(model.history.history['val_loss'], 'b-', label = 'validation loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(os.path.join(exp_dir, 'reports', 'Feed_Forward_NN_{}-{}.pdf'.format(algorithm, experiment)), bbox_inches='tight')
    print(os.path.join(exp_dir, 'reports', 'Feed_Forward_NN_{}-{}.pdf'.format(algorithm, experiment)))

    print()
    print("#########################################################################################")
    print("Final Epoch's Training Accuracy:\t{}".format(model.history.history['binary_accuracy'][-1]))
    print("Final Epoch's Training Loss:\t{}".format(model.history.history['loss'][-1]))
    print("Final Epoch's Testing Accuracy:\t{}".format(model.history.history['val_binary_accuracy'][-1]))
    print("Final Epoch's Testing Loss:\t{}".format(model.history.history['val_loss'][-1]))
    print("Number of Epochs Survived:\t{}".format(len(model.history.history['loss'])))
    print("#########################################################################################")
    print()


def featureContribution(X, y, X_test, y_test, featureNames, comparison):

    # TODO This current approach needs help. 
    # It's looking row-wise and not column-wise; better transpose it. 
    groups, groupNames = list(), [(0, 'wordDur'), (1, 'silDur'), (2, 'CVR'), (3, 'specMoments'), (4, 'consDur'), (5, 'F0'), (6, 'Formants'), (7, 'vocaDur'), (8, 'other')]
    for f in featureNames:
        if f == 'Avg. Word Dur.':
            groups.append(0)
        elif f == 'Avg. Sil. Dur.':
            groups.append(1)
        elif f == 'Consonant Vowel Ratio':
            groups.append(2)
        elif 'CoG' in f or 'Kur' in f or 'Ske' in f or 'Std' in f:
            groups.append(3)
        elif f == 'Avg. Cons. Dur.':
            groups.append(4)
        elif 'F0' in f:
            groups.append(5)
        elif 'F1' in f or 'F2' in f or 'F3' in f:
            groups.append(6)
        elif f == 'Avg. Voca. Dur.':
            groups.append(7)
        else:
            groups.append(8)

    j = 0
    analysisList = list()
    for group, name in groupNames:
        tempX = X.copy()
        tempX_test = X_test.copy()
        tempFeatureNames = featureNames.copy()
        for i, g in enumerate(groups):
            if g == group:
                # Remove the column from the matrices
                tempX = np.delete(tempX, i - j, axis = 1)
                tempX_test = np.delete(tempX_test, i - j, axis = 1)
                # Remove the feature name from the list of names
                del tempFeatureNames[i - j]
                j += 1
        net = baseModel(tempX, y, tempX_test, y_test, 'tmp.ckpt')
        new = max(net.history.history['val_binary_accuracy'][-4:])
        analysisList.append((name, comparison - new))

    print()
    print('******************************')
    for i, j in analysisList:
        print(i, '\t', j)
    print(comparison)
    print('******************************')
    print()


# https://stackoverflow.com/questions/48675487/optimizing-number-of-optimum-features
def gridSearch():

    pass
    

def pcaReduction(X):

    x_axis, y_axis = list(), list()
    for i in range(X.shape[-1]):
        pca = PCA(n_components = i)
        pca.fit(X)
        r = pca.explained_variance_ratio_
        x_axis.append(i)
        y_axis.append(r)

    plt.plot(x_axis, y_axis)
    plt.savefig('auditory_local_pca_reduction.pdf')

# TODO find other feature reduction strategies
    # Look at the QP2 strategies and re-implement here

# TODO especially run pca on the compare feature set to set the variance to ~95%
# and see how far it can be reduced to; if it's like 500 then we have a 500ish feature set to add to it which could potentially
# increase overall performance


def rBest(X, y, X_test, y_test):

    from sklearn.svm import LinearSVC # Compare this with another model to see if it gives you different features
    # TODO try to use the MultilayerPerceptron (in the future)
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestClassifier

    clf = LinearSVC(C=0.01, penalty="l1", dual=False)
    clf.fit(X, y)

    rfe_selector = RFE(clf, n_features_to_select = 10)
    rfe_selector = rfe_selector.fit(X, y)

    # TODO figure out how to subset X, y, X_test, and y_test
    # Run it through the net


def kBest(X, y, X_test, y_test, algorithm, exp_dir, featureList = []):

    checkpoint_path = os.path.join(exp_dir, 'nn_checkpoints', '{}_kbest.ckpt'.format(algorithm))
    best_path = os.path.join(exp_dir, 'nn_checkpoints', '{}_kbest_optimal.ckpt'.format(algorithm))

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif

    highest, bestI, model, featureNames, returnNet = 0, 0, None, list(), -100
    if algorithm == "rasta": # There are only 9 features for rasta
        skip = 1
    else:
        skip = 10
    for i in range(skip, X.shape[-1], skip): # Iterate over groups of skip features

        selector = SelectKBest(f_classif, k=i)
        X_new = selector.fit_transform(X, y)
        X_test_new = selector.transform(X_test)

        support = selector.get_support()

        net = baseModel(X_new, y, X_test_new, y_test, checkpoint_path) # For k == 10: Validation loss cutoff after 11 epochs; Testing accuracy 60.60%, Validation accuracy 59.31%
        if max(net.history.history['val_binary_accuracy'][-4:]) > highest: # Check for last three epochs because of callback
            highest = max(net.history.history['val_binary_accuracy'][-4:])
            bestI = i
            model = net
            featureNames = list()
            if "compare" in algorithm:
                for i, j in zip(support, featureList):
                    if i:
                        featureNames.append(j)
                # with open(os.path.join(exp_dir, 'nn_checkpoints', 'featureNames.pkl'), 'wb') as f:
                #     pickle.dump(featureNames, f)
            elif "auditory" in algorithm:
                for i, j in zip(support, featureList):
                    if i:
                        featureNames.append(j)
                # with open(os.path.join(exp_dir, 'nn_checkpoints', 'featureNames.pkl'), 'wb') as f:
                #     pickle.dump(featureNames, f)

            # model.model.save_weights(best_path)

        K.clear_session() # https://stackoverflow.com/questions/50895110/what-do-i-need-k-clear-session-and-del-model-for-keras-with-tensorflow-gpu
        # return model, featureNames, X_new, X_test_new
    print()
    print("#########################################################################################")
    print("Final Epoch's Training Accuracy:\t{}".format(model.history.history['binary_accuracy'][-1]))
    print("Final Epoch's Training Loss:\t{}".format(model.history.history['loss'][-1]))
    print("Final Epoch's Testing Accuracy:\t{}".format(model.history.history['val_binary_accuracy'][-1]))
    print("Final Epoch's Testing Loss:\t{}".format(model.history.history['val_loss'][-1]))
    print("Number of Epochs Survived:\t{}".format(len(model.history.history['loss'])))
    print("Best value for i (in tens):\t{}".format(bestI))
    print("Feature names (if applicable):\t{}".format(featureNames))
    print("#########################################################################################")
    print()

    return model, featureNames, X_new, X_test_new


def baseModel(X, y, X_test, y_test, checkpoint_path): 

    # Train the model
    net = feedForward(X, y, X_test, y_test, checkpoint_path, checkpoint_path.replace('.ckpt', '.csv'))

    # Evaluate the model
    _, accuracy = net.model.evaluate(X_test, y_test)
    print('Validation Accuracy: %.2f' % (accuracy*100))

    return net


def getData(exp, data, byFrame, experiment):

    y_train = joblib.load('{}/vectors/classifiers/{}-y_train.pkl'.format(exp, experiment))
    X_train = joblib.load('{}/vectors/classifiers/{}-X_train.pkl'.format(exp, experiment))
    trainSpeakerDict = joblib.load('{}/vectors/classifiers/{}-trainSpeakerDict.pkl'.format(exp, experiment))

    y_test = joblib.load('{}/vectors/classifiers/{}-y_test.pkl'.format(exp, experiment))
    X_test = joblib.load('{}/vectors/classifiers/{}-X_test.pkl'.format(exp, experiment))
    devSpeakerDict = joblib.load('{}/vectors/classifiers/{}-devSpeakerDict.pkl'.format(exp, experiment))
    
    scaler = joblib.load('{}/vectors/classifiers/{}-scaler.pkl'.format(exp, experiment))

    for speaker in devSpeakerDict:
        devSpeakerDict[speaker][1] = np.array(devSpeakerDict[speaker][1])

    # Encode from categorical to numerical
    le = LabelEncoder()
    le.fit(y_train)

    return X_train, le.transform(y_train), X_test, le.transform(y_test), devSpeakerDict


def callExperiment(X, y, X_test, y_test, exp_dir, algorithm, experiment, featureList = []):

    model = baseModel(X, y, X_test, y_test, os.path.join(exp_dir, 'nn_checkpoints', '{}_baseline.ckpt'.format(algorithm)))
    plotIt(model, exp_dir, experiment, algorithm, "Baseline Feed Forward Neural Net")

    # With F-score selection
    # model, featureNames, X_new, X_test_new = kBest(X, y, X_test, y_test, algorithm, exp_dir, featureList)
    # featureContribution(X_new, y, X_test_new, y_test, featureNames, max(model.history.history['val_binary_accuracy'][-4:]))
    # plotIt(model, exp_dir, "{}-{}".format(experiment, 'kBest'), algorithm, "kBest Feed Forward Neural Net")

    # Recursive feature elimination
    # rBest(X, y, X_test, y_test)

    # PCA Reduction
    # pcaReduction(X)


def main(exp_dir, data_dir, algorithm, random_state, byFrame, experiment, RUNNUM, loocv_not_withheld):

    # Load in the data
    X, y, X_test, y_test, _ = getData(exp_dir, data_dir, byFrame, experiment)

    if byFrame:

        callExperiment(X, y, X_test, y_test, exp_dir, algorithm, experiment)

    else:

        if "compare" in algorithm:
            with open("/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Code/feature-extraction/compare_list.pkl", "rb") as f:
                featureList = pickle.load(f)
        elif "auditory" in algorithm:
            with open("/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Code/feature-extraction/auditory_list.pkl", "rb") as f:
                featureList = pickle.load(f)

        # Remove columns filled with only np.nan
        indicator = MissingIndicator(missing_values=np.nan, features="all")
        mask_all = indicator.fit_transform(X)
        # Iterate over the columns
        transposed = mask_all.transpose()
        # Check if there are all NaN values in a column (here a row since we transposed for iteration)
        badColumns = [i for i, column in enumerate(transposed) if False not in column]

        # If we're dealing with features with names (compare or auditory), then remove the bad names
        j = 0 # Number of times we've removed something
        for i in badColumns:
            # Remove the column from the matrices
            X = np.delete(X, i - j, axis = 1)
            X_test = np.delete(X_test, i - j, axis = 1)
            # Remove the feature name from the list of names
            del featureList[i - j]
            j += 1
        
        # If there are missing values, impute them
        if np.isnan(X).any() or np.isnan(X_test).any():
            from sklearn.impute import SimpleImputer

            imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(X)
            X = imp.transform(X)
            X_test = imp.transform(X_test)

        callExperiment(X, y, X_test, y_test, exp_dir, algorithm, experiment, featureList)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Call all of the regression algorithms and consolidate a global report.')
    parser.add_argument('exp_dir', nargs='?', type=str, help='Temporary experiment directory.', default='/tmp/tmp.TcRZJvM59O')
    parser.add_argument('data_dir', nargs='?', type=str, help='Location for the map for the scalar score (e.g. MMSE).', default='../../Data/ADReSS-IS2020-data/train')
    parser.add_argument('algorithm', nargs='?', type=str, help='Which feature extraction protocol we are using', default='auditory-global')
    parser.add_argument('random_state', nargs='?', type=int, help='Affects the ordering of the indices, which controls the randomness of each fold in KFold validation.', default=1)
    parser.add_argument('by_frame', nargs='?', type=str, help='True if we need to run this by frame or False if we are using COMPARE or something else distilled.', default="False")
    parser.add_argument('run_num', nargs='?', type=str, help='Which runthrough we are on.', default='3104')
    parser.add_argument('loocv_not_withheld', nargs='?', type=str, help='If True, we will do 5 fold leave-one-out cross validation; if False, we are training on all the training data and testing on the withheld test data ', default='True')

    args = parser.parse_args()

    # This distinction may not actually matter for the neural networking implementation
    if args.by_frame == "True":
        for experiment in ['raw', 'averaged', 'flattened']: #['raw', 'averaged', 'flattened']: #, 'averaged_and_flattened']:
            print('Now working on {}'.format(experiment))
            main(args.exp_dir, args.data_dir, args.algorithm, args.random_state, True, experiment, args.run_num, args.loocv_not_withheld)
    else:
        main(args.exp_dir, args.data_dir, args.algorithm, args.random_state, False, args.algorithm, args.run_num, args.loocv_not_withheld)
