#!/usr/bin/env python3

from sklearn.metrics import confusion_matrix
from scipy import stats

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import glob
import sys
import os


def plot_confusion_matrix(df_confusion, name, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    # plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

    plt.savefig(name)


def predict(featureDir, algorithm):

    # Count regressors by algorithm name
    devSpeakerDict = joblib.load('{}/vectors/regressors/{}-devSpeakerDict.pkl'.format(featureDir, algorithm))
    pickleFiles = glob.glob("{}/models/regressors/*.pkl".format(featureDir))
    pickleNames = list(set([x.rsplit('-', 1)[0] for x in pickleFiles]))

    regName = ""

    for pickle in pickleNames:

        y_actuals, y_preds = list(), list()
        diff = 1000
        winner = 6

        for i in range(5):
            y_actual, y_pred = list(), list()
            mod = joblib.load(pickle + "-{}.pkl".format(i))

            for speaker in devSpeakerDict:
                truth, vectors = devSpeakerDict[speaker]
                pred = mod.predict(vectors)
                for p in pred:

                # avg = np.mean(pred)
                    y_actual.append(truth)
                    y_pred.append(round(p))

            comparison = list(zip(y_actual, y_preds))
            off = np.abs(np.sum([x[0] - x[1] for x in comparison]))

            if off < diff:
                diff = off
                winner = i
                y_actuals = y_actual
                y_preds = y_pred

        for i, prediction in enumerate(y_preds):
            if prediction > 30:
                y_preds[i] = int(np.mean(y_actuals))

        confuDF = pd.crosstab(pd.Series(y_actuals, name = 'Actual'), pd.Series(y_pred, name = 'Predicted'))
        name = '{}/reports/plots/{}-{}-{}.pdf'.format(featureDir, algorithm, pickle.split(algorithm)[-1][1:], i)
        plot_confusion_matrix(confuDF, name, title = "Confusion Matrix")

        normConfuDF = confuDF / confuDF.sum(axis=1)
        name = '{}/reports/plots/norm-{}-{}-{}.pdf'.format(featureDir, algorithm, pickle.split(algorithm)[-1][1:], i)
        plot_confusion_matrix(confuDF, name, title = "Normalized Confusion Matrix")

        print(y_actuals[:])
        print(y_preds[:])
        print(name, print(np.corrcoef(y_actuals, y_preds)))
        print()


def mmse():

    for featureDir in ['/tmp/tmp.Psc7g4V77e']: # Just for COMPARE
    # for featureDir in ['/tmp/tmp.1Uu3jV6Anl', '/tmp/tmp.2CKHTckEZ3', '/tmp/tmp.OnCYm1VXui', '/tmp/tmp.Psc7g4V77e']:
        algorithm, runnum = open("{}/algorithm.txt".format(featureDir)).read().strip().split("-")
        print(algorithm)
        print()

        # Get lists for truth and predictions
        predict(featureDir, algorithm)

        # Plot truth vs averaged prediction



def pearsonr():

    # Get an array of all individual diagnoses
    # Get an array of their MMSE scores
    x, y = list(), list()
    ccDF = pd.read_csv('../../Data/ADReSS-IS2020-data/train/cc_meta_data.txt', sep = ';').rename(columns=lambda x: x.strip())
    cdDF = pd.read_csv('../../Data/ADReSS-IS2020-data/train/cd_meta_data.txt', sep = ';').rename(columns=lambda x: x.strip())
    for i, row in ccDF.iterrows():
        if row['mmse'].strip() != 'NA':
            x.append(0)
            y.append(int(row['mmse'].strip()))
    for i, row in cdDF.iterrows():
        x.append(1)
        y.append(row['mmse'])
    testDF = pd.read_csv('../../Data/ADReSS-IS2020-data/meta_data_test.txt', sep = ';').rename(columns=lambda x: x.strip())
    for i, row in testDF.iterrows():
        x.append(row['Label'])
        y.append(row['mmse'])

    rho = stats.pearsonr(x, y)
    print(rho)


def main():

    # Calculate the Pearson's Correlation Coefficient of the data's binary classes and MMSE scores
    # TODO this needs to be calculating real MMSE compared to the predictions
    # pearsonr()

    # Plot averaged MMSE prediction by ground truth
    mmse()



if __name__ == "__main__":

    main()
