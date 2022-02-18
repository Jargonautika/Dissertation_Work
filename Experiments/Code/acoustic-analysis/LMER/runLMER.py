#!/usr/bin/env python3

# https://towardsdatascience.com/how-to-run-linear-mixed-effects-models-in-python-jupyter-notebooks-4f8079c4b589
#   Rationale and sources
#   Try #1 - with statsmodels

# Load packages
import stepwiseLMER
import numpy as np
import pandas as pd
# from patsy import dmatrices
import statsmodels.formula.api as smf


# Patsy doesn't accept numerals in pandas DataFrame column strings
def fixCols(df):

    colList = df.columns.tolist()
    if 'F0' in colList:
        i = colList.index('F0')
        colList[i] = 'FundFreq'

        df.columns = colList
        
    return df


def runLMER(df, formula, printList, which, step):

    df = fixCols(df)

    # Construct the model
    md = smf.mixedlm(formula = formula, data = df, groups = df['ID'])

    # Fit the model
    res = md.fit(method = ['lbfgs'])

    # Visualize the summary
    printList.append(res.summary())
    printList.append('')
    printList.append('Total number of trials: {}'.format(df.shape[0]))
    printList.append('')
    printList.append('Parameters: ')
    printList.append(res.params.to_string())
    printList.append('')
    printList.append('T-values: ')
    printList.append(res.tvalues.to_string())
    printList.append('')
    # print('Odds Ratio: ')
    # print(np.exp(res.params))
    # print()
    printList.append('Odds Ratio w/ Confidence Intervals: ')
    conf = res.conf_int()
    conf['Odds Ratio'] = res.params
    conf.columns = ['5%', '95%', 'Odds Ratio']
    printList.append(np.exp(conf).to_string())

    # Save it out
    if step:
        run = "BIC"
    else:
        run = "ALL"
    with open('../../../Results/03_Acoustic_Analysis/global/numerical/{}-{}.txt'.format(run, which), 'w') as f:
        for item in printList:
            f.write("%s\n" % item)


def loadDataSet(level, which):

    if level == "global":
        df = pd.read_csv("./{}/GlobalMeasures_{}-numerical.csv".format(level, which))
    else:
        df = pd.read_csv("./{}/SegmentalMeasures_{}-numerical.csv".format(level, which))

    return df


def main(level = "global", which = "Normalised_audio-chunks", step = False):

    # Load the data
    df = loadDataSet(level, which)

    # Step-wise feature selection for best model by Bayes Information Criterion
    if step:
        formula = stepwiseLMER.main(df)

    # Get a baseline with just everything in the model
    else:
        if level == "global":
            # For global features we've got ['Age', 'Gender', 'F0', 'iqr', 'Intensity', 'ArticulationRate', 'PausingRate']
            # Fundamental Frequency and its Interquartile Range have an interaction
            formula = "MMSE ~ Intensity + ArticulationRate + PausingRate + FundFreq*iqr"
        else:
            pass

    # Run LMER
    printList = list()
    printList.append("Level:    {}      Which:  {}      Formula:  {}".format(level, which, formula))
    runLMER(df, formula, printList, which, step)


if __name__ == "__main__":

    main(level = "global", which = "Normalised_audio-chunks", step = True)
