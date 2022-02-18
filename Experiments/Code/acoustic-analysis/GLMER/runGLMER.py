#!/usr/bin/env python3

# https://towardsdatascience.com/how-to-run-linear-mixed-effects-models-in-python-jupyter-notebooks-4f8079c4b589
#   Rationale and sources
#   Try #1 - with statsmodels

# Load packages
import stepwiseGLMER
import numpy as np
import pandas as pd
# from patsy import dmatrices
import statsmodels.api as sm
import statsmodels.formula.api as smf


# Patsy doesn't accept numerals in pandas DataFrame column strings
def fixCols(df):

    colList = df.columns.tolist()
    if 'F0' in colList:
        i = colList.index('F0')
        colList[i] = 'FundFreq'

        df.columns = colList
        
    return df


def runGLMER(df, formula, printList, which, step):

    # Create the two design matrices using patsy (https://www.statsmodels.org/dev/gettingstarted.html)
        # y == endog - endogenous variable(s) (i.e. dependent, response, regressand, etc.)
        # X == exog - exogenous variable(s) (i.e. independent, predictor, regressor, etc.)

    # NOTE: ID, Age and Gender are random effects because they "introduce variance into the data but are not variables of interest" (Granlund, 2012)
    # NOTE: All other variables measured are fixed effects because they "influence the mean and are of interest to the analyst" (Granlund, 2012)

    # The way to split endogenous and exogenous variables without using the formula option
    # q = df.drop(['ID', 'Condition'], axis = 1) 
    # y, X = dmatrices('Condition ~ {}'.format(" + ".join(i for i in q.columns if i in bestFeats)), data = df, return_type = 'dataframe')

    df = fixCols(df)

    # Construct the model
    # glm_binom = smf.glm(formula = formula, data = df, groups = df[['ID', 'Age', 'Gender']], family = sm.families.Binomial())
    glm_binom = smf.glm(formula = formula, data = df, groups = df['ID'], family = sm.families.Binomial())

    # Fit the model
    res = glm_binom.fit()

    # Visualize the summary
    printList.append(res.summary())
    printList.append('')
    printList.append('Total number of trials:\t{}'.format(df.shape[0]))
    printList.append('')
    printList.append('Parameters: ')
    printList.append(res.params.to_string())
    printList.append('')
    printList.append('T-values: ')
    printList.append(res.tvalues.to_string())
    printList.append('')
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
    with open('../../../Results/03_Acoustic_Analysis/global/categorical/{}-{}.txt'.format(run, which), 'w') as f:
        for item in printList:
            f.write("%s\n" % item)


def loadDataSet(level, which):

    if level == "global":
        df = pd.read_csv("./{}/GlobalMeasures_{}-categorical.csv".format(level, which))
    else:
        df = pd.read_csv("./{}/SegmentalMeasures_{}-categorical.csv".format(level, which))

    return df


def main(level = "global", which = "Normalised_audio-chunks", step = False):

    # Load the data
    df = loadDataSet(level, which)

    # Step-wise feature selection for best model by Bayes Information Criterion
    if step:
        formula = stepwiseGLMER.main(df, level)

    # Get a baseline with just everything in the model
    else:
        if level == "global":
            # For global features we've got ['Age', 'Gender', 'F0', 'iqr', 'Intensity', 'ArticulationRate', 'PausingRate']
            # Fundamental Frequency and its Interquartile Range have an interaction
            formula = "Condition ~ Intensity + ArticulationRate + PausingRate + FundFreq*iqr"
        else:
            pass

    # Run GLMER
    printList = list()
    printList.append("Level:    {}      Which:  {}      Formula:  {}".format(level, which, formula))
    runGLMER(df, formula, printList, which, step)


if __name__ == "__main__":

    main(level = "global", which = "Normalised_audio-chunks", step = False)
