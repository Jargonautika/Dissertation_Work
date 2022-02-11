#!/usr/bin/env python3

# https://towardsdatascience.com/how-to-run-linear-mixed-effects-models-in-python-jupyter-notebooks-4f8079c4b589
#   Rationale and sources
#   Try #1 - with statsmodels

# Load packages
import stepwise
import numpy as np
import pandas as pd
from patsy import dmatrices
import statsmodels.api as sm

def runLMER(df, bestFeats):

    # Create the two design matrices using patsy (https://www.statsmodels.org/dev/gettingstarted.html)
        # y == endog - endogenous variable(s) (i.e. dependent, response, regressand, etc.)
        # X == exog - exogenous variable(s) (i.e. independent, predictor, regressor, etc.)

    # q = df.drop(['ID', 'Age', 'Gender', 'Condition'], axis = 1) 
    q = df.drop(['ID', 'Condition'], axis = 1) 
    y, X = dmatrices('Condition ~ {}'.format(" + ".join(i for i in q.columns if i in bestFeats)), data = df, return_type = 'dataframe')

    # Construct the model
    glm_binom = sm.GLM(y, X, family = sm.families.Binomial())

    # Fit the model
    res = glm_binom.fit()

    # Visualize the summary
    print(res.summary())
    print()
    print('Total number of trials:',  X.iloc[:, 0].sum())
    print()
    print('Parameters: ')
    print(res.params)
    print()
    print('T-values: ')
    print(res.tvalues)
    print()
    # print('Odds Ratio: ')
    # print(np.exp(res.params))
    # print()
    print('Odds Ratio w/ Confidence Intervals: ')
    conf = res.conf_int()
    conf['Odds Ratio'] = res.params
    conf.columns = ['5%', '95%', 'Odds Ratio']
    print(np.exp(conf))


def loadDataSet(level, which):

    if level == "global":
        df = pd.read_csv("../{}/GlobalMeasures_{}-numerical.csv".format(level, which))
    else:
        df = pd.read_csv("../{}/SegmentalMeasures_{}-numerical.csv".format(level, which))

    return df


def main(level = "global", which = "Normalised_audio-chunks", step = False):

    # Load the data
    df = loadDataSet(level, which)

    # Step-wise feature selection for best model by Bayes Information Criterion
    if step:
        bestFeats = stepwise.main(df)

    # Get a baseline with just everything in the model
    else:
        bestFeats = df.columns[1:-1].tolist()

    # Run LMER
    print(level, which, bestFeats)
    runLMER(df, bestFeats)



if __name__ == "__main__":

    main()
