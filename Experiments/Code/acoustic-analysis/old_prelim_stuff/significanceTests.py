#!/usr/bin/env python3

from scipy import stats
import numpy as np


# TODO Quick and dirty:
    # Count statistics from the volume of various segments produced by individual per group (weighted by length of the duration of the files?)

# TODO a bit more complex
    # Logistic regression model (binomial regression)
        # Fixed measure effects are the vowels and consonants themselves
        # Other measures seem to be random measures
        # Subject level random effect accounts for the individual
        # "glmer" binomial regression

        # the big issue is that there are repeated measurements for each individual


# TODO use a linear mixed effects model
# https://towardsdatascience.com/how-to-run-linear-mixed-effects-models-in-python-jupyter-notebooks-4f8079c4b589
# https://www.statsmodels.org/stable/examples/notebooks/generated/mixed_lm_example.html
# https://learn.illinois.edu/pluginfile.php/5361567/mod_resource/content/1/bw_LME_tutorial2.pdf


# TODO run a series of ANOVA tests
# https://www.statisticshowto.com/probability-and-statistics/hypothesis-testing/anova/


def chiSquareTest(contingencyTable):
    
    # Using the Pearson's Correlation Coefficient
    X, cP, dof, _ = stats.chi2_contingency(contingencyTable)

    # Using the G-test with log likelihood ratio
    G, lP, ddof, _ = stats.chi2_contingency(contingencyTable, lambda_="log-likelihood")

    return X, cP, dof, G, lP, ddof


# Tests the null hypothesis that two related paired samples come from the same distribution
# NOTE this is only possible if the sample sizes are the same length; so we must pull from 
# the same distribution
def wilcoxonTest(x, y):

    wStatistic, twoTailedPValue = stats.wilcoxon(x, y)
    return wStatistic, twoTailedPValue


# NOTE violates some assumptions because Speaker A is going to produce their own /v/ sounds more 
# similarly to themselves than within their group. 
def independentT(x, y):

    # x == len(50); y == len(64)

    # Null Hypothesis: There is no significant difference for a given speech sound between diagnosed and non-diagnosed individuals
    # Alternate Hypothesis: There is a significant difference for a given speech sound between diagnosed and non-diagnosed individuals

    tStatistic, twoTailedPValue = stats.ttest_ind(x, y, equal_var = False)
    return tStatistic, twoTailedPValue


def main(x, y, crossTab, text):

    print(text)

    # An Independent Samples t-test compares the means of two groups
    t, iP = independentT(x, y)

    # We have non-gaussian data in some cases; let's run some nonparametric tests
    # w, wP = wilcoxonTest(x, y)

    # A chi-square test for independence compares two variables in a contingency 
    # table to see if they are related. In a more general sense, it tests to see 
    # whether distributions of categorical variables differ from each another
    X, cP, dof, G, lP, ddof = chiSquareTest(crossTab)
    print(1)

    print("#############################################")
    print(text)
    print('Independent T-test')
    print('\t\t\tt-statistic: {}\t pValue: {}'.format(t, iP))
    # print('Wilcoxon Nonparametric Test')
    # print('\tw-statistic: {}\t\t pValue: {}'.format(w, wP))
    print('Chi-Square Ind. Test')
    print('\t\t\tX2-statistic: {}\t X2 pValue: {}\t DOF: {}'.format(X, cP, dof))
    print('G-Test')
    print('\t\t\tG-statistic: {}\t G pValue: {}\t DOF: {}'.format(G, lP, ddof))
    print("#############################################")
