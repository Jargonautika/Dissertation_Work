#!/usr/bin/env python3

# Follow the instructions / rationale here:
    # https://medium.com/@garrettwilliams90/stepwise-feature-selection-for-statsmodels-fda269442556

import numpy as np
import pandas as pd
# from patsy import dmatrices
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Patsy doesn't accept numerals in pandas DataFrame column strings
def fixCols(df):

    colList = df.columns.tolist()
    i = colList.index('F0')
    colList[i] = 'FundFreq'

    df.columns = colList
    return df


# Scores less than 5 are good scores
def computeVIF(X, featList):

    # Define an empty dataframe to capture the VIF scores
    vif = pd.DataFrame()

    # Create a Dataframe with only our selected features 
    X_new = X[featList]

    # Label the scores with their related columns
    vif["features"] = X_new.columns

    # For each column,run a variance_inflaction_factor against all other columns
        # to get a VIF Factor score
    vif["VIF"] = [variance_inflation_factor(X_new.values, i) \
                    for i in range(len(X_new.columns))]
    
    return vif


def potentiallyRemoveFeature(df, current_features, base):

    features_to_remove = list()
    # Iterate through every column of interest for this run
    for feat in current_features:

        # Make the formula
        if any(['*' in x for x in current_features]):
            formula = "Condition ~ " + " + ".join([x for x in current_features if x != feat]).replace(" + FundFreq*iqr", "").replace(" FundFreq*iqr", "") + " + FundFreq*iqr"
        else:
            formula = "Condition ~ " + " + ".join([x for x in current_features if x != feat])

        # Fit a model for our target and our selected columns 
        glm_binom = smf.glm(formula, data = df, groups = df[['ID', 'Age', 'Gender']], family = sm.families.Binomial())

        # Fit the model and get the results
        res = glm_binom.fit()

        # Calculate the BIC
        bic = res.bic_llf

        if bic > base:
            features_to_remove.append(feat)

    # Make the formula
    if "FundFreq*iqr" in current_features and "FundFreq*iqr" not in features_to_remove:
        formula = "Condition ~ " + " + ".join([x for x in current_features if x not in features_to_remove]).replace(" + FundFreq*iqr", "").replace(" FundFreq*iqr", "") + " + FundFreq*iqr"
    else:
        formula = "Condition ~ " + " + ".join([x for x in current_features if x not in features_to_remove])

    return_features = [feat for feat in current_features if feat not in features_to_remove]

    current_best_bic = smf.glm(formula, data = df, groups = df[['ID', 'Age', 'Gender']], family = sm.families.Binomial()).fit().bic_llf

    return formula, return_features, current_best_bic


def next_possible_feature(df, oldFormula, newFormula, current_features, col, base = 0.0):

    # Fit a model for our target and our selected columns 
    glm_binom = smf.glm(formula = newFormula, data = df, groups = df[['ID', 'Age', 'Gender']], family = sm.families.Binomial())

    # Fit the model and get the results
    res = glm_binom.fit()

    # Calculate the BIC
    bic = res.bic_llf

    # The model is improved by adding the feature
    if bic < base:

        current_features = current_features + [col]

        # Do a top-down backward pass to see if removing any of the variables improves the model
        newFormula, current_features, bic = potentiallyRemoveFeature(df, current_features, bic)

        return newFormula, current_features, bic
    
    # The model is not improved by adding the feature
    else:
        return oldFormula, current_features, base


def bottomUp(df, bestFormula, best, base):

    # Start with the first predictor, and see all possibilities for our second.
    selected_features = [best]

    for col in df.columns.tolist()[:-1] + ["FundFreq*iqr"]:

        if col not in ['ID', 'Age', 'Gender'] and col not in best:

            # Make the new formula; make sure that the interaction comes at the end
            if "FundFreq*iqr" in bestFormula:
                formula = bestFormula.replace(" + FundFreq*iqr", "").replace(" FundFreq*iqr", "") + " + {}".format(col) + " + FundFreq*iqr"
            else:
                formula = bestFormula + " + {}".format(col)

            # Add a new column to the best performing column
            currentFormula, current_features, currentBase = next_possible_feature(df, bestFormula, formula, current_features = selected_features, col = col, base = base)

            # We've added something and therefore need to update things
            if currentFormula != bestFormula: 

                # Check for multicollinearity
                if "*" in col:
                    tmp_features = selected_features + col.split('*')
                else:
                    tmp_features = selected_features + [col]
                vifDF = computeVIF(df, tmp_features)

                # If all is right in the world, add the next best to the good list
                if any(vifDF.VIF > 5):

                    print("Multicollinearity detected.")
                    print(vifDF)
                    raise AssertionError

                else:

                    bestFormula = currentFormula 
                    selected_features = current_features
                    base = currentBase

    return bestFormula


def firstPass(df):

    # The old way
    # q = df.drop(['ID', 'Condition'], axis = 1)
    # y, X = dmatrices('Condition ~ {}'.format(" + ".join(i for i in q.columns)), data = df, return_type = 'dataframe')

    #Create an empty dictionary that will be used to store our results
    function_dict = {'predictor': [], 'formula': [], 'BIC':[]}

    # Fix the column names
    df = fixCols(df)

    # Calculate the Bayes Information Criterion (BIC) from the log likelihood function
    # for the baseline model (when only the constant is included; see Fields, 2012)
    # NOTE I switched from using the R-statistic because Fields says "R should be treated 
    # with some caution, and it is invalid tosquare this value and interpret it as you 
    # would in linear regression"
    # base = smf.glm(formula = "Condition ~ Intensity + ArticulationRate + PausingRate + FundFreq*iqr", data = df, groups = df[['ID', 'Age', 'Gender']], family = sm.families.Binomial()).fit().bic_llf

    #Iterate through every column in X (skip the constant 'Intercept' column)
    for col in df.columns.tolist()[:-1] + ["FundFreq*iqr"]:

        if col not in ['ID', 'Age', 'Gender']:

            # Create a formula for testing the factor by itself
            colFormula = "Condition ~ {}".format(col)

            # Fit a model for our target and our selected column 
            glm_binom = smf.glm(formula = colFormula, data = df, groups = df[['ID', 'Age', 'Gender']], family = sm.families.Binomial())

            # Fit the model and get the results
            res = glm_binom.fit()

            # Add the column name to our dictionary
            function_dict['predictor'].append(col)

            # Add the formula to our dictionary
            function_dict['formula'].append(colFormula)

            # Calculate Hosmer and Lemeshow (1989)'s R2L (analogous to linear regression's r2)
            # See Fields 2012, equation 8.8
            # See above NOTE
            # r2l = (base - res.deviance) / base

            # Calculate the BIC
            bic = res.bic_llf

            # Add the r-squared value to our dictionary
            function_dict['BIC'].append(bic)
        
    # Once it's iterated through every column, turn our dictionary into a DataFrame and sort it
    # Smaller is better
    function_df = pd.DataFrame(function_dict).sort_values(by=['BIC'], ascending = True)

    # Get the top performer
    best = function_df['predictor'].tolist()[0]

    bestFormula = function_df['formula'].tolist()[0]

    newBase = function_df['BIC'].tolist()[0]

    return best, bestFormula, newBase
    

def main(df, level = "global"):

    # First pass
    best, bestFormula, base = firstPass(df)

    # Bottom-up, hierarchical approach to feature selection
    bestFormula = bottomUp(df, bestFormula, best, base)

    return bestFormula


if __name__ == "__main__":

    df = pd.read_csv("../global/GlobalMeasures_Full_wave_enhanced_audio-categorical.csv")

    bestFormula = main(df)

# NOTE: 
    # For "Global" and "Full_wave_enhanced_audio", F0 is multicollinear with "Articulation Rate" and "iqr", so it is unincluded
# TODO 
    # Manually check for the other 4-6 model configurations
    