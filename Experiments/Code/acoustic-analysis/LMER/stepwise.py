#!/usr/bin/env python3

# Follow the instructions / rationale here:
    # https://medium.com/@garrettwilliams90/stepwise-feature-selection-for-statsmodels-fda269442556

import numpy as np
import pandas as pd
# from patsy import dmatrices
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


def potentiallyRemoveFeature(newFormula, current_features, base):

    features_to_remove = list()
    # Iterate through every column of interest for this run
    for feat in current_features:

        # Make the formula
        if any(['*' in x for x in current_features]):
            formula = "MMSE ~ " + " + ".join([x for x in current_features if x != feat]).replace(" + FundFreq*iqr", "").replace(" FundFreq*iqr", "") + " + FundFreq*iqr"
        else:
            formula = "MMSE ~ " + " + ".join([x for x in current_features if x != feat])

        # Fit a model for our target and our selected columns 
        md = smf.mixedlm(formula = formula, data = df, groups = df[['ID', 'Age', 'Gender']])

        # Fit the model and get the results
        res = md.fit()

        # Calculate the BIC
        bic = -2 * res.llf + np.log(res.nobs) * (res.df_modelwc)

        if bic > base:
            features_to_remove.append(feat)

    # Make the formula
    if "FundFreq*iqr" in current_features and "FundFreq*iqr" not in features_to_remove:
        formula = "MMSE ~ " + " + ".join([x for x in current_features if x not in features_to_remove]).replace(" + FundFreq*iqr", "").replace(" FundFreq*iqr", "") + " + FundFreq*iqr"
    else:
        formula = "MMSE ~ " + " + ".join([x for x in current_features if x not in features_to_remove])

    return_features = [feat for feat in current_features if feat not in features_to_remove]

    md = smf.mixedlm(formula = formula, data = df, groups = df[['ID', 'Age', 'Gender']])
    res = md.fit()
    current_best_bic = -2 * res.llf + np.log(res.nobs) * (res.df_modelwc)

    return formula, return_features, current_best_bic


def next_possible_feature(oldFormula, newFormula, current_features, col, base = 0.0):

    # Fit a model for our target and our selected columns 
    md = smf.mixedlm(formula = newFormula, data = df, groups = df[['ID', 'Age', 'Gender']])

    # Fit the model and get the results
    res = md.fit()

    # Calculate the BIC
    bic = -2 * res.llf + np.log(res.nobs) * (res.df_modelwc)

    # The model is improved by adding the feature
    if bic < base:

        current_features = current_features + [col]

        # Do a top-down backward pass to see if removing any of the variables improves the model
        newFormula, current_features, bic = potentiallyRemoveFeature(newFormula, current_features, bic)

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
            currentFormula, current_features, currentBase = next_possible_feature(bestFormula, formula, current_features = selected_features, col = col, base = base)

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

    #Create an empty dictionary that will be used to store our results
    function_dict = {'predictor': [], 'formula':[], 'BIC': []}

    # Fix the column names
    df = fixCols(df)

    # baseModel = smf.mixedlm(formula, df, groups = df['ID']).fit()
    # base = -2 * baseModel.llf + np.log(baseModel.nobs) * (baseModel.df_modelwc) # https://stackoverflow.com/questions/69172584/mixedlmresults-object-return-nan-bic-what-can-be-the-reason

    #Iterate through every column in X (skip the constant 'Intercept' column)
    for col in df.columns.tolist()[:-1] + ["FundFreq*iqr"]:

        if col not in ['ID', 'Age', 'Gender']:

            # Create a formula for testing the factor by itself
            colFormula = "MMSE ~ {}".format(col)

            # Fit a model for our target and our selected column 
            md = smf.mixedlm(formula = colFormula, data = df, groups = df[['ID', 'Age', 'Gender']])

            # Fit the model and get the results
            res = md.fit()

            # Add the column name to our dictionary
            function_dict['predictor'].append(col)

            # Add the formula to our dictionary
            function_dict['formula'].append(colFormula)

            # Calculate the BIC
            bic = -2 * res.llf + np.log(res.nobs) * (res.df_modelwc)

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

    df = pd.read_csv("../global/GlobalMeasures_Full_wave_enhanced_audio-numerical.csv")

    bestFormula = main(df)
