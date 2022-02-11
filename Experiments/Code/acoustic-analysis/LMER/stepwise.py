#!/usr/bin/env python3

# Follow the instructions / rationale here:
    # https://medium.com/@garrettwilliams90/stepwise-feature-selection-for-statsmodels-fda269442556

import numpy as np
import pandas as pd
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


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
    
    return vif[vif['features'] == featList[-1]]['VIF'].values[0], vif


# TODO figure out the logic or use collections
def potentiallyRemoveFeature(X_prf, y_prf, current_features, new_feature, base = 0.0):

    '''
    This function will loop through groups of 3+ features to see if our modeling
    improves when we take one out of the mix based on BIC_LLF values.
    X_prf = X dataframe
    y_prf = y dataframe
    current_features = list of features that are already in your model
    new_feature = string with the new candidate
    base = float with the number to beat
    '''   

    # Create an empty dictionary that will be used to store our results
    function_dict = {'group': [], 'BIC':[]}

    # Iterate through every column of interest for this run
    allList = current_features + [new_feature]
    for feat in allList:

        # Subset the group to everything but the one in question
        groupList = allList.copy()
        groupList.remove(feat)

        # Add the column name to our dictionary
        function_dict['group'].append(groupList)

        # Create a dataframe called function_X with our current features + 1
        selected_X = X_prf[groupList + ['Intercept']]

        # Fit a model for our target and our selected columns 
        glm_binom = sm.GLM(y_prf, selected_X, family = sm.families.Binomial())

        # Fit the model and get the results
        res = glm_binom.fit()

        # Calculate the BIC
        bic = res.bic_llf

        # Add the BIC_LLF value to our dictionary
        function_dict['BIC'].append(bic)

    # Once it's iterated through every column, turn our dict into a sorted DataFrame
    function_df = pd.DataFrame(function_dict).sort_values(by=['BIC'], ascending = True)

    # We have actually found a better configuration than what we already have
    if function_df['BIC'].tolist()[0] < base:
        disincludedFeats = [feat for feat in allList if feat not in function_df['group'].tolist()[0]]
        return function_df['group'].tolist()[0], disincludedFeats, function_df['BIC'].tolist()[0]
    # Otherwise the current configuration is best and we should not add to it
    else:
        return function_df['group'].tolist()[0], [new_feature], base


def next_possible_feature(X_npf, y_npf, current_features, ignore_features=[], base = 0.0):

    '''
    This function will loop through each column that isn't in your feature model and 
    calculate the BIC_LLF value if it were the next feature added to your model. 
    X_npf = X dataframe
    y_npf = y dataframe
    current_features = list of features that are already in your model
    ignore_features = list of unused features we want to skip over
    '''   

    # Create an empty dictionary that will be used to store our results
    function_dict = {'predictor': [], 'BIC':[]}

    # Iterate through every column in X (exclude the standardizing intercept)
    for col in X_npf.columns[1:]:

        # But only create a model if the feature isn't already selected or ignored
        if col not in (current_features+ignore_features):

            # Add the column name to our dictionary
            function_dict['predictor'].append(col)

            # Create a dataframe called function_X with our current features + 1
            selected_X = X_npf[current_features + ['Intercept', col]]

            # Fit a model for our target and our selected columns 
            glm_binom = sm.GLM(y_npf, selected_X, family = sm.families.Binomial())

            # Fit the model and get the results
            res = glm_binom.fit()

            # Calculate the BIC
            bic = res.bic_llf

            # Add the BIC_LLF value to our dictionary
            function_dict['BIC'].append(bic)

            # # Calculate the r-squared-L value
            # r2l = (base - res.deviance) / base

    # Once it's iterated through every column, turn our dict into a sorted DataFrame
    function_df = pd.DataFrame(function_dict).sort_values(by=['BIC'], ascending = True)

    # Get the new additional top performer
    newBest = function_df['predictor'].tolist()[0]

    # For # variables > 2 (we're on the 3rd pass or beyond), do a top-down backward
    # pass to see if removing any of the variables improves the model
    if len(current_features) >= 2:

        current_features, badList, newBase = potentiallyRemoveFeature(X_npf, y_npf, current_features, newBest, base)
        ignore_features += badList

    else:

        # Get the new base value
        newBase = function_df[function_df['predictor'] == newBest]['BIC'].values[0]

        # Update the current_features list
        current_features = current_features + [newBest]

    return current_features, ignore_features, newBase


def bottomUp(X, y, best, function_df, base):

    # Start with the first predictor, and see all possibilities for our second.
    selected_features = [best]
    features_to_ignore = []

    for _ in range(X.shape[-1] - 2):

        # Add a new column to the best performing column
        current_features, features_to_ignore, newBase = next_possible_feature(X_npf=X, y_npf=y, current_features = selected_features, ignore_features = features_to_ignore, base = base)

        # Check for multicollinearity
        vif, vifDF = computeVIF(X, current_features)

        # If all is right in the world, add the next best to the good list
        if vif < 5:
            selected_features = current_features
            base = newBase
        else:
            raise AssertionError

    return selected_features


def firstPass(df):

    # Split our DataFrame into X (input) and y (output); remove metadata
    # q = df.drop(['ID', 'Age', 'Gender', 'Condition'], axis = 1)
    q = df.drop(['ID', 'Condition'], axis = 1)
    y, X = dmatrices('Condition ~ {}'.format(" + ".join(i for i in q.columns)), data = df, return_type = 'dataframe')

    #Create an empty dictionary that will be used to store our results
    function_dict = {'predictor': [], 'BIC':[]}

    # Calculate the Bayes Information Criterion (BIC) from the log likelihood function
    # for the baseline model (when only the constant is included; see Fields, 2012)
    # NOTE I switched from using the R-statistic because Fields says "R should be treated 
    # with some caution, and it is invalid tosquare this value and interpret it as you 
    # would in linear regression"
    base = sm.GLM(y, np.ones(df.shape[0]), family = sm.families.Binomial()).fit().bic_llf

    #Iterate through every column in X (skip the constant 'Intercept' column)
    for col in X.columns[1:]:

        # Create a dataframe called selected_X with only the 1 column
        selected_X = X[['Intercept', col]]

        # Fit a model for our target and our selected column 
        glm_binom = sm.GLM(y, selected_X, family = sm.families.Binomial())

        # Fit the model and get the results
        res = glm_binom.fit()

        # Add the column name to our dictionary
        function_dict['predictor'].append(col)

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

    return best, function_df, X, y, base
    

def main(df):

    # First pass
    best, function_df, X, y, base = firstPass(df)

    # Bottom-up, hierarchical approach to feature selection
    bestFeats = bottomUp(X, y, best, function_df, base)

    return bestFeats


if __name__ == "__main__":

    df = pd.read_csv("../global/GlobalMeasures_Full_wave_enhanced_audio-categorical.csv")

    bestFeats = main(df)

# NOTE: 
    # For "Global" and "Full_wave_enhanced_audio", F0 is multicollinear with "Articulation Rate" and "iqr", so it is unincluded
# TODO 
    # Manually check for the other 4-6 model configurations
    