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


def next_possible_feature (X_npf, y_npf, current_features, ignore_features=[], base = 0.0):

    '''
    This function will loop through each column that isn't in your feature model and 
    calculate the r-squared-L value if it were the next feature added to your model. 
    It will display a dataframe with a sorted r-squared-L value.
    X_npf = X dataframe
    y_npf = y dataframe
    current_features = list of features that are already in your model
    ignore_features = list of unused features we want to skip over
    '''   

    # Create an empty dictionary that will be used to store our results
    function_dict = {'predictor': [], 'r-squared-L':[], 'deviance': [], 'contribution': []}

    # Iterate through every column in X (exclude the standardizing intercept)
    for col in X_npf.columns[1:]:

        # But only create a model if the feature isn't already selected or ignored
        if col not in (current_features+ignore_features):

            # Create a dataframe called function_X with our current features + 1
            selected_X = X_npf[current_features + ['Intercept', col]]

            # Fit a model for our target and our selected columns 
            glm_binom = sm.GLM(y_npf, selected_X, family = sm.families.Binomial())

            # Fit the model and get the results
            res = glm_binom.fit()

            # Add the column name to our dictionary
            function_dict['predictor'].append(col)

            # Save out the deviance
            function_dict['deviance'] = res.deviance

            # Calculate the contribution (and check if it's negative 
            # meaning a predictor that is working against the model as a whole)
            if base - res.deviance < 0.0:
                function_dict['contribution'] = False
            else:
                function_dict['contribution'] = True

            # Calculate the r-squared-L value
            r2l = (base - res.deviance) / base

            # Add the r-squared value to our dictionary
            function_dict['r-squared-L'].append(r2l)

    # Once it's iterated through every column, turn our dict into a sorted DataFrame
    function_df = pd.DataFrame(function_dict).sort_values(by=['r-squared-L'],\
                                                          ascending = False)
    
    # Get the new additional top performer
    newBest = function_df['predictor'].tolist()[0]

    # Get the new base value
    newBase = function_df[function_df['predictor'] == newBest]['deviance'].values[0]

    # Get the contribution boolean
    contributor = function_df[function_df['predictor'] == newBest]['contribution'].values[0]

    return newBest, newBase, contributor


def bottomUp(X, y, best, function_df, base):

    #Start with the first predictor, and see all possibilities for our second.
    selected_features = [best]
    features_to_ignore = []

    for _ in range(X.shape[-1] - 2):

        # Add a new column to the best performing column
        newBest, newBase, contributor = next_possible_feature(X_npf=X, y_npf=y, current_features = selected_features, ignore_features = features_to_ignore, base = base)

        if contributor:

            # Check for multicollinearity
            vif, vifDF = computeVIF(X, selected_features + [newBest])

            # If all is right in the world, add the next best to the good list
            if vif < 5:
                selected_features.append(newBest)
                base = newBase
            else:
                features_to_ignore.append(newBest)

        else:
            features_to_ignore.append(newBest)

    return selected_features


def firstPass(df):

    # Split our DataFrame into X (input) and y (output); remove metadata
    q = df.drop(['ID', 'Age', 'Gender', 'Condition'], axis = 1)
    y, X = dmatrices('Condition ~ {}'.format(" + ".join(i for i in q.columns)), data = df, return_type = 'dataframe')

    #Create an empty dictionary that will be used to store our results
    function_dict = {'predictor': [], 'r-squared-L':[]}

    # Calculate the deviance for the baseline model (when only the constant is included; see Fields, 2012)
    base = sm.GLM(y, np.ones(df.shape[0]), family = sm.families.Binomial()).fit().deviance

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
        r2l = (base - res.deviance) / base

        # Add the r-squared value to our dictionary
        function_dict['r-squared-L'].append(r2l)
        
    # Once it's iterated through every column, turn our dictionary into a DataFrame and sort it
    function_df = pd.DataFrame(function_dict).sort_values(by=['r-squared-L'], ascending = False)

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

    df = pd.read_csv("../global/GlobalMeasures_Full_wave_enhanced_audio.csv")

    bestFeats = main(df)

# NOTE: 
    # For "Global" and "Full_wave_enhanced_audio", F0 is multicollinear with "Articulation Rate" and "iqr", so it is unincluded
# TODO 
    # Manually check for the other 4-6 model configurations
    