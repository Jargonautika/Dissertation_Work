#!/usr/bin/env python3

# Follow the instructions / rationale here:
    # https://medium.com/@garrettwilliams90/stepwise-feature-selection-for-statsmodels-fda269442556

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit(formula, data, groups, returnRes = True, returnBIC = True, generalized = True):

    if generalized:
        # Fit a model for our target and our selected column 
        glm_binom = smf.glm(formula = formula, data = data, groups = groups, family = sm.families.Binomial())

        # Fit the model and get the results
        res = glm_binom.fit()

        # Calculate the BIC
        bic = res.bic_llf

    else:

        # Construct the model
        md = smf.mixedlm(formula = formula, data = data, groups = groups)

        # Fit the model
        res = md.fit(method = ['lbfgs'])

        # Calculate the BIC
        bic = -2 * res.llf + np.log(res.nobs) * (res.df_modelwc)

    if returnRes and returnBIC:
        return res, bic

    if returnRes:
        return res

    if returnBIC:
        return bic


def finalCheck(df, bestFormula, best, response_variable, fixed_variables, random_variable, random_intercept, generalized):

    formula = "{} ~ {} + (1|{})".format(response_variable, ' + '.join(fixed_variables), random_variable[0])

    bic = fit(formula, df, df[random_intercept[0]], returnRes = False, generalized = generalized)

    if bic < best:
        return formula, bic

    else:
        return bestFormula, best


def potentiallyRemoveFeature(df, oldFormula, predictors, base, response_variable, random_intercept, generalized = True):

    predictors_to_keep = list()
    # Iterate through every column of interest for this run
    for predictor in predictors:

        formula = "{} ~ {}".format(response_variable, predictor)

        bic = fit(formula, df, df[random_intercept[0]], returnRes = False, generalized = generalized)

        # predictor is better alone than in conjunction with the others
        if bic < base:
            predictors_to_keep.append(predictor)

    if len(predictors_to_keep) == 0:

        return oldFormula, predictors, base

    else:

        formula = "{} ~ {}".format(response_variable, ' + '.join(predictors_to_keep))

        current_best_bic = fit(formula, df, df[random_intercept[0]], returnRes = False)

        return formula, predictors_to_keep, current_best_bic


def next_possible_feature(df, oldFormula, newFormula, current_features, response_variable, var, random_intercept, base = 0.0, generalized = True):

    bic = fit(newFormula, df, df[random_intercept[0]], returnRes = False, generalized = generalized)

    # The model is improved by adding the feature
    if bic < base:

        current_features = current_features + [var]

        # Do a top-down backward pass to see if removing any of the variables improves the model
        newFormula, current_features, bic = potentiallyRemoveFeature(df, newFormula, current_features, bic, response_variable, random_intercept, generalized = generalized)

        return newFormula, current_features, bic
    
    # The model is not improved by adding the feature
    else:
        return oldFormula, current_features, base


def bottomUp(df, bestFormula, best, base, response_variable, fixed_variables, random_variable, random_intercept, generalized):

    # Start with the first predictor, and see all possibilities for our second.
    selected_features = [best]

    for var in fixed_variables + random_variable:

        if var in selected_features:
            continue

        if var in random_variable:
            var = "(1|{})".format(var)

        formula = bestFormula + " + {}".format(var)

        # Add a new variable to the best performing column
        currentFormula, current_features, currentBase = next_possible_feature(df, bestFormula, formula, selected_features, response_variable, var, random_intercept, base = base, generalized = generalized)

        # We've added something and therefore need to update things
        if currentFormula != bestFormula: 

            bestFormula = currentFormula 
            selected_features = current_features
            base = currentBase

    return bestFormula, base


def firstPass(df, response_variable, fixed_variables, random_variable, random_intercept, generalized):

    #Create an empty dictionary that will be used to store our results
    function_dict = {'predictor': [], 'formula': [], 'BIC':[]}

    for fixedVar in fixed_variables:

        # Add the column name to our dictionary
        function_dict['predictor'].append(fixedVar)

        # Create a formula for testing the factor by itself
        formula = "{} ~ {}".format(response_variable, fixedVar)

        # Add the formula to our dictionary
        function_dict['formula'].append(formula)

        bic = fit(formula, df, df[random_intercept[0]], returnRes = False, generalized = generalized)

        # Add the r-squared value to our dictionary
        function_dict['BIC'].append(bic)

    for randVar in random_variable:

        # Add the column name to our dictionary
        function_dict['predictor'].append(randVar)

        # Create a formula for testing the factor by itself
        formula = "{} ~ (1|{})".format(response_variable, randVar)

        # Add the formula to our dictionary
        function_dict['formula'].append(formula)

        bic = fit(formula, df, df[random_intercept[0]], returnRes = False, generalized = generalized)

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
    

def main(df, response_variable, fixed_variables, random_variable, random_intercept, generalized = True):

    # First pass
    best, bestFormula, base = firstPass(df, response_variable, fixed_variables, random_variable, random_intercept, generalized)

    # Bottom-up, hierarchical approach to feature selection
    bestFormula, best = bottomUp(df, bestFormula, best, base, response_variable, fixed_variables, random_variable, random_intercept, generalized)

    # Final check to make sure everything isn't just inherently better
    bestFormula, best = finalCheck(df, bestFormula, best, response_variable, fixed_variables, random_variable, random_intercept, generalized)

    return bestFormula


if __name__ == "__main__":

    df = pd.read_csv("../global/GlobalMeasures_Full_wave_enhanced_audio-categorical.csv")

    bestFormula, base = main(df)
    