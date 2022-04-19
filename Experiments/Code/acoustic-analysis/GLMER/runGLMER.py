#!/usr/bin/env python3

# https://towardsdatascience.com/how-to-run-linear-mixed-effects-models-in-python-jupyter-notebooks-4f8079c4b589
#   Rationale and sources
#   Try #1 - with statsmodels

# Load packages
import stepwise
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer


def runGLMER(df, formula, printList, which, step, level, segmentalModel, ttype, random_intercept, response_variable, print = True):

    # Construct the model
    # glm_binom = smf.glm(formula = formula, data = df, groups = df[random_intercept[0]], family = sm.families.Binomial())
    md = smf.mixedlm(formula = formula, data = df, groups = df[random_intercept[0]])

    # Fit the model
    # res = glm_binom.fit()
    res = md.fit(method = ['lbfgs'])

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
    if print:
        if step:
            run = "BIC"
        else:
            run = "ALL"
        if not isinstance(segmentalModel, type(None)):
            out = '../../../Results/03_Acoustic_Analysis/{}/{}/{}-{}-{}-{}.txt'.format(level, ttype, response_variable, run, which, segmentalModel)
        else:
            out = '../../../Results/03_Acoustic_Analysis/{}/{}/{}-{}-{}.txt'.format(level, ttype, response_variable, run, which)
        with open(out, 'w') as f:
            for item in printList:
                f.write("%s\n" % item)


def imputeMissingData(df):

    # Separate the data into the categories
    CC = df[df['Condition'] == 'cc']
    CD = df[df['Condition'] == 'cd']

    dfList = list()
    for conditionDF in [CC, CD]:
        metaDF = conditionDF.loc[:, df.columns[0]:df.columns[2]]
        valuesDF = conditionDF.loc[:, df.columns[3]:df.columns[-2]]

        imp = IterativeImputer(max_iter=10, random_state=42)
        valuesDF.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Sometimes we have a whole column of nothing
        nothingList = [valuesDF[mys].isnull().all() for mys in valuesDF]
        # If there are any columns with no observations
        if any(nothingList):
            # Iterate over the columns
            for i, j in enumerate(nothingList):
                # Find the completely NaN columns for category
                if j:
                    # Instantiate a single factor SimpleImputer
                    imp1 = SimpleImputer(missing_values=np.nan, strategy='mean')
                    # Fit the imputer to ALL of the data (irrespective of category) for that factor
                    imp1.fit(df[valuesDF.columns[i]].to_numpy().reshape(-1, 1))
                    # Fill a column with the mean value from the data for this category
                    vec = imp1.transform(valuesDF[valuesDF.columns[i]].to_numpy().reshape(-1, 1))
                    # Set this new column of mean values into the values frame at hand
                    valuesDF[valuesDF.columns[i]] = vec

        imputedDF = pd.DataFrame(imp.fit_transform(valuesDF), columns = valuesDF.columns)
        imputedDF.index = metaDF.index

        synthDF = metaDF.join(imputedDF)
        synthDF['Condition'] = conditionDF['Condition']

        dfList.append(synthDF)

    result = pd.concat(dfList)
    assert result.shape == df.shape, "You've lost data somehow."

    return result


# Patsy doesn't accept numerals in pandas DataFrame column strings
def fixCols(df):

    colList = df.columns.tolist()
    if 'F0' in colList:
        i = colList.index('F0')
        colList[i] = 'FundFreq'

        df.columns = colList
        
    return df


def loadDataSet(level, which, segmentalModel):

    if level == "global":
        df = pd.read_csv("./{}/GlobalMeasures_{}-categorical.csv".format(level, which))
    elif level == "segmental":
        # To avoid multicollinearity issues, only compare all Between-Category-Distance measures at a time 
        # (since SH_ZH_BCD and SH_ZH_WCD are going to correlate; they're only mathematical functions on top of the numbers)
        if segmentalModel in ["BCD", "WCD", "CO", "CD"]:
            dfList = [pd.read_csv('./{}/SegmentalMeasures_{}-categorical-Phoneme_Category-{}_categories.csv'.format(level, which, feature)) for feature in ['fricative', 'plosive', 'vowel_dur', 'vowel_erb']]
            df = dfList[0]
            for i in dfList[1:]:
                df = pd.merge(df, i, on = ['ID', 'Age', 'Gender', 'Condition'])
            df = df[['ID', 'Age', 'Gender'] + [col for col in df.columns if segmentalModel in col] + ['Condition']]
        else:
            df = pd.read_csv("./{}/SegmentalMeasures_{}-categorical-{}.csv".format(level, which, segmentalModel))
    else:
        raise AssertionError

    df = fixCols(df)

    # We get exact separation in the glm models if the 'cd' condition has NaN values resulting in a (-)inf columnwise mean for any particular factor
    if df.isnull().sum().sum() > 0:
        df = imputeMissingData(df)

    return df


def main(level = "segmental", ttype = 'categorical', which = "Full_wave_enhanced_audio", segmentalModel = "Phoneme_Category-vowel_erb_categories", step = False):

    # Load the data
    df = loadDataSet(level, which, segmentalModel)

    # http://web.pdx.edu/~newsomj/mlrclass/ho_randfixd.pdf
    fixed_variables = ['Condition', 'Gender']
    random_variable = ['Age']
    random_intercept = ['ID']

    response_variables = [x for x in df.columns if x not in fixed_variables and x not in random_variable and x not in random_intercept]
    for response_variable in response_variables:

        # Step-wise feature selection for best model by Bayes Information Criterion
        formula = stepwise.main(df, response_variable, fixed_variables, random_variable, random_intercept, generalized = True)

        # Run GLMER with BIC
        printList = list()
        printList.append("Level:    {}      Which:  {}      Formula:  {}".format(level, which, formula))
        print(printList)
        runGLMER(df, formula, printList, which, step, level, segmentalModel, ttype, random_intercept, response_variable)

        # Run GLMER with ALL
        printList = list()
        printList.append("Level:    {}      Which:  {}      Formula:  {}".format(level, which, formula))
        formula = "{} ~ {} + (1|{})".format(response_variable, ' + '.join(fixed_variables), random_variable[0])
        runGLMER(df, formula, printList, which, False, level, segmentalModel, ttype, random_intercept, response_variable)


if __name__ == "__main__":

    main()
    # main(level = "global", which = "Full_wave_enhanced_audio", segmentalModel = None, step = False)
