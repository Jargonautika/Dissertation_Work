#!/usr/bin/env python3

# https://towardsdatascience.com/how-to-run-linear-mixed-effects-models-in-python-jupyter-notebooks-4f8079c4b589
#   Rationale and sources
#   Try #1 - with statsmodels

# Load packages
import warnings

import stepwiseLMER
import numpy as np
import pandas as pd
# from patsy import dmatrices
import statsmodels.formula.api as smf
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

from joblib import Parallel, delayed
import multiprocessing as mp

from itertools import chain, combinations


from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def parallel_process(array, function, n_jobs=32, use_kwargs=False, front_num=5):
    """
        A parallel version of the map function with a progress bar. 

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
                keyword arguments to function 
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job. 
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures. 
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out


def handler(df, miniFormula, which, step, level):

    if len(miniFormula) > 1:

        f = "MMSE ~ " + " + ".join(miniFormula)
        printList = list()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                runLMER(df, f, printList, which, step, False)
                # return None
                return "SUCCESS:\t\tLevel:    {}      Which:  {}      Formula:  {}".format(level, which, f)
            except:
                return "FAILURE:\t\tLevel:    {}      Which:  {}      Formula:  {}".format(level, which, f)


def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))


def imputeMissingData(df):

    # Separate the data into the categories
    fifteen     = df[df['MMSE'] < 15.0]
    twenty      = df[(df['MMSE'] >= 15.0) & (df['MMSE'] < 20.0)]
    twentyfive  = df[(df['MMSE'] >= 20.0) & (df['MMSE'] < 25.0)]
    thirty      = df[df['MMSE'] >= 25.0]

    dfList = list()
    for conditionDF in [fifteen, twenty, twentyfive, thirty]:
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
                # Find the completely NaN columns for this range of MMSE scores
                if j:
                    # Instantiate a single factor SimpleImputer
                    imp1 = SimpleImputer(missing_values=np.nan, strategy='mean')
                    # Fit the imputer to ALL of the data (irrespective of MMSE score) for that factor
                    imp1.fit(df[valuesDF.columns[i]].to_numpy().reshape(-1, 1))
                    # Fill a column with the mean value from the data for this range of MMSE scores
                    vec = imp1.transform(valuesDF[valuesDF.columns[i]].to_numpy().reshape(-1, 1))
                    # Set this new column of mean values into the values frame at hand
                    valuesDF[valuesDF.columns[i]] = vec

        imputedDF = pd.DataFrame(imp.fit_transform(valuesDF), columns = valuesDF.columns)
        imputedDF.index = metaDF.index

        synthDF = metaDF.join(imputedDF)
        synthDF['MMSE'] = conditionDF['MMSE']

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


def runLMER(df, formula, printList, which, step, level, segmentalModel, ttype, print = True):

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
    if print:
        if step:
            run = "BIC"
        else:
            run = "ALL"
        if not isinstance(segmentalModel, type(None)):
            out = '../../../Results/03_Acoustic_Analysis/{}/{}/{}-{}-{}.txt'.format(level, ttype, run, which, segmentalModel)
        else:
            out = '../../../Results/03_Acoustic_Analysis/{}/{}/{}-{}.txt'.format(level, ttype, run, which)
        with open(out, 'w') as f:
            for item in printList:
                f.write("%s\n" % item)


def loadDataSet(level, which, segmentalModel):

    if level == "global":
        df = pd.read_csv("./{}/GlobalMeasures_{}-numerical.csv".format(level, which))
    elif level == "segmental":
        # To avoid multicollinearity issues, only compare all Between-Category-Distance measures at a time 
        # (since SH_ZH_BCD and SH_ZH_WCD are going to correlate; they're only mathematical functions on top of the numbers)
        if segmentalModel in ["BCD", "WCD", "CO", "CD"]:
            dfList = [pd.read_csv('./{}/SegmentalMeasures_{}-numerical-Phoneme_Category-{}_categories.csv'.format(level, which, feature)) for feature in ['fricative', 'plosive', 'vowel_dur', 'vowel_erb']]
            df = dfList[0]
            for i in dfList[1:]:
                df = pd.merge(df, i, on = ['ID', 'Age', 'Gender', 'MMSE'])
            df = df[['ID', 'Age', 'Gender'] + [col for col in df.columns if segmentalModel in col] + ['MMSE']]
        else:
            df = pd.read_csv("./{}/SegmentalMeasures_{}-numerical-{}.csv".format(level, which, segmentalModel))
    else:
        raise AssertionError

    if df.isnull().sum().sum() > 0:
        df = imputeMissingData(df)

    return fixCols(df)


def main(level = "segmental", ttype = 'categorical', which = "Full_wave_enhanced_audio", segmentalModel = "Phoneme_Category-vowel_dur_categories", step = False, interaction = None):

    # Load the data
    df = loadDataSet(level, which, segmentalModel)

    # Step-wise feature selection for best model by Bayes Information Criterion
    if step:
        formula = stepwiseLMER.main(df, interaction)

    # Get a baseline with just everything in the model
    else:
        if level == "global":
            # For global features we've got ['Age', 'Gender', 'F0', 'iqr', 'Intensity', 'ArticulationRate', 'PausingRate']
            # Fundamental Frequency and its Interquartile Range have an interaction
            formula = "MMSE ~ Intensity + ArticulationRate + PausingRate + FundFreq*iqr"
        else:
            assert not isinstance(segmentalModel, type(None)), "You're not passing the right model for segmental LMER."
            if segmentalModel == "Vowel_Space":
                formula = "MMSE ~ Vowel_Rate + Vowel_Area_2D + Vowel_Area_3D + F1_Range + F2_Range + F3_Range"
            elif segmentalModel == "Phoneme_Category-phonetic_contrasts":
                formula = "MMSE ~ P_VOT + B_VOT + T_VOT + D_VOT + K_VOT + G_VOT + F_COG + V_COG + S_COG + Z_COG + SH_COG + ZH_COG + TH_COG + DH_COG + IY_ERB + IH_ERB + UW_ERB + UH_ERB + AA_ERB + AE_ERB + IY_DUR + IH_DUR + UW_DUR + UH_DUR + AA_DUR + AE_DUR"
            elif segmentalModel == "BCD":
                formula = "MMSE ~ " + " + ".join(col for col in df if 'BCD' in col)
            elif segmentalModel == "WCD":
                formula = "MMSE ~ " + " + ".join(col for col in df if 'WCD' in col)
            elif segmentalModel == "CD":
                formula = "MMSE ~ " + " + ".join(col for col in df if '_CD' in col)
            elif segmentalModel == "CO":
                formula = "MMSE ~ " + " + ".join(col for col in df if 'CO' in col)             
            else:
                raise AssertionError

    # Run LMER
    printList = list()
    printList.append("Level:    {}      Which:  {}      Formula:  {}".format(level, which, formula))
    print(printList)
    runLMER(df, formula, printList, which, step, level, segmentalModel, ttype)
    return

    # Brute force appraoch to figuring out which formulae do/don't work
    formulaChunks = formula.split(' + ')
    formulaChunks[0] = formulaChunks[0].split(' ~ ')[-1]

    # Parallelization with progress bar
    myArr = [{"df": df, "miniFormula": miniFormula, "which": which, "step": step, "level": level} for miniFormula in all_subsets(formulaChunks[:])]
    failures = parallel_process(myArr, handler, use_kwargs = True)

    for i in failures:
        # if isinstance(i, str):
        print(i)

    # # Parallelization version
    # X = Parallel(n_jobs=mp.cpu_count())(delayed(handler)(df, miniFormula, which, step, level) for miniFormula in all_subsets(formulaChunks))
    # for x in X:
    #     if isinstance(x, str):
    #         print(x)

    # Iteration version
    # for miniFormula in all_subsets(formulaChunks):
    #     if len(miniFormula) > 1:

    #         f = "MMSE ~ " + " + ".join(miniFormula)

    #         printList = list()
            
    #         try:
    #             runLMER(df, f, printList, which, step, False)
    #             # print("SUCCESS:\t\tLevel:    {}      Which:  {}      Formula:  {}".format(level, which, f))
    #         except:
    #             print("FAILURE:\t\tLevel:    {}      Which:  {}      Formula:  {}".format(level, which, f))


if __name__ == "__main__":

    main()
    # main(level = "global", which = "Normalised_audio-chunks", step = True)
