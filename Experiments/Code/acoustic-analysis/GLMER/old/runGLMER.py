#!/usr/bin/env python3

# https://towardsdatascience.com/how-to-run-linear-mixed-effects-models-in-python-jupyter-notebooks-4f8079c4b589
#   Rationale and sources
#   Try #1 - with statsmodels

# Load packages
from lib2to3.pgen2.pgen import DFAState
import stepwiseGLMER
import numpy as np
import pandas as pd
# from patsy import dmatrices
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

from joblib import Parallel, delayed
import multiprocessing as mp

from itertools import chain, combinations

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from stepwiseGLMER import computeVIF


def parallel_process(array, function, n_jobs=32, use_kwargs=False, front_num=3):
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

    f = "Condition ~ " + " + ".join(miniFormula)
    printList = list()

    try:
        runGLMER(df, f, printList, which, step, False)
        # return None
        # return "SUCCESS:\t\tLevel:    {}      Which:  {}      Formula:  {}".format(level, which, f)
        return ["Success", level, which, f]
    except:
        # return "FAILURE:\t\tLevel:    {}      Which:  {}      Formula:  {}".format(level, which, f)
        return ["Failure", level, which, f]


def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))


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


def runGLMER(df, formula, printList, which, step, interaction, level, segmentalModel, ttype, print = True):

    # Look out for multicollinearity first
    # Note we're doing this twice when doing forward-backward but 
    if isinstance(interaction, type(None)):
        featList = formula.split(' ~ ')[-1].split(' + ')
    else:
        featList = [x for x in formula.split(' ~ ')[-1].split(' + ') if x not in interaction]
    if len(featList) > 1:
        vifDF = computeVIF(df, featList)

        if any(vifDF.VIF > 5):

            vifDF.sort_values(by = ['VIF'])
            raise AssertionError

    # Create the two design matrices using patsy (https://www.statsmodels.org/dev/gettingstarted.html)
        # y == endog - endogenous variable(s) (i.e. dependent, response, regressand, etc.)
        # X == exog - exogenous variable(s) (i.e. independent, predictor, regressor, etc.)

    # NOTE: ID, Age and Gender are random effects because they "introduce variance into the data but are not variables of interest" (Granlund, 2012)
    # NOTE: All other variables measured are fixed effects because they "influence the mean and are of interest to the analyst" (Granlund, 2012)

    # The way to split endogenous and exogenous variables without using the formula option
    # q = df.drop(['ID', 'Condition'], axis = 1) 
    # y, X = dmatrices('Condition ~ {}'.format(" + ".join(i for i in q.columns if i in bestFeats)), data = df, return_type = 'dataframe')

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


def main(level = "segmental", ttype = 'categorical', which = "Full_wave_enhanced_audio", segmentalModel = "Phoneme_Category-vowel_erb_categories", step = False, interaction = None):

    # Load the data
    df = loadDataSet(level, which, segmentalModel)

    # Step-wise feature selection for best model by Bayes Information Criterion
    if step:
        formula = stepwiseGLMER.main(df, level, interaction)

    # Get a baseline with just everything in the model
    else:
        if level == "global":
            # For global features we've got ['Age', 'Gender', 'F0', 'iqr', 'Intensity', 'ArticulationRate', 'PausingRate']
            # Fundamental Frequency and its Interquartile Range have an interaction
            # Brute force 16; 4
            formula = "Intensity ~  Condition/MMSE (?) + Age + Gender (?)" # NOTE use other cues as random effects
            formula = "Condition ~ Intensity + ArticulationRate + PausingRate + FundFreq*iqr" 
        else:
            assert not isinstance(segmentalModel, type(None)), "You're not passing the right model for segmental GLMER."
            if segmentalModel == "Phoneme_Category-phonetic_contrasts":
                # Brute force 67108864; 26
                # formula = "Condition ~ P_VOT + B_VOT + T_VOT + D_VOT + K_VOT + G_VOT + F_COG + V_COG + S_COG + Z_COG + SH_COG + ZH_COG + TH_COG + DH_COG + IY_ERB + IH_ERB + UW_ERB + UH_ERB + AA_ERB + AE_ERB + IY_DUR + IH_DUR + UW_DUR + UH_DUR + AA_DUR + AE_DUR"
                # Items that were multicollinear:
                    # IY_ERB  43.669310
                    # IH_DUR  36.165292
                    # AA_ERB  24.235895
                    # S_COG  23.586580
                    # SH_COG  18.118203
                    # UW_ERB  17.088865
                    # IH_ERB  14.686552
                    # F_COG  13.732354
                    # AE_DUR  13.269208
                    # UH_DUR  11.127737
                    # UH_ERB  10.338288
                    # DH_COG  9.849646
                    # IY_DUR  9.393727
                    # AA_DUR  7.853067
                    # AE_ERB  7.434390
                    # K_VOT  7.090852
                    # Z_COG  6.983581
                    # D_VOT  5.061954
                formula = "Condition ~ P_VOT + B_VOT + T_VOT + G_VOT + V_COG + ZH_COG + TH_COG + UW_DUR"
            elif segmentalModel == "BCD":
                # formula = 'Condition ~ F_V_BCD + S_Z_BCD + SH_ZH_BCD + TH_DH_BCD + P_B_BCD + T_D_BCD + K_G_BCD + IY_IH_BCD_DUR + UW_UH_BCD_DUR + AA_AE_BCD_DUR + IY_IH_BCD_ERB + UW_UH_BCD_ERB + AA_AE_BCD_ERB'
                # SH_ZH_BCD  8.421546
                formula = 'Condition ~ F_V_BCD + S_Z_BCD + TH_DH_BCD + P_B_BCD + T_D_BCD + K_G_BCD + IY_IH_BCD_DUR + UW_UH_BCD_DUR + AA_AE_BCD_DUR + IY_IH_BCD_ERB + UW_UH_BCD_ERB + AA_AE_BCD_ERB'

            elif segmentalModel == "WCD":
                # formula = 'Condition ~ F_V_WCD + S_Z_WCD + SH_ZH_WCD + TH_DH_WCD + P_B_WCD + T_D_WCD + K_G_WCD + IY_IH_WCD_DUR + UW_UH_WCD_DUR + AA_AE_WCD_DUR + IY_IH_WCD_ERB + UW_UH_WCD_ERB + AA_AE_WCD_ERB'
                # AA_AE_WCD_DUR  10.920330
                # S_Z_WCD  9.186114
                # IY_IH_WCD_ERB  8.263611
                # IY_IH_WCD_DUR  6.595875
                formula = 'Condition ~ F_V_WCD + SH_ZH_WCD + TH_DH_WCD + P_B_WCD + T_D_WCD + K_G_WCD + UW_UH_WCD_DUR + UW_UH_WCD_ERB + AA_AE_WCD_ERB'
            
            elif segmentalModel == "CD":
                # TODO start here
                # formula = 'Condition ~ F_V_CD + S_Z_CD + SH_ZH_CD + TH_DH_CD + P_B_CD + T_D_CD + K_G_CD + IY_IH_CD_DUR + UW_UH_CD_DUR + AA_AE_CD_DUR + IY_IH_CD_ERB + UW_UH_CD_ERB + AA_AE_CD_ERB'
                # UW_UH_CD_DUR  1298.187928
                # UW_UH_CD_ERB  12.746253
                formula = 'Condition ~ F_V_CD + S_Z_CD + SH_ZH_CD + TH_DH_CD + P_B_CD + T_D_CD + K_G_CD + IY_IH_CD_DUR + AA_AE_CD_DUR + IY_IH_CD_ERB + AA_AE_CD_ERB'
            
            elif segmentalModel == "CO":
                # formula = 'Condition ~ F_V_CO + S_Z_CO + SH_ZH_CO + TH_DH_CO + P_B_CO + T_D_CO + K_G_CO + IY_IH_CO_DUR + UW_UH_CO_DUR + AA_AE_CO_DUR + IY_IH_CO_ERB + UW_UH_CO_ERB + AA_AE_CO_ERB'
                # IY_IH_CO_DUR  5.990306
                formula = 'Condition ~ F_V_CO + S_Z_CO + SH_ZH_CO + TH_DH_CO + P_B_CO + T_D_CO + K_G_CO + UW_UH_CO_DUR + AA_AE_CO_DUR + IY_IH_CO_ERB + UW_UH_CO_ERB + AA_AE_CO_ERB'

            elif segmentalModel == "Vowel_Space":
                # Brute force 64; 6
                # formula = "Condition ~ Vowel_Rate + Vowel_Area_2D + Vowel_Area_3D + F1_Range + F2_Range + F3_Range"
                # Items that were multicollienar:
                    # F2_Range  72.996510
                    # F3_Range  31.486032
                    # Vowel_Range_3D  7.137960
                    # Vowel_Area_2D  6.581394
                formula = "Condition ~ Vowel_Rate + F1_Range"
                
            else:
                raise AssertionError

    # Run GLMER
    printList = list()
    printList.append("Level:    {}      Which:  {}      Formula:  {}".format(level, which, formula))
    print(printList)
    runGLMER(df, formula, printList, which, step, interaction, level, segmentalModel, ttype)

    return


    # # Brute force appraoch to figuring out which formulae do/don't work
    # formulaChunks = formula.split(' + ')
    # formulaChunks[0] = formulaChunks[0].split(' ~ ')[-1]

    # # Parallelization with progress bar
    # myArr = [{"df": df, "miniFormula": miniFormula, "which": which, "step": step, "level": level} for miniFormula in all_subsets(formulaChunks[:]) if len(miniFormula) > 1]
    # failures = parallel_process(myArr, handler, use_kwargs = True)

    # # for i in failures:
    # #     # if isinstance(i, str):
    # #     print(i)

    # allDF = pd.DataFrame(failures, columns = ['Status', 'Level', 'Which', 'Formula'])
    # failuresDF = allDF[allDF['Status'] == 'Failure']

    # for i in failuresDF['Formula']:
    #     print(i)
    # print(1)

    # # Parallelization version
    # X = Parallel(n_jobs=mp.cpu_count())(delayed(handler)(df, miniFormula, which, step, level) for miniFormula in all_subsets(formulaChunks))
    # for x in X:
    #     if isinstance(x, str):
    #         print(x)

    # Iteration version
    # for miniFormula in all_subsets(formulaChunks):
    #     if len(miniFormula) > 1:

    #         f = "Condition ~ " + " + ".join(miniFormula)

    #         printList = list()
            
    #         try:
    #             runGLMER(df, f, printList, which, step, False)
    #             # print("SUCCESS:\t\tLevel:    {}      Which:  {}      Formula:  {}".format(level, which, f))
    #         except:
    #             print("FAILURE:\t\tLevel:    {}      Which:  {}      Formula:  {}".format(level, which, f))



if __name__ == "__main__":

    main()
    # main(level = "global", which = "Full_wave_enhanced_audio", segmentalModel = None, step = False)

# NOTES

            # if segmentalModel == "Phoneme_Category-fricative_categories":
                # Brute force 65536; 16
                # formula = "Condition ~ F_V_BCD + S_Z_BCD + SH_ZH_BCD + TH_DH_BCD + F_V_WCD + S_Z_WCD + SH_ZH_WCD + TH_DH_WCD + F_V_CO + S_Z_CO + SH_ZH_CO + TH_DH_CO + F_V_CD + S_Z_CD + SH_ZH_CD + TH_DH_CD"
                # Items that were multicollinear: 
                    # F_V_WCD  35.183721
                    # S_Z_WCD  28.977337
                    # SH_ZH_BCD  15.549044
                    # TH_DH_WCD  15.181886
                    # S_Z_BCD  14.930329
                    # S_Z_CO  5.951997
                # formula = "Condition ~ F_V_BCD + TH_DH_BCD + SH_ZH_WCD + F_V_CO + SH_ZH_CO + TH_DH_CO + F_V_CD + S_Z_CD + SH_ZH_CD + TH_DH_CD"


            # elif segmentalModel == "Phoneme_Category-plosive_categories":
                # Brute force 4096; 12
                # formula = "Condition ~ P_B_BCD + T_D_BCD + K_G_BCD + P_B_WCD + T_D_WCD + K_G_WCD + P_B_CO + T_D_CO + K_G_CO + P_B_CD + T_D_CD + K_G_CD"
                # Items that were multicollinear:
                    # K_G_WCD  31.074230
                    # P_B_CO  17.244600
                    # T_D_WCD  14.411502
                # formula = "Condition ~ P_B_BCD + T_D_BCD + K_G_BCD + P_B_WCD + T_D_CO + K_G_CO + P_B_CD + T_D_CD + K_G_CD"

            # elif segmentalModel == "Phoneme_Category-vowel_dur_categories":
                # Brute force 4096; 12
                # formula = "Condition ~ IY_IH_BCD_DUR + UW_UH_BCD_DUR + AA_AE_BCD_DUR + IY_IH_WCD_DUR + UW_UH_WCD_DUR + AA_AE_WCD_DUR + IY_IH_CO_DUR + UW_UH_CO_DUR + AA_AE_CO_DUR + IY_IH_CD_DUR + UW_UH_CD_DUR + AA_AE_CD_DUR"
                # Items that were multicollinear:
                    # UW_UH_WCD_DUR  36.918048
                    # IY_IH_WCD_DUR  36.602800
                    # AA_AE_WCD_DUR  16.787520
                    # IY_IH_CD_DUR  7.509463
                    # AA_AE_BCD_DUR  6.590251
                    # AA_AE_CO_DUR  6.110035
                # formula = "Condition ~ IY_IH_BCD_DUR + UW_UH_BCD_DUR + IY_IH_CO_DUR + UW_UH_CO_DUR + UW_UH_CD_DUR + AA_AE_CD_DUR"

            # elif segmentalModel == "Phoneme_Category-vowel_erb_categories":
                # Brute force 1024; 10
                # formula = "Condition ~ IY_IH_BCD_ERB + UW_UH_BCD_ERB + AA_AE_BCD_ERB + IY_IH_WCD_ERB + UW_UH_WCD_ERB + AA_AE_WCD_ERB + UW_UH_CO_ERB + AA_AE_CO_ERB + UW_UH_CD_ERB + AA_AE_CD_ERB"
                # Items that were multicollinear:
                    # UW_UH_WCD_ERB  39.920944
                    # AA_AE_WCD_ERB  33.389921
                    # IY_IH_WCD_ERB  6.117303
                # formula = "Condition ~ IY_IH_BCD_ERB + UW_UH_BCD_ERB + AA_AE_BCD_ERB + UW_UH_CO_ERB + AA_AE_CO_ERB + UW_UH_CD_ERB + AA_AE_CD_ERB"
