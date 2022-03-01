#!/usr/bin/env python3

# https://towardsdatascience.com/how-to-run-linear-mixed-effects-models-in-python-jupyter-notebooks-4f8079c4b589
#   Rationale and sources
#   Try #1 - with statsmodels

# Load packages
import stepwiseLMER
import numpy as np
import pandas as pd
# from patsy import dmatrices
import statsmodels.formula.api as smf
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer


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


def runLMER(df, formula, printList, which, step):

    df = fixCols(df)

    if df.isnull().sum().sum() > 0:
        df = imputeMissingData(df)

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
    if step:
        run = "BIC"
    else:
        run = "ALL"
    with open('../../../Results/03_Acoustic_Analysis/global/numerical/{}-{}.txt'.format(run, which), 'w') as f:
        for item in printList:
            f.write("%s\n" % item)


def loadDataSet(level, which, segmentalModel):

    if level == "global":
        df = pd.read_csv("./{}/GlobalMeasures_{}-numerical.csv".format(level, which))
    else:
        df = pd.read_csv("./{}/SegmentalMeasures_{}-numerical-{}.csv".format(level, which, segmentalModel))

    return df


def main(level = "segmental", which = "Full_wave_enhanced_audio", segmentalModel = "Phoneme_Category-fricative_categories", step = False):

    # Load the data
    df = loadDataSet(level, which, segmentalModel)

    # Step-wise feature selection for best model by Bayes Information Criterion
    if step:
        formula = stepwiseLMER.main(df)

    # Get a baseline with just everything in the model
    else:
        if level == "global":
            # For global features we've got ['Age', 'Gender', 'F0', 'iqr', 'Intensity', 'ArticulationRate', 'PausingRate']
            # Fundamental Frequency and its Interquartile Range have an interaction
            formula = "MMSE ~ Intensity + ArticulationRate + PausingRate + FundFreq*iqr"
        else:
            assert not isinstance(segmentalModel, type(None)), "You're not passing the right model for segmental GLMER."
            if segmentalModel == "Phoneme_Category-fricative_categories":
                formula = "MMSE ~ F_V_BCD + S_Z_BCD + SH_ZH_BCD + TH_DH_BCD + F_V_WCD + S_Z_WCD + SH_ZH_WCD + TH_DH_WCD + F_V_CO + S_Z_CO + SH_ZH_CO + TH_DH_CO + F_V_CD + S_Z_CD + SH_ZH_CD + TH_DH_CD"
            elif segmentalModel == "Phoneme_Category-phonetic_contrasts":
                formula = "MMSE ~ P_VOT + B_VOT + T_VOT + D_VOT + K_VOT + G_VOT + F_COG + V_COG + S_COG + Z_COG + SH_COG + ZH_COG + TH_COG + DH_COG + IY_ERB + IH_ERB + UW_ERB + UH_ERB + AA_ERB + AE_ERB + IY_DUR + IH_DUR + UW_DUR + UH_DUR + AA_DUR + AE_DUR"
            elif segmentalModel == "Phoneme_Category-plosive_categories":
                formula = "MMSE ~ P_B_BCD + T_D_BCD + K_G_BCD + P_B_WCD + T_D_WCD + K_G_WCD + P_B_CO + T_D_CO + K_G_CO + P_B_CD + T_D_CD + K_G_CD"
            elif segmentalModel == "Phoneme_Category-vowel_dur_categories":
                formula = "MMSE ~ IY_IH_BCD_DUR + UW_UH_BCD_DUR + AA_AE_BCD_DUR + IY_IH_WCD_DUR + UW_UH_WCD_DUR + AA_AE_WCD_DUR + IY_IH_CO_DUR + UW_UH_CO_DUR + AA_AE_CO_DUR + IY_IH_CD_DUR + UW_UH_CD_DUR + AA_AE_CD_DUR"
            elif segmentalModel == "Phoneme_Category-vowel_erb_categories":
                formula = "MMSE ~ IY_IH_BCD_ERB + UW_UH_BCD_ERB + AA_AE_BCD_ERB + IY_IH_WCD_ERB + UW_UH_WCD_ERB + AA_AE_WCD_ERB + IY_IH_CO_ERB + UW_UH_CO_ERB + AA_AE_CO_ERB + IY_IH_CD_ERB + UW_UH_CD_ERB + AA_AE_CD_ERB"
            elif segmentalModel == "Vowel_Space":
                formula = "Vowel_Rate + Vowel_Area_2D + Vowel_Area_3D + F1_Range + F2_Range + F3_Range"
                               
            else:
                raise AssertionError

    # Run LMER
    printList = list()
    printList.append("Level:    {}      Which:  {}      Formula:  {}".format(level, which, formula))
    runLMER(df, formula, printList, which, step)


if __name__ == "__main__":

    main()
    # main(level = "global", which = "Normalised_audio-chunks", step = True)
