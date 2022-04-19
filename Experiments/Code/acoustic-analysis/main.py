#!/usr/bin/env python3

# This acoustic analysis brought to you by Sonia Granlund via Yan Tang
import sys
import glob
import numpy as np
import pandas as pd
sys.path.insert(1, './LMER')
sys.path.insert(1, './GLMER')
sys.path.insert(1, './global')
sys.path.insert(1, './segmental')

import runLMER
import runGLMER
import globalAnalysis
import segmentalAnalysis


def fix():

    catDF = pd.read_csv("segmental/SegmentalMeasures_Full_wave_enhanced_audio-categorical-Vowel_Space.csv")
    if len(list(set(catDF['Condition']))) > 2:
        catDF['Condition'] = catDF['Condition'].replace(np.nan, 'cc') # That one NaN guy
        catDF.to_csv("segmental/SegmentalMeasures_Full_wave_enhanced_audio-categorical-Vowel_Space.csv", index = None)
    csvList = glob.glob("segmental/*categorical-P*.csv")
    badDFs = [pd.read_csv(csv) for csv in csvList]

    numSpeakers = catDF.shape[0]
    conditionList = catDF['Condition'].tolist()
    for df, csv in zip(badDFs, csvList):
        assert df.shape[0] == numSpeakers, "We have a mismatched number of speakers in {}".format(csv)

        df['Condition'] = conditionList
        if 'Unnamed: 0' in df.columns:
            df.drop('Unnamed: 0', axis = 1, inplace = True)
        df.to_csv(csv, index = None)
        

# Do the global work
def segmentalStuff(which):

    # Run Segmental Acoustic-Phonetic Deprecation Analysis
    # Do the Vowel Space Degradation model
    # segmentalAnalysis.main(which, True)

    # # Do the Phonemic Contrast Degradation Models
    # segmentalAnalysis.main(which, False)

    for segmentalModel in [#"Phoneme_Category-phonetic_contrasts",
                            "Vowel_Space",
                            "BCD",
                            "WCD",
                            "CD",
                            "CO"]:

        # Run Generalized Mixed Effects Models (categorical: 'cc' vs 'cd')
        # runGLMER.main(level = 'segmental', ttype = 'categorical', which = which, segmentalModel = segmentalModel, step = True)     # Use BIC stepwise feature selection

        # Run Linear Mixed Effects Models (numerical: MMSE 0 - 30)
        runLMER.main(level = 'segmental', ttype = 'numerical', which = which, segmentalModel = segmentalModel, step = True) # Use BIC stepwise feature selection


# Do the global work
def globalStuff(which):

    # # Run Global Acoustic-Phonetic Deprecation Analysis
    globalAnalysis.main(which)
    # longitudinalAnalysis.main(which) # TODO use the same protocol for global feature extraction on the 2021 data as with the 2020 data
    # We should just call the functions from globalAnalysis, though, to make sure it's consistent. 

    # Run Generalized Mixed Effects Models (categorical: 'cc' vs 'cd')
    # runGLMER.main(level = 'global', ttype = 'categorical', which = which, segmentalModel = None, step = True) # Use BIC stepwise feature selection

    # Run Linear Mixed Effects Models (numerical: MMSE 0 - 30)
    runLMER.main(level = 'global', ttype = 'numerical', which = which, segmentalModel = None, step = True) # Use BIC stepwise feature selection

    # TODO it's unclear what the labels look like (should be deltas)
    # runLMER.main(level = 'global', ttype = 'longitudinal', which = which, segmentalModel = None, step = True) # Use BIC stepwise feature selection


def main():

    # The analysis is actually the exact same if we're doing Norm or Full here. 
    for which in ["Full_wave_enhanced_audio"]: 

        globalStuff(which)
        return

        segmentalStuff(which)

    # Something isn't right with the Categorical vs Numerical stuff so this is a hack to fix it 
    # after the fact
    # fix()


if __name__ == "__main__":

    main()
