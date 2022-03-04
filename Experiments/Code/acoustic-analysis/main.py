#!/usr/bin/env python3

# This acoustic analysis brought to you by Sonia Granlund via Yan Tang
import sys
sys.path.insert(1, './LMER')
sys.path.insert(1, './GLMER')
sys.path.insert(1, './global')
sys.path.insert(1, './segmental')

import runLMER
import runGLMER
import globalAnalysis
import segmentalAnalysis


# Do the global work
def segmentalStuff(which):

    # # Run Segmental Acoustic-Phonetic Deprecation Analysis
    # # Do the Vowel Space Degradation model
    # segmentalAnalysis.main(which, True)

    # Do the Phonemic Contrast Degradation Models
    segmentalAnalysis.main(which, False)

    # for segmentalModel in ["Phoneme_Category-fricative_categories",
    #                         "Phoneme_Category-phonetic_contrasts",
    #                         "Phoneme_Category-plosive_categories",
    #                         "Phoneme_Category-vowel_dur_categories",
    #                         "Phoneme_Category-vowel_erb_categories",
    #                         "Vowel_Space"]:

    #     # Run Generalized Mixed Effects Models (categorical: 'cc' vs 'cd')
    #     if task == "categorical":
    #             # runGLMER.main(level = 'segmental', which = which, segmentalModel = segmentalModel, step = True)     # Use BIC stepwise feature selection
    #             runGLMER.main(level = 'segmental', which = which, segmentalModel = segmentalModel, step = False)    # Model all measurements

    #     # Run Linear Mixed Effects Models (numerical: MMSE 0 - 30)
    #     elif task == "numerical":
    #         # runLMER.main(level = 'segmental', which = which, segmentalModel = segmentalModel, step = True) # Use BIC stepwise feature selection
    #         runLMER.main(level = 'segmental', which = which, segmentalModel = segmentalModel, step = False) # Model all measurements


# Do the global work
def globalStuff(which):

    # Run Global Acoustic-Phonetic Deprecation Analysis
    globalAnalysis.main(which)

    # # Run Generalized Mixed Effects Models (categorical: 'cc' vs 'cd')
    # if task == "categorical":
    #     runGLMER.main(level = 'global', which = which, segmentalModel = None, step = True) # Use BIC stepwise feature selection
    #     runGLMER.main(level = 'global', which = which, segmentalModel = None, step = False) # Model all measurements

    # # Run Linear Mixed Effects Models (numerical: MMSE 0 - 30)
    # elif task == "numerical":
    #     runLMER.main(level = 'global', which = which, segmentalModel = None, step = True) # Use BIC stepwise feature selection
    #     runLMER.main(level = 'global', which = which, segmentalModel = None, step = False) # Model all measurements


def main():

    # Choose between Full_wave_enhanced_audio and Normalised_audio-chunks
    # for which in ["Normalised_audio-chunks", "Full_wave_enhanced_audio"]:
    for which in ["Full_wave_enhanced_audio"]: # The analysis is actually the exact same if we're doing Norm or Full here. 

        # globalStuff(which)

        segmentalStuff(which)


if __name__ == "__main__":

    main()
