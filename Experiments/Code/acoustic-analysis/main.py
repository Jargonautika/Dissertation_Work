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
def segmentalStuff(which, task):

    # Run Segmental Acoustic-Phonetic Deprecation Analysis
    segmentalAnalysis.main(which)


# Do the global work
def globalStuff(which, task):

    # Run Global Acoustic-Phonetic Deprecation Analysis
    globalAnalysis.main(which, task)

    # Run Generalized Mixed Effects Models (categorical: 'cc' vs 'cd')
    if task == "categorical":
        runGLMER.main(level = 'global', which = which, step = True) # Use BIC stepwise feature selection
        runGLMER.main(level = 'global', which = which, step = False) # Model all measurements

    # Run Linear Mixed Effects Models (numerical: MMSE 0 - 30)
    elif task == "numerical":
        runLMER.main(level = 'global', which = which, step = True) # Use BIC stepwise feature selection
        runLMER.main(level = 'global', which = which, step = False) # Model all measurements


def main():

    # Choose between Full_wave_enhanced_audio and Normalised_audio-chunks
    for which in ["Normalised_audio-chunks", "Full_wave_enhanced_audio"]:

        for task in ["categorical", "numerical"]:

            globalStuff(which, task)

            segmentalStuff(which, task)


if __name__ == "__main__":

    main()
