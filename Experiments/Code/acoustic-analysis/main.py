#!/usr/bin/env python3

# This acoustic analysis brought to you by Sonia Granlund via Yan Tang
import sys
sys.path.insert(1, './global')
sys.path.insert(1, './segmental')

import globalAnalysis
import segmentalAnalysis

def main():

    # Choose between Full_wave_enhanced_audio and Normalised_audio-chunks
    for which in ["Normalised_audio-chunks", "Full_wave_enhanced_audio"]:

        # Run Global Acoustic-Phonetic Deprecation Analysis
        globalAnalysis.main(which)

        # Run Segmental Acoustic-Phonetic Deprecation Analysis
        segmentalAnalysis.main(which)


if __name__ == "__main__":

    main()
