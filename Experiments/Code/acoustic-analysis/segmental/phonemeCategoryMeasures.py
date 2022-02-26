#!/usr/bin/env python3

import os
import string
import parselmouth
import numpy as np
from scipy.signal import lfilter
from scipy.signal import find_peaks
from vowelSpaceMeasures import locateTextGrid, removePhonesFromInterviewer, getVocalicFormants

import sys
sys.path.append('./global')

from globalAnalysis import filterbank


def categoryDiscriminability(first: list, second: list):

    return (np.nanmean(first) - np.nanmean(second)) * np.sqrt(2) / np.sqrt((np.std(first) ** 2) + (np.std(second) ** 2))


def categoryOverlap(first: list, second: list):

    if len(first) > 0 and len(second) > 0:
        return np.max(second) - np.min(first)
    else:
        return 0 # Sometimes we don't have a particular phoneme produced for a speaker


def withinCategoryDispersion(first: list, second: list):

    return np.nanmean([np.std(first), np.std(second)])


def betweenCategoryDistance(first: list, second: list):

    return np.nanmean(first) - np.nanmean(second)


def distinctions(plosives, fricatives, vowelsERB, vowelsDur):

    plosivePairs = [('P', 'B'), ('T', 'D'), ('K', 'G')]                     # Voiceless and then voiced
    fricativePairs = [('F', 'V'), ('S', 'Z'), ('SH', 'ZH'), ('TH', 'DH')]   # Voiceless and then voiced
    vowelPairs = [('IY', 'IH'), ('UW', 'UH'), ('AA', 'AE')]                 # Tense and then lax

    plosiveBCD, plosiveWCD, plosiveCO, plosiveCD = list(), list(), list(), list()
    for voiceless, voiced in plosivePairs:
        first, second = plosives[voiceless], plosives[voiced]
        # Calculate Between-Category Distance
        plosiveBCD.append(betweenCategoryDistance(first, second))
        # Calculate Within-Category Dispersion
        plosiveWCD.append(withinCategoryDispersion(first, second))
        # Calculate Category Overlap
        plosiveCO.append(categoryOverlap(first, second))
        # Calculate Category Discriminability
        plosiveCD.append(categoryDiscriminability(first, second))

    fricativeBCD, fricativeWCD, fricativeCO, fricativeCD = list(), list(), list(), list()
    for voiceless, voiced in fricativePairs:
        first, second = fricatives[voiceless], fricatives[voiced]
        fricativeBCD.append(betweenCategoryDistance(first, second))
        fricativeWCD.append(withinCategoryDispersion(first, second))
        fricativeCO.append(categoryOverlap(first, second))
        fricativeCD.append(categoryDiscriminability(first, second))

    vowelERBBCD, vowelERBWCD, vowelERBCO, vowelERBCD = list(), list(), list(), list()
    for tense, lax in vowelPairs:
        first, second = vowelsERB[tense], vowelsERB[lax]
        vowelERBBCD.append(betweenCategoryDistance(first, second))
        vowelERBWCD.append(withinCategoryDispersion(first, second))
        vowelERBCO.append(categoryOverlap(first, second))
        vowelERBCD.append(categoryDiscriminability(first, second))

    vowelDurBCD, vowelDurWCD, vowelDurCO, vowelDurCD = list(), list(), list(), list()
    for tense, lax in vowelPairs:
        first, second = vowelsDur[tense], vowelsDur[lax]
        vowelDurBCD.append(betweenCategoryDistance(first, second))
        vowelDurWCD.append(withinCategoryDispersion(first, second))
        vowelDurCO.append(categoryOverlap(first, second))
        vowelDurCD.append(categoryDiscriminability(first, second))

    return [plosiveBCD, plosiveWCD, plosiveCO, plosiveCD], \
           [fricativeBCD, fricativeWCD, fricativeCO, fricativeCD], \
           [vowelERBBCD, vowelERBWCD, vowelERBCO, vowelERBCD], \
           [vowelDurBCD, vowelDurWCD, vowelDurCO, vowelDurCD]


# ERB means "Equivalent retangular band(-width)"
def herzToERB(frequency):

    # Constants:
    _ERB_L = 24.7
    _ERB_Q = 9.265

    return _ERB_Q * np.log(1 + frequency / (_ERB_L * _ERB_Q))


def getParticularVowels(basename, particulars):

    tg = locateTextGrid(basename)

    startWord, isSegment = False, False
    startData = list()
    counter = 0
    totalDur = 0.0
    a, b = 0.0, 0.0
    vowels = dict()
    # We're dealing with a normalized file
    if "-" in basename:
        _, _, startSIL, _, _, offsetA, offsetB = basename.split('_')[0].split('-')
        startPoint = int(startSIL)
        endPoint = int(startSIL) + int(offsetB)
    # We're dealing with a full file
    else:
        startPoint = None
        endPoint = None
        isSegment = True

    with open(tg, 'r') as f:
        lastLine = ""
        for line in f:
            if line == '"IntervalTier"\n' and b == totalDur:
                if "-" in basename:
                    return vowels
                else:
                    return removePhonesFromInterviewer(basename, tg, vowels)
            if line == '<exists>\n':
                totalDur = float(lastLine.strip())
            if startWord:
                if counter == 0:
                    startData.append(0)
                    counter += 1
                elif counter == 1:
                    startData.append(float(line.strip()))
                    counter += 1
                elif counter == 2:
                    startData.append(int(line.strip()))
                    counter = 10
                elif counter == 10:
                    a = float(line.strip())
                    counter += 1
                elif counter == 11:
                    b = float(line.strip())
                    counter += 1
                elif counter == 12:
                    if not isinstance(startPoint, type(None)) and a < (startPoint / 1000) < b:
                        isSegment = True
                    elif not isinstance(endPoint, type(None)) and a < (endPoint / 1000) < b:
                        isSegment = False
                    # It's a consonant we're interested in if it passes the .isdigit() check
                    # LG is laughter, NS is noise, CG is cough, LS is lip smack, and SL can be silence  (https://phon.wordpress.ncsu.edu/lab-manual/forced-alignment/)
                    if isSegment:
                        if line != '"sil"\n' and line != '"lg"\n' and line != '"ns"\n' and line != '"cg"\n' and line != '"ls"\n' and line != '"sl"\n' and any(map(str.isdigit, line.strip())):
                            vowel = line.strip().translate(str.maketrans('', '', string.punctuation))[:-1]
                            if vowel in particulars:
                                if vowel not in vowels:
                                    vowels[vowel] = list()
                                if isinstance(startPoint, type(None)):
                                    vowels[vowel].append((a,b))
                                else:
                                    vowels[vowel].append((a - (startPoint / 1000), b - (startPoint / 1000)))
                    counter = 10
            else:
                if line == '"phones"\n' and lastLine == '"IntervalTier"\n':
                    startWord = True
                lastLine = line


def lenisFortis(basename, sound):

    # arpabetVocalicList = ['AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AXR', 'AY',
    #                       'EH', 'ER', 'EY', 'IH', 'IX', 'IY', 'OW', 'OY', 
    #                       'UH', 'UW', 'UX']

    # For deciding tense-lax pairs: 
        # http://journals.linguisticsociety.org/proceedings/index.php/amphonology/article/viewFile/3653/3370
    vowelsOfInterest = ['IY', 'IH', 'UW', 'UH', 'AA', 'AE'] # Just the point vowels and their lax partners
    vowelsERBDict = {key: list() for key in vowelsOfInterest}  
    vowelsDURDict = {key: list() for key in vowelsOfInterest}  

    vowels = getParticularVowels(basename, vowelsOfInterest)

    for vowel in vowels:
        for start, end in vowels[vowel]:

            spaces = np.linspace(0, end - start, num = 7)
            part = sound.extract_part(from_time = start, to_time = end)

            burg = part.to_formant_burg()
            formants = np.array([getVocalicFormants(burg, i) for i in spaces[1:-1]]).flatten().tolist()

            F1s = formants[::3]
            F2s = formants[1::3]

            F1MidpointERB = herzToERB(F1s[3])
            F2MidpointERB = herzToERB(F2s[3])

            # Calculate the euclidean distance between the points
            dist = np.sqrt((F1MidpointERB - F2MidpointERB) ** 2)

            # Save out this spectral measure
            vowelsERBDict[vowel].append(dist)

            vowelsDURDict[vowel].append(end - start)

            # vowelsDict[vowel]['F1'].append(F1MidpointERB)
            # vowelsDict[vowel]['F2'].append(F2MidpointERB)

    return [np.nanmean(vowelsERBDict[v]) for v in vowelsOfInterest], vowelsERBDict, [np.nanmean(vowelsDURDict[v]) for v in vowelsOfInterest], vowelsDURDict


def getParticularConsonants(basename, particulars):

    tg = locateTextGrid(basename)

    startWord, isSegment = False, False
    startData = list()
    counter = 0
    totalDur = 0.0
    a, b = 0.0, 0.0
    consonants = dict()
    # We're dealing with a normalized file
    if "-" in basename:
        _, _, startSIL, _, _, offsetA, offsetB = basename.split('_')[0].split('-')
        startPoint = int(startSIL)
        endPoint = int(startSIL) + int(offsetB)
    # We're dealing with a full file
    else:
        startPoint = None
        endPoint = None
        isSegment = True

    with open(tg, 'r') as f:
        lastLine = ""
        for line in f:
            if line == '"IntervalTier"\n' and b == totalDur:
                if "-" in basename:
                    return consonants
                else:
                    return removePhonesFromInterviewer(basename, tg, consonants)
            if line == '<exists>\n':
                totalDur = float(lastLine.strip())
            if startWord:
                if counter == 0:
                    startData.append(0)
                    counter += 1
                elif counter == 1:
                    startData.append(float(line.strip()))
                    counter += 1
                elif counter == 2:
                    startData.append(int(line.strip()))
                    counter = 10
                elif counter == 10:
                    a = float(line.strip())
                    counter += 1
                elif counter == 11:
                    b = float(line.strip())
                    counter += 1
                elif counter == 12:
                    if not isinstance(startPoint, type(None)) and a < (startPoint / 1000) < b:
                        isSegment = True
                    elif not isinstance(endPoint, type(None)) and a < (endPoint / 1000) < b:
                        isSegment = False
                    # It's a consonant we're interested in if it passes the .isdigit() check
                    # LG is laughter, NS is noise, CG is cough, LS is lip smack, and SL can be silence  (https://phon.wordpress.ncsu.edu/lab-manual/forced-alignment/)
                    if isSegment:
                        if line != '"sil"\n' and line != '"lg"\n' and line != '"ns"\n' and line != '"cg"\n' and line != '"ls"\n' and line != '"sl"\n' and not any(map(str.isdigit, line.strip())):
                            consonant = line.strip().translate(str.maketrans('', '', string.punctuation))
                            if consonant in particulars:
                                if consonant not in consonants:
                                    consonants[consonant] = list()
                                if isinstance(startPoint, type(None)):
                                    consonants[consonant].append((a,b))
                                else:
                                    consonants[consonant].append((a - (startPoint / 1000), b - (startPoint / 1000)))
                    counter = 10
            else:
                if line == '"phones"\n' and lastLine == '"IntervalTier"\n':
                    startWord = True
                lastLine = line


def getSpectralContrast(basename, sound):

    # fricatives = ['CH', 'DH', 'F', 'H', 'HH', 'JH', 'S', 'SH', 'TH', 'V', 'WH', 'Z', 'ZH']
    fricativesOfInterest = ['F', 'V', 'S', 'Z', 'SH', 'ZH', 'TH', 'DH']
    fricativesDict = {key: list() for key in fricativesOfInterest}  
    
    fricatives = getParticularConsonants(basename, fricativesOfInterest)
    
    for fricative in fricatives: 

        for start, end in fricatives[fricative]:

            # Get the data slice
            s, e = int(np.floor(start * sound.sampling_frequency)), int(np.ceil(end * sound.sampling_frequency))
            data = sound.as_array()[0][s:e]

            # Bandpass filter data between 300 Hz and 20 kHz
            bs = filterbank(sound.sampling_frequency, low = 300, high = 20000)
            bpFilteredSig = lfilter(bs, 1, data, axis = 0)

            # Limit to just the middle 50% of the fricative (where we expect most of the frication)
            sig = bpFilteredSig[int(np.floor(data.shape[0] * .25)):int(np.ceil(data.shape[0] * .75))]

            # Cast to spectrum
            spectrum = parselmouth.Sound(sig).to_spectrum()

            # Calculate the Center of Gravity for the mid 50% of the fricative
            COG = spectrum.get_center_of_gravity()

            fricativesDict[fricative].append(COG)

    return [np.nanmean(fricativesDict[fricative]) for fricative in fricativesOfInterest], fricativesDict


def getPlosivesBeforeVowels(basename, plosives):

    tg = locateTextGrid(basename)

    startWord, isSegment = False, False
    startData = list()
    counter = 0
    totalDur = 0.0
    a, b, c, d = 0.0, 0.0, 0.0, 0.0
    consonants = dict()
    # We're dealing with a normalized file
    if "-" in basename:
        _, _, startSIL, _, _, offsetA, offsetB = basename.split('_')[0].split('-')
        startPoint = int(startSIL)
        endPoint = int(startSIL) + int(offsetB)
    # We're dealing with a full file
    else:
        startPoint = None
        endPoint = None
        isSegment = True

    with open(tg, 'r') as f:
        lastLine = ""
        for line in f:
            if line == '"IntervalTier"\n' and b == totalDur:
                if "-" in basename:
                    return consonants
                else:
                    return removePhonesFromInterviewer(basename, tg, consonants)
            if line == '<exists>\n':
                totalDur = float(lastLine.strip())
            if startWord:
                if counter == 0:
                    startData.append(0)
                    counter += 1
                elif counter == 1:
                    startData.append(float(line.strip()))
                    counter += 1
                elif counter == 2:
                    startData.append(int(line.strip()))
                    counter = 10
                elif counter == 10:
                    a = float(line.strip())
                    counter += 1
                elif counter == 11:
                    b = float(line.strip())
                    counter += 1
                elif counter == 12:
                    if not isinstance(startPoint, type(None)) and a < (startPoint / 1000) < b:
                        isSegment = True
                    elif not isinstance(endPoint, type(None)) and a < (endPoint / 1000) < b:
                        isSegment = False
                    # It's a consonant we're interested in if it passes the .isdigit() check
                    # LG is laughter, NS is noise, CG is cough, LS is lip smack, and SL can be silence  (https://phon.wordpress.ncsu.edu/lab-manual/forced-alignment/)
                    if isSegment:
                        if line != '"sil"\n' and line != '"lg"\n' and line != '"ns"\n' and line != '"cg"\n' and line != '"ls"\n' and line != '"sl"\n' and not any(map(str.isdigit, line.strip())):
                            if c != 0.0 and d != 0.0:
                                c, d = 0.0, 0.0
                            consonant = line.strip().translate(str.maketrans('', '', string.punctuation))
                            if consonant in plosives:
                                if consonant not in consonants:
                                    consonants[consonant] = list()
                                # Save out the current one to check if the next line is a vowel
                                # consonants[consonant].append((a,b))
                                c = a
                                d = b
                        elif line != '"sil"\n' and line != '"lg"\n' and line != '"ns"\n' and line != '"cg"\n' and line != '"ls"\n' and line != '"sl"\n' and any(map(str.isdigit, line.strip())):
                            # Check if the last line was a consonant
                            vowel = line.strip().translate(str.maketrans('', '', string.punctuation))[:-1]
                            if c != 0.0 and d != 0.0:
                                if isinstance(startPoint, type(None)):
                                    consonants[consonant].append([(consonant, c, d), (vowel, a, b)])
                                else:
                                    consonants[consonant].append([(consonant, c - (startPoint / 1000), d - (startPoint / 1000)), (vowel, a - (startPoint / 1000), b - (startPoint / 1000))])
                                c, d = 0.0, 0.0
                        else:
                            # It's silence or something other than a vowel or consonant
                            if c != 0.0 and d != 0.0:
                                c, d = 0.0, 0.0
                    counter = 10
            else:
                if line == '"phones"\n' and lastLine == '"IntervalTier"\n':
                    startWord = True
                lastLine = line


def getVoiceOnsetTime(basename, sound):

    # Get the stop voicing 
    # plosives = ['B', 'D', 'DX', 'G', 'K', 'P', 'Q', 'T']
    plosivesOfInterest = ['P', 'B', 'T', 'D', 'K', 'G']
    plosivesDict = {key: list() for key in plosivesOfInterest}

    plosivesBeforeVowels = getPlosivesBeforeVowels(basename, plosivesOfInterest)

    for plosive in plosivesBeforeVowels: # NOTE this may be empty for any given file; it's not clear that we have a plosive + vowel combo in each file
        for (orth1, pStart, pEnd), (orth2, vStart, vEnd) in plosivesBeforeVowels[plosive]:

            ################################
            # Get stop burst first peak
            ################################
            # We are trying here to automatically approximate finding the first peak of the stop burst
            spaces = np.linspace(pStart, pEnd, num = 100)
            energies = [sound.get_energy(from_time = s, to_time = e) for s, e in zip(spaces, spaces[1:])][10:90]

            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
            peaks, properties = find_peaks(energies, height = 0)
            bestCandidateBurstPeak = peaks[np.argmax(properties['peak_heights'])] + 10 # +10 because we're getting rid of the start and end of the plosive above with [10:90]

            stopBurstTime = spaces[bestCandidateBurstPeak]

            ################################
            # Get zero-crossing of the onset of the first glottal cycle for the vowel
            ################################

            s, e = int(np.floor(vStart * sound.sampling_frequency)), int(np.ceil(vEnd * sound.sampling_frequency))
            pressureWave = sound.as_array()[0][s:e]

            peaks, properties = find_peaks(pressureWave, distance = 100, height = 0)

            # Visualize this
            # from matplotlib import pyplot as plt
            # plt.plot(pressureWave)
            # plt.plot(peaks, pressureWave[peaks], "x")
            # plt.savefig('hi.png')

            betweenTwoPeaks = peaks[1] - peaks[0] # Trust the forced aligner; just grab between the first two, separated by at least 100 samples at 41 kHz

            zeroCrossing = sound.get_nearest_zero_crossing((betweenTwoPeaks + s) / sound.sampling_frequency)

            VOT = zeroCrossing - stopBurstTime

            plosivesDict[orth1].append(VOT)

    return [np.nanmean(plosivesDict[plosive]) for plosive in plosivesOfInterest], plosivesDict


def contrastWork(basename, sound):

    # Get the voice onset timings
    plosiveVOTMeans, plosiveVOTDict = getVoiceOnsetTime(basename, sound)

    # Get the spectral contrast of certain fricative pairs
    fricativeCOGMeans, fricativeSpectralContrastDict = getSpectralContrast(basename, sound)

    # Get the tense-lax distinction between certain vowel pairs
    vowelsERBMeans, vowelsERBDict, vowelsDURMeans, vowelsDURDict = lenisFortis(basename, sound)

    pD, fD, vERBD, vDurD = distinctions(plosiveVOTDict, fricativeSpectralContrastDict, vowelsERBDict, vowelsDURDict)

    return plosiveVOTMeans, fricativeCOGMeans, vowelsERBMeans, vowelsDURMeans, pD, fD, vERBD, vDurD


def main(file):

    # Get the file's basename
    basename = os.path.basename(file).split('.')[0]

    # Create a parselmouth sound object
    sound = parselmouth.Sound(file)

    # Phonemic Distinction Work
    plosiveVOTMeans, fricativeCOGMeans, vowelsERBMeans, vowelsDURMeans, pD, fD, vERBD, vDurD = contrastWork(basename, sound)

    return plosiveVOTMeans, fricativeCOGMeans, vowelsERBMeans, vowelsDURMeans, pD, fD, vERBD, vDurD


if __name__ == "__main__":

    main("/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cc/S001.wav")
    # main("/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSS-IS2020-data/train/Normalised_audio-chunks/cc/S077-240-9902-11373-1-0-1390.wav")
