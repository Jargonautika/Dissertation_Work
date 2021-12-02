#!/usr/bin/env python3

import os
import math
import string
import librosa
import parselmouth
import numpy as np
import pandas as pd

from scipy.io import wavfile as wave
from DynamicRange import calDynamicRange
from silence import highPass, validateTs
from lib.WAVReader import WAVReader as WR
from lib.DSP_Tools import findEndpoint


class Extractor:

    def __init__(self, wav, tg, bw):
        
        self.name = wav # Short, "normalised" waveforms (3-10 seconds)
        self.id = self.name.split('-')[0].split('/')[-1]
        self.tg = tg # The TextGrid - from the "full-wave enhanced"
        self.bw = bw # The Big Waveform - "full wave enhanced"
        self.sound = parselmouth.Sound(self.bw)
        self.bw_sound = parselmouth.Sound(self.bw)
        self.wav = WR(self.bw)
        self.data = self.wav.getData()
        self.fs = self.wav.getSamplingRate()
        self.dur = self.wav.getDuration()
        self.arpabetConsonantalList = ['B', 'CH', 'D', 'DH', 'CX', 'EL', 'EM', 'EN', 'F', 'G',
                                       'H', 'JH', 'K', 'L', 'M', 'N', 'NX', 'NG', 'P', 'Q', 'R',
                                       'S', 'SH', 'T', 'TH', 'V', 'W', 'WH', 'Y', 'Z', 'ZH']
        self.arpabetVocalicList     = ['AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AXR', 'AY',
                                       'EH', 'ER', 'EY', 'IH', 'IX', 'IY', 'OW', 'OY', 
                                       'UH', 'UW', 'UX']
        self.matrix, self.matrix_labels = self.getMatrix()


    # Landing space for extracting all values and placing them into an NxM matrix
    def getMatrix(self):

        dynamicRange = self.getDynamicRange()
        energy = self.getEnergy()
        intensity = self.getIntensity()
        zcr = self.getZeroCrossingRate()
        rms, spl = self.getRootMeanSquare()

        # The following metrics come from the big waveform, not the individual, normalized ones
        averageWordDuration, averageSilenceDuration = self.getAverageWordDuration() # This will be a global score using the non-normalized audio recordings.
        consonantalArray, consonantCount = self.getConsonantalInformation()
        vocalicArray, vowelCount = self.getVocalicInformation()
        try: # This shouldn't come up for the global metrics, but at the local level some utterances just don't have consonants or vowels. 
            consonantVowelRatio = consonantCount / vowelCount
        except:
            consonantVowelRatio = 0.0

        matrixList = [averageWordDuration, averageSilenceDuration, dynamicRange, energy, intensity, zcr, rms, spl, consonantVowelRatio]
        for c in consonantalArray:
            matrixList.append(c)
        for v in vocalicArray:
            matrixList.append(v)

        matrixLabelsList = ['Avg. Word Dur.', 'Avg. Sil. Dur.', 'Dynamic Range', 'Energy',
                            'Intensity', 'ZCR', 'Root Mean Square', 'Sound Pressure Level', 'Consonant Vowel Ratio']

        # Consonants
        for arpaC in self.arpabetConsonantalList:
            for moment in ['CoG', 'Kur', 'Ske', 'Std']:
                matrixLabelsList.append('{}_{}'.format(arpaC, moment))
        matrixLabelsList.append('Avg. Cons. Dur.')

        # Vowels
        for arpaV in self.arpabetVocalicList:
            for spacing in ['F0_1', 'F0_2', 'F0_3', 'F0_4', 'F0_5']:
                matrixLabelsList.append('{}-{}'.format(arpaV, spacing))
            for spacing in ['1', '2', '3', '4', '5']:
                for formant in ['F1', 'F2', 'F3']:
                    matrixLabelsList.append('{}-{}_{}'.format(arpaV, formant, spacing))
        matrixLabelsList.append('Avg. Voca. Dur.')

        assert len(matrixList) == len(matrixLabelsList), "You are missing labels or data."

        return matrixList, matrixLabelsList


    def returnMatrix(self):

        return self.matrix


    # Calculate the mean word duration for a particular speaker
    # We do this by looking at the TextGrid for the speaker and subsetting 
        # down to the 'word' tier. All durations are calculated and then averaged.
    def getAverageWordDuration(self):

        startWord = False
        startData, wordList, wordDuration, silenceDuration = list(), list(), list(), list()
        counter = 0
        a, b = 0.0, 0.0
        with open(self.tg, 'r') as f:
            lastLine = "" # Keep an eye on what we saw previous to this line
            for line in f:
                if startWord: # startWord is flipped to True after we have gotten past the 'phone' IntervalTier
                    if counter == 0: # At the beginning of a Tier there is a duration start (usually 0 here)
                        startData.append(0)
                        counter += 1
                    elif counter == 1: # Tier-beginning duration maximum
                        startData.append(float(line.strip()))
                        counter += 1
                    elif counter == 2: # Tier-beginning number of items in the tier (here words)
                        startData.append(int(line.strip()))
                        counter = 10
                    elif counter == 10: # We have finally hit the triplets; this is the start of each word duration
                        a = float(line.strip())
                        counter += 1
                    elif counter == 11: # End of each word duration
                        b = float(line.strip())
                        counter += 1
                    elif counter == 12: # Actual word segment, could be silence
                        if line != '"SIL"\n':
                            wordList.append(line.strip())
                            wordDuration.append(b - a)
                        else:
                            silenceDuration.append(b - a)
                        counter = 10 # reset to the beginning of a new triplet
                    
                else:
                    if line == '"word"\n' and lastLine == '"IntervalTier"\n': # Make sure we skip over the 'phone' IntervalTier
                        startWord = True
                    lastLine = line

        self.wordList = wordList # We're not currently using this, but it might be nice to do an analysis of the types of words used, if we want to get into lexical stuff
        return np.mean(wordDuration), np.mean(silenceDuration) # Return a global average of both silence and word duration for a given speaker


    # Compute Dynamic Range in decibels
    def getDynamicRange(self):

        return calDynamicRange(self.data, self.fs)


    def getEnergy(self):

        energy = np.sum(self.data ** 2)
        return energy


    def getIntensity(self):

        intensity = self.sound.to_intensity().get_average()
        return intensity


    def getZeroCrossingRate(self):

        Nz = np.diff((self.data >= 0))

        return np.sum(Nz) / len(self.data)


    # If we're looking at the normalized files, this *should* all be the same, right?
    def getRootMeanSquare(self): 

        rms = np.sqrt(np.mean(self.data**2))
        spl = self.getSoundPressureLevel(rms)
        return rms, spl


    def getSoundPressureLevel(self, rms, p_ref = 2e-5):

        spl = 20 * np.log10(rms/p_ref)
        return spl


    # Landing space for extracting consonant-by-consonant metrics
    def getConsonantalInformation(self):

        consonants = self.getConsonants()
        
        # Check to make sure we don't have anything non-ARPABET
        for c in consonants:
            if c not in self.arpabetConsonantalList:
                print("There are consonants not known to the ARPABET list: {}".format(c))
                raise AssertionError

        averageConsonantalDuration = self.getConsonantalDuration(consonants)
        spectralArray = list()
        consonantCount = 0
        for arpaC in self.arpabetConsonantalList:
            if arpaC in consonants: # If the speaker actually produced the consonant at least one time
                cog, kur, ske, std, count = self.getSpectralMoments(consonants[arpaC])
                spectralArray.append(cog)
                spectralArray.append(kur)
                spectralArray.append(ske)
                spectralArray.append(std)
                consonantCount += count # Append the count per consonant type in the ARPABET list to create one global count of all consonants uttered by a speaker
            else: # This speaker may or may not have actually produced a given consonant, but the feature vector needs to be the same shape
                for i in range(4):
                    spectralArray.append(np.nan)
        
        spectralArray.append(averageConsonantalDuration)
        return spectralArray, consonantCount


    # Return a dictionary with consonants as keys and a list of tuples filled with start and end times as the values
    def getConsonants(self):

        startWord = False
        startData, wordList, wordDuration, silenceDuration = list(), list(), list(), list()
        counter = 0
        totalDur = 0.0
        a, b = 0.0, 0.0
        consonants = dict()
        with open(self.tg, 'r') as f:
            lastLine = "" # Keep an eye on what we saw previous to this line
            for line in f:
                if line == '"IntervalTier"\n' and b == totalDur: # Make sure we're not going on to the 'word' IntervalTier
                    return consonants
                if line == '<exists>\n': # End of TextGrid header metadata
                    totalDur = float(lastLine.strip())
                if startWord: # Makes sure we are past the header metadata and into the 'phone' IntervalTier
                    if counter == 0: # Start duration
                        startData.append(0)
                        counter += 1
                    elif counter == 1: # End duration
                        startData.append(float(line.strip()))
                        counter += 1
                    elif counter == 2: # Number of items (here phones)
                        startData.append(int(line.strip()))
                        counter = 10
                    elif counter == 10: # Beginning of triple; float of start duration for triple
                        a = float(line.strip())
                        counter += 1
                    elif counter == 11: # Float of end duration for triple
                        b = float(line.strip())
                        counter += 1
                    elif counter == 12: # Actual segment
                        # It's a consonant if it passes the lack of .isdigit() check below
                        # LG is laughter, NS is noise, CG is cough, LS is lip smack, and SL can be silence  (https://phon.wordpress.ncsu.edu/lab-manual/forced-alignment/)
                        if line != '"sil"\n' and line != '"lg"\n' and line != '"ns"\n' and line != '"cg"\n' and line != '"ls"\n' and line != '"sl"\n' and not any(map(str.isdigit, line.strip())):
                            consonant = line.strip().translate(str.maketrans('', '', string.punctuation))
                            if consonant == "HH":
                                consonant = "H" # We'll just collapse this category and leave the NX/NG distinction to allow for the voiced nasal alveolar flap in /winner/
                            if consonant not in consonants:
                                consonants[consonant] = list()
                            consonants[consonant].append((a,b)) # Append the start and end times for that consonant
                        counter = 10 # Reset to the next triple
                else:
                    if line == '"phones"\n' and lastLine == '"IntervalTier"\n': # Make sure we are only looking at the first IntervalTier
                        startWord = True
                    lastLine = line


    # Return four spectral moments and the raw count of consonants produced
    def getSpectralMoments(self, c):

        cog, kur, ske, std, consonantCount = list(), list(), list(), list(), 0
        # Iterate over the number of times a given consonant was produced and measured
        for start, end in c:
            consonantCount += 1
            # TODO this 'part' variable is a subset of the Big Waveform (not normalised chunk stuff)
                # Those files need to be rms-normalized before these values mean anything
                # This here is what marks 'global' most from 'local' measurements: we are looking at all consonants throughout
                # the speaker's TextGrid and then collapsing them down into a single floating point value
                # for each speaker. This means that in the input feature vectors to the ML and NN models, each
                # speaker's consonantal spectral moments will be constant, and we should probably reduce the space
                # by having one speaker have one input feature vector (to make it truly 'global')
            part = self.bw_sound.extract_part(from_time = start, to_time = end)
            spectrum = part.to_spectrum()
            cog.append(self.getConsonantalCenterOfGravity(spectrum))
            kur.append(self.getConsonantalKurtosis(spectrum))
            ske.append(self.getConsonantalSkewness(spectrum))
            std.append(self.getConsonantalStandardDeviation(spectrum))

        return np.mean(cog), np.mean(kur), np.mean(ske), np.mean(std), consonantCount


    def getConsonantalCenterOfGravity(self, spectrum):

        return spectrum.get_center_of_gravity()


    def getConsonantalKurtosis(self, spectrum):

        return spectrum.get_kurtosis()


    def getConsonantalSkewness(self, spectrum):

        return spectrum.get_skewness()


    def getConsonantalStandardDeviation(self, spectrum):

        return spectrum.get_standard_deviation()

    
    # Return a single floating point mean for all consonants produced by the individual throughout the entire TextGrid
    def getConsonantalDuration(self, consonants):

        durList = list()
        # Iterate over the set of consonants produced by the individual 
        for c in consonants:
            # Iterate over all of the phonetic realizations of that consonant in the set of all consonants produced
            for start, end in consonants[c]:
                durList.append(end - start)

        return np.mean(durList)


    # Landing space for extracting vowel-by-vowel metrics
    def getVocalicInformation(self):

        vowels = self.getVowels()
        
        # Check to make sure we don't have anything non-ARPABET
        for v in vowels:
            if v not in self.arpabetVocalicList: # The -1 indexes for the stress, which we may want to look into later
                print("There are consonants not known to the ARPABET list: {}".format(v))
                raise AssertionError

        averageVocalicDuration = self.getVocalicDuration(vowels)
        vocalicArray = list()
        vowelCount = 0
        for arpaV in self.arpabetVocalicList:
            if arpaV in vowels: # Sometimes an individual won't have produced a vowel in our ARPABET set
                information, count = self.getVocalicPitch(vowels[arpaV])
                vowelCount += count
                for i in information:
                    vocalicArray.append(i)
            else:
                for _ in range(20): # If the vowel hasn't been produced at all, fill the matrix space
                    vocalicArray.append(np.nan)

        vocalicArray.append(averageVocalicDuration)
        return vocalicArray, vowelCount


    # Return a dictionary with vowels as keys and a list of tuples filled with start and end times as the values
    def getVowels(self):

        startWord = False
        startData, wordList, wordDuration, silenceDuration = list(), list(), list(), list()
        counter = 0
        totalDur = 0.0
        a, b = 0.0, 0.0
        vowels = dict()
        with open(self.tg, 'r') as f:
            lastLine = "" # Keep an eye on what we saw previous to this line
            for line in f:
                if line == '"IntervalTier"\n' and b == totalDur: # Make sure we're not going on to the 'word' IntervalTier
                    return vowels
                if line == '<exists>\n': # End of TextGrid header metadata
                    totalDur = float(lastLine.strip())
                if startWord: # Makes sure we are past the header metadata and into the 'phone' IntervalTier
                    if counter == 0: # Start duration
                        startData.append(0)
                        counter += 1
                    elif counter == 1: # End duration
                        startData.append(float(line.strip()))
                        counter += 1
                    elif counter == 2: # Number of items (here phones)
                        startData.append(int(line.strip()))
                        counter = 10
                    elif counter == 10: # Beginning of triple; float of start duration for triple
                        a = float(line.strip())
                        counter += 1
                    elif counter == 11: # Float of end duration for triple
                        b = float(line.strip())
                        counter += 1
                    elif counter == 12:
                        # It's a vowel if it passes the .isdigit() check below
                        # LG is laughter, NS is noise, CG is cough, LS is lip smack, and SL can be silence  (https://phon.wordpress.ncsu.edu/lab-manual/forced-alignment/)
                        if line != '"sil"\n' and line != '"lg"\n' and line != '"ns"\n' and line != '"cg"\n' and line != '"ls"\n' and line != '"sl"\n' and any(map(str.isdigit, line.strip())):
                            vowel = line.strip().translate(str.maketrans('', '', string.punctuation))[:-1] # Get rid of the stress counter (maybe useful later, but not used yet)
                            if vowel not in vowels: # Make a set of all the speaker's vowels
                                vowels[vowel] = list()
                            vowels[vowel].append((a,b))
                        counter = 10 # Reset to the beginning of a triple
                else:
                    if line == '"phones"\n' and lastLine == '"IntervalTier"\n': # Make sure we are only looking at the first IntervalTier
                        startWord = True
                    lastLine = line


    # Iterate over all of the instances of the vowels a given speaker has produced; extract measures
    def getVocalicPitch(self, v):

        tempArray, vowelCount = list(), 0
        for start, end in v:
            vowelCount += 1
            # We are extracting the sound from the big waveform here
            part = self.bw_sound.extract_part(from_time = start, to_time = end, preserve_times = True)
            # Cast to a pitch object
            pitch = part.to_pitch_cc()
            # Calculate seven individual time points (the first is at time 0.0 and the last is at the very end, so discard those)
            spaces = np.linspace(start, end, num = 7)
            # Calculate F0 for every one of those spaces in the time series
            pitches = [pitch.get_value_at_time(i) for i in spaces[1:-1]]

            # Prep for Formant extraction
            burg = part.to_formant_burg()
            # Calculate Formants 1, 2, and 3 at those same time stamps as above with the pitches
            formants = np.array([self.getVocalicFormants(burg, i) for i in spaces[1:-1]]).flatten().tolist()
            
            # Save out this observation
            tempArray.append(pitches + formants)

        return np.nanmean(tempArray, axis = 0), vowelCount #TODO We keep getting Mean of Empty Slice here...


    # Return a three-part list for Formants 1, 2, and 3 at a given time point in the vowel
    def getVocalicFormants(self, burg, i):

        outList = list()
        # Get Formants 1, 2, and 3
        for j in range(1, 4):
            bark = burg.get_value_at_time(formant_number = j, time = i, unit = parselmouth.FormantUnit.HERTZ)
            outList.append(bark)

        return outList


    # Return a single floating point mean for all consonants produced by the individual throughout the entire TextGrid
    def getVocalicDuration(self, vowels):

        durList = list()
        for v in vowels: # Iterate over all of the vowels in the set of vowels produced by the speaker
            for start, end in vowels[v]: # Iterate over all of the instances of the vowel; get their start and end times
                durList.append(end - start)

        return np.mean(durList)

    # This is the old version that we got the 87.15% accuracy on with LOO-cross validation with baseline neural model and 230 kbest feature space reduction
    # Hopefully Yan's auditory Dynamic Range calculation works better since it's in dB
    # def getDynamicRange(self):

    #     x_min = np.min(self.data)
    #     x_max = np.max(self.data)
    #     R = np.abs(x_max - x_min)
    #     return R

    # # https://www.programmersought.com/article/6348312465/
    # def getZeroCrossingRate(self):

    #     frameSize = 512
    #     overLap = 0
    #     wavLen = len(self.data)
    #     step = frameSize - overLap
    #     frameNum = math.ceil(wavLen/step)
    #     zcr = np.zeros((frameNum, 1))
    #     for i in range(frameNum):
    #         currentFrame = self.data[np.arange(i*step, min(i * step + frameSize, wavLen))]
    #         currentFrame = currentFrame - np.mean(currentFrame) # Zero-justified
    #         zcr[i] = sum(currentFrame[0:-1] * currentFrame[1::]<=0)

    #     return np.mean(zcr)
