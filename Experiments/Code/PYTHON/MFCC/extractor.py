#!/usr/bin/env python3

import numpy as np
import parselmouth
from scipy.io import wavfile as wave
from silence import highPass, validateTs
from lib.WAVReader import WAVReader as WR
from python_speech_features import mfcc
from lib.DSP_Tools import findEndpoint


class Extractor:

    def __init__(self, wav):
        
        self.name = wav
        self.sound = parselmouth.Sound(wav)
        self.wav = WR(self.name)
        self.data = self.wav.getData()
        self.fs = self.wav.getSamplingRate()
        self.dur = self.wav.getDuration()
   
    #Timing features
    def findSilences(self):
        
        lowerCutoff = 40
        order = 1000
        w = 0.005
        
        bs = highPass(self.fs, lowerCutoff, order)
        ampData = np.array(self.ampData)
        ampData = np.squeeze(ampData)
        ampData = np.convolve(ampData[:], bs)
        ampData = np.reshape(ampData, (len(ampData), 1))
        silences = findEndpoint(ampData, self.fs, win_size=w)
        silences = validateTs(silences[0])
        
        return silences, w
    
    def getTimingStats(self):
        
        silence, win_size = self.findSilences()
        #average pause length (in ms)
        pauseLengths = list()
        currentCount = 0
        for i, item in enumerate(silence):
            if i+1 != len(silence):
                nextItem = silence[i+1]
                if item == True:
                    currentCount += 1
                    if item != nextItem:
                        pauseLengths.append(currentCount)
                        currentCount = 0
            else:
                if item == True:
                    currentCount += 1
                    pauseLengths.append(currentCount)

        avgPauseLength = (sum(pauseLengths)/len(pauseLengths))/win_size

        #sound-to-silence ratio
        sound = np.count_nonzero(silence==False)
        shh = np.count_nonzero(silence==True)
        s2sRatio = sound/shh
        
        #total number of pauses
        totalPauses = len(pauseLengths)

        return avgPauseLength, s2sRatio, totalPauses

    def getHNR(self):
        harmonicity = self.sound.to_harmonicity()
        hnr = harmonicity.values.mean()
        return hnr

    def getJitter(self):
        jitter = 0
        return jitter

    def getShimmer(self):
        shimmer = 0
        return shimmer

    def getMFCCs(self, ms=0.01):
        (rate, sig) = wave.read(self.name)
        mfccs = mfcc(sig, samplerate=rate, winlen=ms, winstep=ms)
        #mfccs is an array of size (YO-IDK [Helen says it's windows], 13) aka: first 13 mfccs for each...
        return mfccs

