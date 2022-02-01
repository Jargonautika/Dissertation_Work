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
        self.wav = WR(self.name)
        self.data = self.wav.getData()
        self.fs = self.wav.getSamplingRate()
        self.dur = self.wav.getDuration()
        self.NFFT = 512
   

    # Adapted from https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    def getMelFilterBank(self):

        # Pre-emphasize the signal
        preEmphasis = 0.97
        emphasizedSignal = np.append(self.data[0], self.data[1:] - preEmphasis * self.data[:-1])

        # Framing
        frames, frameLength = self.getFrames(emphasizedSignal)

        # Windowing
        frames *= np.hamming(frameLength)

        # Calculate power spectrum
        magFrames = np.absolute(np.fft.rfft(frames, self.NFFT))  # Magnitude of the FFT
        powFrames = ((1.0 / self.NFFT) * ((magFrames) ** 2))  # Power Spectrum

        # Filter banks
        filterBanks = self.getFilterBanks(powFrames)

        return filterBanks


    def getFrames(self, emphasizedSignal):

        frameSize, frameStride = 0.025, 0.01
        frameLength, frameStep = frameSize * self.fs, frameStride * self.fs  # Convert from seconds to samples
        signalLength = len(emphasizedSignal)
        frameLength = int(round(frameLength))
        frameStep = int(round(frameStep))
        numFrames = int(np.ceil(float(np.abs(signalLength - frameLength)) / frameStep))  # Make sure that we have at least 1 frame

        padSignalLength = numFrames * frameStep + frameLength
        z = np.zeros((padSignalLength - signalLength))
        padSignal = np.append(emphasizedSignal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

        indices = np.tile(np.arange(0, frameLength), (numFrames, 1)) + np.tile(np.arange(0, numFrames * frameStep, frameStep), (frameLength, 1)).T
        frames = padSignal[indices.astype(np.int32, copy=False)]

        return frames, frameLength


    def getFilterBanks(self, powFrames):

        lowFreqMel = 0
        nfilt = 40 # TODO check the center frequencies for the filters
        
        highFreqMel = (2595 * np.log10(1 + (self.fs / 2) / 700))  # Convert Hz to Mel
        melPoints = np.linspace(lowFreqMel, highFreqMel, nfilt + 2)  # Equally spaced in Mel scale
        hzPoints = (700 * (10**(melPoints / 2595) - 1))  # Convert Mel to Hz
        bin = np.floor((self.NFFT + 1) * hzPoints / self.fs)

        fbank = np.zeros((nfilt, int(np.floor(self.NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filterBanks = np.dot(powFrames, fbank.T)
        filterBanks = np.where(filterBanks == 0, np.finfo(float).eps, filterBanks)  # Numerical Stability
        filterBanks = 20 * np.log10(filterBanks)  # dB

        return filterBanks
