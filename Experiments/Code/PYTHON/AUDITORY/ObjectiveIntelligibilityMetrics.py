#!/usr/bin/env python3

import numpy as np
np.random.seed(42)

from lib.WAVReader import WAVReader as WR
from lib.WAVWriter import WAVWriter as WW
import soundfile as sf
import pandas as pd
import librosa
import shutil
import glob
import os

class Intelligibility:

    def __init__(self, wavFolder, dBSNR_smn, dBSNR_ssn):

        self.folder         = wavFolder.rsplit('/', 1)[0]
        self.dBSNR_smn      = dBSNR_smn
        self.dBSNR_ssn      = dBSNR_ssn

        # Make a place to put the noise signals (isolated only; speech+noise mixture done downstream in MATLAB) 
        self.SMNFolder      = os.path.join(self.folder, 'smn')
        self.SSNFolder      = os.path.join(self.folder, 'ssn')

        # An output place for the actual intelligibility metrics (DWGP, SII, STI)
        self.SMNDWGP        = os.path.join(self.folder, 'smn_dwgp')
        self.SSNDWGP        = os.path.join(self.folder, 'ssn_dwgp')
        self.SMNSII         = os.path.join(self.folder, 'smn_sii')
        self.SSNSII         = os.path.join(self.folder, 'ssn_sii')
        self.SMNSTI         = os.path.join(self.folder, 'smn_sti')
        self.SSNSTI         = os.path.join(self.folder, 'ssn_sti')

        # All waveforms need to be downsampled to 16kHz from 44.1kHz
        self.DSFolder       = os.path.join(self.folder, 'downsample')
        self.wavs           = glob.glob(os.path.join(wavFolder, '*.wav'))
        self.SMNDWGPValues  = dict()
        self.SSNDWGPValues  = dict()
        self.SMNSIIValues   = dict()
        self.SSNSIIValues   = dict()
        self.SMNSTIValues   = dict()
        self.SSNSTIValues   = dict()

        # Make sure every run is clean
        for folder in [self.SMNFolder, self.SSNFolder, self.SSNSII, 
                       self.SMNDWGP,   self.SSNDWGP,   self.SMNSII,       
                       self.SMNSTI,    self.SSNSTI,    self.DSFolder]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.mkdir(folder)

        # Set up to create the individual noise for each waveform
        smnNoise = "/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/Noise/smn_16k.wav"
        ssnNoise = "/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/Noise/ssn_16k.wav"
        self.smnNoiseData = WR(smnNoise).getData()
        self.ssnNoiseData = WR(ssnNoise).getData()

        for name in self.wavs[:]:
            self.extractNoise(name)

        # Calculate all three Objective Intelligibility Metrics
        self.getDWGP()
        self.getSII()
        self.getSTI()

        for name in self.wavs[:]:

            smnDWGP, ssnDWGP, smnSII, ssnSII, smnSTI, ssnSTI = self.extractFromCSVs(name)
            self.SMNDWGPValues[name] = smnDWGP
            self.SSNDWGPValues[name] = ssnDWGP
            self.SMNSIIValues[name]  = smnSII
            self.SSNSIIValues[name]  = ssnSII
            self.SMNSTIValues[name]  = smnSTI
            self.SSNSTIValues[name]  = ssnSTI

        self.cleanUp()


    def extractNoise(self, name):

        basename = os.path.basename(name)

        # Downsample to 16kHz
        # It's ok to use librosa and not Yan's WR/WW classes: 
            # https://stackoverflow.com/questions/30619740/downsampling-wav-audio-file
        y, sr = librosa.load(name, sr = 16000)
        tempWAV = os.path.join(self.DSFolder, basename)
        sf.write(tempWAV, y, sr)
        
        # Extract solitary noise signals for both SMN and SSN
        # NOTE: OIMs require the noise files to be the same length as the speech file
        fs = 16000
        bits = 16
        for data, folder in [(self.smnNoiseData, self.SMNFolder), 
                             (self.ssnNoiseData, self.SSNFolder)]:

            # Get a trimmed bit of noise paired to the length of the speech signal
            maxStart = len(data) - len(y)
            startPoint = np.random.choice(maxStart, 1)[0]
            q = data[startPoint:startPoint + len(y)]

            # Leveling of the SNR also needs to happen in MATLAB so that we don't have to scale while writing
            # We'll create the speech+noise mixture in MATLAB so we don't run into clipping problems
            
            # Write out the noise files
            WW(os.path.join(folder, basename), q, fs, bits).write()


    def getDWGP(self):

        # Go to the MATLAB code directory
        current_dir = os.getcwd()
        os.chdir('/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Code/MATLAB/OIMs/DWGP')

        # Get DWGP for Speech Modulated Noise
        command = "callDWGP('$inputFolder', '$destFolder', '$noiseFolder', $SNR)"
        os.system("inputFolder={}; destFolder={}; noiseFolder={}; SNR={}; matlab -batch \"{}\";".format(self.DSFolder, self.SMNDWGP, self.SMNFolder, self.dBSNR_smn, command))

        # Get DWGP for Speech Shaped Noise
        command = "callDWGP('$inputFolder', '$destFolder', '$noiseFolder', $SNR)"
        os.system("inputFolder={}; destFolder={}; noiseFolder={}; SNR={}; matlab -batch \"{}\";".format(self.DSFolder, self.SSNDWGP, self.SSNFolder, self.dBSNR_ssn, command))

        # Come back home
        os.chdir(current_dir)


    def getSII(self):

        # Go to the MATLAB code directory
        current_dir = os.getcwd()
        os.chdir('/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Code/MATLAB/OIMs/SII')

        # Get SII for Speech Modulated Noise
        command = "callSII('$inputFolder', '$destFolder', '$noiseFolder', $SNR)"
        os.system("inputFolder={}; destFolder={}; noiseFolder={}; SNR={}; matlab -batch \"{}\";".format(self.DSFolder, self.SMNSII, self.SMNFolder, self.dBSNR_smn, command))

        # Get SII for Speech Shaped Noise
        command = "callSII('$inputFolder', '$destFolder', '$noiseFolder', $SNR)"
        os.system("inputFolder={}; destFolder={}; noiseFolder={}; SNR={}; matlab -batch \"{}\";".format(self.DSFolder, self.SSNSII, self.SSNFolder, self.dBSNR_ssn, command))

        # Come back home
        os.chdir(current_dir)


    def getSTI(self):

        # Go to the MATLAB code directory
        current_dir = os.getcwd()
        os.chdir('/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Code/MATLAB/OIMs/STI')

        # Get STI for Speech Modulated Noise
        command = "callSTI('$inputFolder', '$destFolder', '$noiseFolder', $SNR)"
        os.system("inputFolder={}; destFolder={}; noiseFolder={}; SNR={}; matlab -batch \"{}\";".format(self.DSFolder, self.SMNSTI, self.SMNFolder, self.dBSNR_smn, command))

        # Get STI for Speech Shaped Noise
        command = "callSTI('$inputFolder', '$destFolder', '$noiseFolder', $SNR)"
        os.system("inputFolder={}; destFolder={}; noiseFolder={}; SNR={}; matlab -batch \"{}\";".format(self.DSFolder, self.SSNSTI, self.SSNFolder, self.dBSNR_ssn, command))

        # Come back home
        os.chdir(current_dir)


    def extractFromCSVs(self, name):

        # Get the OIMs for each
        csv = os.path.basename(name).split('.')[0] + '.csv'

        #####################
        # DWGP
        #####################
        smnDWGP = os.path.join(self.SMNDWGP, csv)
        df = pd.read_csv(smnDWGP)
        smnDWGP = df['DWGP'][0]
        # smnGP = df['GP'][0]

        ssnDWGP = os.path.join(self.SSNDWGP, csv)
        df = pd.read_csv(ssnDWGP)
        ssnDWGP = df['DWGP'][0]
        #ssnGP = df['GP'][0]

        #####################
        # SII
        #####################
        smnSII = os.path.join(self.SMNSII, csv)
        df = pd.read_csv(smnSII)
        smnSII = float(df.columns.tolist()[0])

        ssnSII = os.path.join(self.SSNSII, csv)
        df = pd.read_csv(ssnSII)
        ssnSII = float(df.columns.tolist()[0])

        #####################
        # STI
        #####################
        smnSTI = os.path.join(self.SMNSTI, csv)
        df = pd.read_csv(smnSTI)
        smnSTI = float(df.columns.tolist()[0])

        ssnSTI = os.path.join(self.SSNSTI, csv)
        df = pd.read_csv(ssnSTI)
        ssnSTI = float(df.columns.tolist()[0])

        # Return all objective metrics
        return smnDWGP, ssnDWGP, smnSII, ssnSII, smnSTI, ssnSTI


    def cleanUp(self):

        # Delete the temporary files
        for folder in [self.SMNFolder, self.SSNFolder, self.DSFolder, self.SMNDWGP,
                       self.SSNDWGP, self.SMNSII, self.SSNSII, self.SMNSTI, self.SSNSTI]:
            shutil.rmtree(folder)

    
    def returnValues(self):

        return self.SMNDWGPValues, self.SSNDWGPValues, self.SMNSIIValues, self.SSNSIIValues, self.SMNSTIValues, self.SSNSTIValues
