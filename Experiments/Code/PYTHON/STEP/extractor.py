"""
lib_gt.py -- a set of functions used for gammatone filtering

Python (c) 2020 Yan Tang University of Illinois at Urbana-Champaign (UIUC)

Created: May 15, 2020
"""

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from lib.WAVReader import WAVReader as WR
from decimal import Decimal, ROUND_HALF_UP

class Extractor:

    def __init__(self, wav):
        
        self.name = wav
        self.wav = WR(self.name)
        self.data = self.wav.getData()
        self.fs = self.wav.getSamplingRate()
        self.dur = self.wav.getDuration()


    def mkRateMap(self, x, fs, cf_l=100, cf_u=7500, nb_ch=34, size_win=0.01, ti=0.008, compression=None, earmodel=None, lvl_ref=0):
        """
        mkRateMap.py -- generates the spectro-temporal excitation pattern (STEP) representation for given signal. 
        This is to model smoothed and compressed representation of the envelope of the basilar membrane response to sound.
        The wavform is was initially processed by gammatone filterbankds using a pole-mapping procedure described in (Cooke, 1993). 
        The Hilbert envelope in each channel of the filterbank was computed and smoothed with a leaky integrator with a ti ms (in default) time constant (Moore et al., 1988).
        The smoothed envelope was further downsampled.
        
        Inputs:
            x               input signal
            fs              sampling frequency in Hz
            cf_l            centre frequency of lowest filter in Hz (100 in default)
            cf_u            centre frequency of highest filter in Hz (1/2 sampling freqeuency in default)
            nb_ch           number of channels in filterbank (34 in default)
            size_win        interval between successive frames in second (0.01 in default)
            ti              temporal integration in second (0.008 in default)
            compression     type of compression ['cuberoot','log', None] (no compression in default)

        output:
            ratemap       STEP representation of the input signal x
            cf            central frequences for each band
        """

        sig = x.copy().flatten()

        nb_sample = len(sig)

        nb_spwin = int(Decimal(size_win * fs).quantize(1, ROUND_HALF_UP))

        nb_frame = int(np.ceil(nb_sample / nb_spwin))
        
        nb_sample2 = nb_frame * nb_spwin
        nb_diff = nb_sample2 - nb_sample
        if nb_diff > 0:
            sig = np.pad(sig, (0, nb_diff))

        ratemap = np.zeros((nb_ch, nb_frame))
        bm = np.zeros((nb_ch, nb_sample2))

        cfs = self.MakeERBCFs(cf_l, cf_u, n=nb_ch)

        wcf = 2 * np.pi * cfs
        tpt = 2 * np.pi / fs

        kT = np.asarray([i for i in range(nb_sample2)]) / fs
        bw = self.ERB(cfs) * self.bwcorrection()

        As = np.exp(-bw * tpt)
        gain = ((bw * tpt)**4 ) / 3

        if earmodel:
            gain = gain * self.db2amp(-self.earModel(cfs, earmodel), lvl_ref)

        intdecay = np.exp(-(1/(fs*ti)))
        intgain = 1 - intdecay

        for c in range(nb_ch): #nb_ch
            a = As[c]
            q = np.exp(-1j * wcf[c] * kT) * sig
            u = signal.lfilter([1, 4*a, a**2, 0], [1, -4*a, 6*a**2, -4*a**3, a**4], q)

            bm[c,:] = gain[c] * np.real(np.exp(1j * wcf[c] * kT) * u)
            env = gain[c] * np.abs(u)

            env_smoothed = signal.lfilter([1], [1, -intdecay], env)
            tmp = intgain * np.mean(np.reshape(env_smoothed, (nb_frame, nb_spwin)), axis=1)
            #avoid zero amplitude which causes -inf value when doing log compression
            tmp[tmp<1e-10] = 1e-10
            ratemap[c, :] = tmp

        if compression:
            if compression.upper() == "LOG":
                ratemap = 20 * np.log10(ratemap)
            elif compression.upper() == "CUBEROOT":
                ratemap = ratemap ** 0.3

        return ratemap, bm, cfs


    def earModel(self, cfs, model="ISO"):
        if model.upper() == "ISO":
            if np.min(cfs) < 20 or np.max(cfs) > 12500:
                raise ValueError("Central frequency out of range")

            f = (20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
            1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500)

            tf = (74.3, 65, 56.3, 48.4, 41.7, 35.5, 29.8, 25.1, 20.7, 16.8, 13.8, 11.2, 8.9, 7.2,
            6, 5, 4.4, 4.2, 3.7, 2.6, 1, -1.2, -3.6, -3.9, -1.1, 6.6, 15.3, 16.4, 11.6)

            plotter = interp1d(f, tf, kind="cubic")
            trshld = plotter(cfs)

        elif model.upper() == "TERHARDT":
            cfs = cfs / 1000
            trshld = 3.64 * (cfs**-0.8) - 6.5 * np.exp(-0.6 * (cfs-3.3)**2) + 10e-3 * (cfs ** 4)
        else:
            raise ValueError("Unsupported ear model!")

        return trshld


    def db2amp(self, level, ref=0):
        return np.power(10, (level-ref)/20)


    def ERB(self, x):
        return 24.7 * (4.37e-3 * x + 1)


    def HzToERBRate(self, x):
        return  21.4 * np.log10(4.37e-3 * x + 1)


    def ERBRateToHz(self, x):
        return  (10 ** (x/21.4)-1) / 4.37e-3


    def MakeERBCFs(self, fr_l, fr_h, n=34):
        return self.ERBRateToHz(np.linspace(self.HzToERBRate(fr_l), self.HzToERBRate(fr_h),  n)) 


    def bwcorrection(self):
        return 1.019


    def polemapping(self, x, wcf, kT, k, gain):
        q = np.exp(-1j * wcf * kT) * x
        u = signal.lfilter([1, k], [1, -2*k, k*k], q, axis=1)
        bm = np.sqrt(gain) * np.real(np.exp(1j * wcf * kT) * u)

        return bm


    def gammatonefilter(self, x, fs, cfr):
        """
            x: input signal
            fs: the sampling frequency of x
            cfr: the centre frequency in Hz of the band that is analysed

            return: the signal of the frequency band with a centre frequency of cfr, extacted from x
        """
        if x.shape[0] > x.shape[1]:
            x = x.T

        wcf = 2 * np.pi * cfr
        tpt = 2 * np.pi / fs
        bw = self.ERB(cfr) * self.bwcorrection()

        a = np.exp(-bw * tpt)
        kT = np.array([i for i in range(0, np.max(x.shape))])
        kT = kT / fs

        gain = ( (bw * tpt)**4 ) / 3  
        bm = self.polemapping(np.flip(self.polemapping(x, wcf, kT, a, gain)), wcf, kT, a, gain)

        return np.flip(bm)
        

if __name__ == "__main__":
    pass
    # wr = WAVReader("/Users/Kee/Desktop/sig.wav")
    # sig = wr.getData()

    # ratemap = mkRateMap(sig, wr.getSamplingRate(), compression=None, earmodel=None)[0]
    # plt.imshow(ratemap)
    # plt.show()

    # print(earModel(500))
