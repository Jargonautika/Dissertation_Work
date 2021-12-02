#!/usr/bin/env python3

import pandas as pd
import opensmile
import argparse
import glob
import os


def egemapsCall(wavFolder, destFolder):

    for waveform in glob.glob("{}/*".format(wavFolder)):
        name = os.path.basename(waveform).split('.')[0]
        smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv01b,
                                feature_level=opensmile.FeatureLevel.Functionals
                                )
        y = smile.process_file(waveform)
        y.to_csv(os.path.join(destFolder, name + '.csv'), header = None, index = False)


def compareCall(wavFolder, destFolder):

    for waveform in glob.glob("{}/*".format(wavFolder)):
        name = os.path.basename(waveform).split('.')[0]
        smile = opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016,
                                feature_level=opensmile.FeatureLevel.Functionals
                                )
        y = smile.process_file(waveform)
        y.to_csv(os.path.join(destFolder, name + '.csv'), header = None, index = False)


def extract(exp_dir, feature_set, which):

    wavFolder = os.path.join(exp_dir, 'data', which, 'wav')
    destFolder = os.path.join(exp_dir, 'data', which, 'csv')

    if feature_set == "compare":
        compareCall(wavFolder, destFolder)
    elif feature_set == "gemaps":
        egemapsCall(wavFolder, destFolder)
    else:
        # You shouldn't be here; soemthing is wrong in extract-compare and/or extract-gemaps
        raise Exception


def main():

    parser = argparse.ArgumentParser(description='Description of part of pipeline.')
    parser.add_argument('exp_dir', type=str, help='Temporary experiment directory.')
    parser.add_argument('feature_set', type=str, help='Whether we are using COMPARE or eGeMAPS')
    parser.add_argument('train_dev', type=str, help='Whether we are doing training or development extraction at the moment.')

    args = parser.parse_args()
    extract(args.exp_dir, args.feature_set, args.train_dev)


if __name__ == "__main__":

    main()

