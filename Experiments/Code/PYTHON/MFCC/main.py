#!/usr/bin/env python3

from extractor import Extractor
import pandas as pd
import argparse
import glob
import os


def pythonCall(wavFolder, destFolder):

    for wav in glob.glob(os.path.join(wavFolder, '*.wav')):
        base = os.path.basename(os.path.splitext(wav)[0].strip())
        mfcc = os.path.join(destFolder, base + '.csv')
        ext = Extractor(wav)
        mfccs = ext.getMFCCs()

        df = pd.DataFrame(mfccs)
        dfT = df.T # We have to transpose this because currently it gives us (# windows, 13) instead of the other way
        dfT.to_csv(mfcc, index_label = False, header = False)


def extract(exp_dir, which):

    wavFolder = os.path.join(exp_dir, 'data', which, 'wav')
    destFolder = os.path.join(exp_dir, 'data', which, 'csv')

    pythonCall(wavFolder, destFolder)


def main():

    parser = argparse.ArgumentParser(description='Description of part of pipeline.')
    parser.add_argument('exp_dir', nargs='?', type=str, help='Temporary experiment directory.', default='/tmp/tmp.withheldMFCC')
    parser.add_argument('train_dev', nargs='?', type=str, help='Whether we are doing training or development extraction at the moment.', default='/tmp/tmp.withheldMFCC/data/train')

    args = parser.parse_args()
    extract(args.exp_dir, args.train_dev)


if __name__ == "__main__":

    main()

