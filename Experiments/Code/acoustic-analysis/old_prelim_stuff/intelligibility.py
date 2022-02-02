#!/usr/bin/env python3

import os
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.stats import zscore


# For every little file that has been scored for DWGP (with both smn and ssn), get that speaker's id and AD condition and mmse score
def getData(exp_dir, OIM, level):

    A = pd.read_csv('/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSS-IS2020-data/train/cc_meta_data.txt', sep = ';').rename(columns=lambda x: x.strip())
    A['ID'] = A['ID'].str.strip()
    B = pd.read_csv('/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSS-IS2020-data/train/cd_meta_data.txt', sep = ';').rename(columns=lambda x: x.strip())
    B['ID'] = B['ID'].str.strip()
    C = pd.read_csv('/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSS-IS2020-data/meta_data_test.txt', sep = ';').rename(columns=lambda x: x.strip())
    C['ID'] = C['ID'].str.strip()

    dfList = list()
    for i in ['train', 'dev']:
        fileList = glob.glob(os.path.join(exp_dir, 'data', i, 'csv', str(level), '*'))[:]
        for csv in fileList:
            basename = os.path.basename(csv)
            condition = basename.split('.')[0].split('_')[-1]
            speaker = basename.split('-')[0]
            mmse = None
            for mmseDF in [A, B, C]:
                if speaker in mmseDF['ID'].tolist():
                    mmse = mmseDF[mmseDF['ID'] == speaker]['mmse'].tolist()[0]
                    break

            # There's just that one speaker who doesn't have a score, so we gotta skip him
            if not mmse:
                pass

            # Get intelligibility metrics
            df = pd.read_csv(csv, header = None)
            if OIM == 'DWGP':
                df = df.iloc[: , -6:-4]
                df.columns = ['smnDWGP', 'ssnDWGP']
            elif OIM == 'SII':
                df = df.iloc[: , -4:-2]
                df.columns = ['smnSII', 'ssnSII']
            elif OIM == 'STI':
                df = df.iloc[: , -2:]
                df.columns = ['smnSTI', 'ssnSTI']
            else:
                raise Exception

            # Get other identifiers
            df['speaker'] = speaker
            df['condition'] = condition
            df['mmse'] = mmse

            dfList.append(df)

    return pd.concat(dfList)


def main():

    # Run this with the STATS virtual environment
    parser = argparse.ArgumentParser(description='Run acoustic analyses on the consonants and vowels for all speakers by condition')
    parser.add_argument('exp_dir', nargs='?', type=str, help='Temporary experiment directory.', default = '/tmp/tmp.OgrVUR6iE4')
    args = parser.parse_args()
    zScoreList =  [1.04, 1.15, 1.28, 1.44, 1.645, 1.75, 1.96, 2.05, 2.33, 2.58, 'all']

    for OIM in ['DWGP', 'SII', 'STI']:
        for level in [70, 50, 30]:
            df = getData(args.exp_dir, OIM, level)

            for z in zScoreList:
                if z == 'all': # No zscore filtering; just gimme it all
                    df.to_csv(os.path.join(args.exp_dir, 'data', '{}_all_{}.csv'.format(str(level), OIM)), index = None)

                else: # incremental zscore filtering
                    miniDF = df.iloc[:,:2]
                    absZScores = np.abs(zscore(miniDF))
                    filteredEntries = (absZScores < z).all(axis=1)
                    df = df[filteredEntries]

                    df.to_csv(os.path.join(args.exp_dir, 'data', '{}_{}_{}.csv'.format(str(level), str(z), OIM)), index = None)


if __name__ == "__main__":

    main()
