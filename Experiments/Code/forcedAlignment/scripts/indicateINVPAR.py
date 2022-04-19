#!/usr/bin/env python3

import pandas as pd
import numpy as np
import glob
import os


def figureINVPAR(line, a, b, df):

    if isinstance(df, type(None)):
        return '"{}___PAR"\n'.format(line.lower().strip().replace('"', ''))

    firstSpkr, firstBegin, firstEnd = df.iloc[0]['speaker'], np.floor(df.iloc[0]['begin'] * 0.001), np.ceil(df.iloc[0]['end'] * 0.001)
    if a <= firstBegin and b <= firstEnd:
        return '"{}___{}"\n'.format(line.lower().strip().replace('"', ''), firstSpkr)

    recentSpkr, recentBegin = None, None
    for i, row in df.iterrows():
        if i > 0:
            recentSpkr, recentBegin = spkr, begin
        spkr, begin, end = row['speaker'], np.floor(row['begin'] * 0.001), np.ceil(row['end'] * 0.001)
        if a >= begin and b <= end:
            return '"{}___{}"\n'.format(line.lower().strip().replace('"', ''), spkr)

        elif i > 0 and a >= recentBegin and b <= end:
            return '"{}___{}"\n'.format(line.lower().strip().replace('"', ''), recentSpkr)

    lastSpkr, lastBegin, lastEnd = df.iloc[df.shape[0]-1]['speaker'], np.floor(df.iloc[df.shape[0]-1]['begin'] * 0.001), np.ceil(df.iloc[df.shape[0]-1]['end'] * 0.001)
    if a >= lastBegin and b >- lastEnd:
        return '"{}___{}"\n'.format(line.lower().strip().replace('"', ''), lastSpkr)

    # Sometimes things just don't line up perfectly
    beforeBegin, afterEnd, iList = None, None, list()
    for i, row in df.iterrows():
         
        spkr, begin, end = row['speaker'], np.floor(row['begin'] * 0.001), np.ceil(row['end'] * 0.001)
        if begin <= a and end >= a:
            beforeBegin = begin
            iList.append(i)
        
        if end >= b:
            afterEnd = end

        if not isinstance(beforeBegin, type(None)) and not isinstance(afterEnd, type(None)):
            iList.append(i)
            break

    # Final catch-all that works for the corpus
    spkrList = df.iloc[iList[0]:iList[-1]+1]['speaker'].tolist()
    spkr = max(set(spkrList), key = spkrList.count)
    return '"{}___{}"\n'.format(line.lower().strip().replace('"', ''), spkr)


def traverseFile(textgrid, df):

    outList, startData, invList, startWord, counter = list(), list(), list(), False, 0
    with open(textgrid, 'r') as f:
        lastLine = "" # Keep an eye on what we saw previous to this line
        for line in f:
            outList.append(line)
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
                    if line == '"sp"\n':
                        outList[-1] = '"SIL"\n'
                    else:
                        outList[-1] = figureINVPAR(line, a, b, df)
                    counter = 10 # Reset to the beginning of a new triplet
            else:
                if line == '"word"\n' and lastLine == '"IntervalTier"\n': # Make sure we skip over the 'phone' IntervalTier
                    startWord = True
                lastLine = line

    outFile = textgrid.split('.')[0] + '.TextGrid.Fixed'
    with open(outFile, 'w') as f:
        for line in outList:
            f.write(line)


def reSegment(files, textgrids, segmentations):

    for file in files:
        basename = os.path.basename(file).split('.')[0]
        # Files without segmentation are all participant
        if any([basename in segmentation for segmentation in segmentations]):
            df = pd.read_csv(segmentations[0].rsplit('/', 1)[0] + '/{}.csv'.format(basename))
        else:
            df = None

        traverseFile(textgrids[0].rsplit('/', 1)[0] + '/{}.TextGrid'.format(basename), df)


def main():

    dataDir = "/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSSo-IS2021-data"
    for partition in ['train', 'test']:
        if partition == 'train':
            ddir = os.path.join(dataDir, partition, 'progression', partition, 'audio')
            for condition in ['decline', 'no_decline']:
                files = glob.glob(os.path.join(ddir, condition, '*.wav'))
                textgrids = glob.glob(os.path.join(ddir, condition, 'textgrids/*.TextGrid'))
                segmentations = glob.glob(os.path.join(ddir, '../segmentation', condition, '*.csv'))

                reSegment(files, textgrids, segmentations)

        else:

            ddir = os.path.join(dataDir, partition, 'progression', 'test-dist', 'audio')
            files = glob.glob(os.path.join(ddir, '*.wav'))
            textgrids = glob.glob(os.path.join(ddir, 'textgrids/*.TextGrid'))
            segmentations = glob.glob(os.path.join(ddir, '../segmentation/*'))

            reSegment(files, textgrids, segmentations)


if __name__ == "__main__":

    main()
