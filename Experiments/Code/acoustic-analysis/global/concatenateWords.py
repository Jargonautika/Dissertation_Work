#!/usr/bin/env python3

import os
import numpy as np


def concatenateWords(sig, segments, fs):

    wordList = list()
    for start, finish in segments:
        a = int(np.floor(start * fs))
        b = int(np.ceil(finish * fs))

        # Determine the number of integers to be included
        span = b - a
        # Get the whole array of interviewer speech segment from the correct start to the correct end
        wordInts = np.linspace(a, b, num = span, dtype = int)

        for wordInt in wordInts:
            if wordInt < len(sig): # Protection for the end of the signal
                wordList.append(sig[wordInt])

    return np.array(wordList)


# Get a list of tuples with start and end times of all of 
# the words in the file not spoken by the interviewer
def findWords(tg, returnString = 0): # 0 for just the word time stamps
                                     # 1 for the word time stamps and the word
                                     # 2 for silence time stamps

    startData, wordList, startWord = list(), list(), False
    a, b, counter = 0.0, 0.0, 0
    with open(tg, 'r') as f:
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
                    if returnString == 2:
                        if "SIL" in line: # Get the silences
                            wordList.append((a, b))
                    else:
                        # Remove interviewer and all instances of 'laughs', 'sings', 'uh', 'um', 'clearsthroat', etc.
                        if "INV" not in line and "&" not in line and "SIL" not in line:
                            if returnString == 0:
                                wordList.append((a, b))
                            elif returnString == 1:
                                wordList.append((a, b, line.strip()))

                            
                    counter = 10 # Reset to the beginning of a new triplet
            else:
                if line == '"word"\n' and lastLine == '"IntervalTier"\n': # Make sure we skip over the 'phone' IntervalTier
                    startWord = True
                lastLine = line

    return wordList


def main(tmpFile, partition, condition, sig, fs):

    tgDir = "/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/TextGrids/"
    id = os.path.basename(tmpFile).split('.')[0].split('-')[0]

    if partition == "train":
        tg = os.path.join(tgDir, partition, condition, '{}.TextGrid'.format(id))

    else: 
        tg = os.path.join(tgDir, partition, '{}.TextGrid'.format(id))

    words = findWords(tg)

    concatSig = concatenateWords(sig, words, fs)

    return concatSig



if __name__ == "__main__":

    main()
