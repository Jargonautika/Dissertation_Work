#!/usr/bin/env python3

import re
import numpy as np


def trimAudio(wav, segments):

    data = wav.getData()
    fs = wav.getSamplingRate()
    bits = wav.getBitsPerSample()

    # Iterate over the segments to be cut backwards so that we don't mess up the indices
    for start, finish in reversed(segments):
        a = int(np.floor(start * fs))
        b = int(np.ceil(finish * fs))

        # Determine the number of integers to be deleted
        span = b - a
        # Get the whole array of interviewer speech segment from the correct start to the correct end
        invInts = np.linspace(a, b, num = span, dtype = int)

        # Update data
        data = np.delete(data, invInts)

    return data, fs, bits


def findInterviewerSegments(tg):

    startData, invList, startWord = list(), list(), False
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
                    if "INV" in line:
                        invList.append((a, b))
                    counter = 10 # Reset to the beginning of a new triplet
            else:
                if line == '"word"\n' and lastLine == '"IntervalTier"\n': # Make sure we skip over the 'phone' IntervalTier
                    startWord = True
                lastLine = line

    return invList


# See if it exists before we send it through
def grepINV(textGrid):

    pattern = "INV"

    file = open(textGrid, "r")
    for line in file:
        if re.search(pattern, line):
            return True
    
    return False


# We need to get rid of all speech from "___INV" segments
def main(wav, textGrid, tmpFolder):

    # Find out if "___INV" is in the file
    if grepINV(textGrid):

        # Get a list of areas that need to be deleted
        invList = findInterviewerSegments(textGrid)

        # Remove the offending areas
        data, _, _ = trimAudio(wav, invList)
        
        # Overwrite the old stuff
        wav.__data = data

    return wav


if __name__ == "__main__":

    main('/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cc/S051.wav', 
         '/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/TextGrids/train/cc/S051.TextGrid')
