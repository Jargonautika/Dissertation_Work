#!/usr/bin/env python3

import articulationRate
import concatenateWords
import os


def getSilences(tmpFile, partition, condition, tg):

    wordList = concatenateWords.findWords(tg, returnString = 2)
    return wordList


def main(tmpFile, partition, condition, tg):

    # Get all the silence durations
    silList = getSilences(tmpFile, partition, condition, tg)

    # Get all of the words spoken in the transcript by the participant
    wordList = articulationRate.getWords(tmpFile, partition, condition, tg)

    # Determine the duration of all these silences
    durs = [b-a for a, b in silList]

    # Get the duration of SIL pauses in the file
    totalSilence = sum(durs)

    # Count each word
    wordCount = len(wordList)

    # Calculate # syllables per second in the file
    pauseRate = totalSilence / wordCount

    return pauseRate


if __name__ == "__main_":

    main()

