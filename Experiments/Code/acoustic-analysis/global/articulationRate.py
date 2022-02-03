#!/usr/bin/env python3

import concatenateWords
import syllables
import os


def postProcessWord(word):

    return word.replace("___PAR", "").replace("\\", "").replace('"', "")


def getWords(tmpFile, partition, condition):

    tgDir = "/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/TextGrids/"
    id = os.path.basename(tmpFile).split('.')[0].split('-')[0]

    if partition == "train":
        tg = os.path.join(tgDir, partition, condition, '{}.TextGrid'.format(id))

    else: 
        tg = os.path.join(tgDir, partition, '{}.TextGrid'.format(id))

    wordList = concatenateWords.findWords(tg, returnString = 1)
    return wordList


def main(tmpFile, partition, condition):

    # Get all the words except hesitation, exclamation, and/or noise
    wordList = getWords(tmpFile, partition, condition)

    # Determine the duration of all these words
    durs = [b-a for a, b, _ in wordList]

    # Get the words out of the tuple
    words = [c for _, _, c in wordList]

    # Count syllables for each word
    syllabified = [syllables.estimate(postProcessWord(word)) for word in words]

    # Calculate # syllables per second in the file
    articulationRate = sum(syllabified) / sum(durs)

    return articulationRate


if __name__ == "__main_":

    main()
