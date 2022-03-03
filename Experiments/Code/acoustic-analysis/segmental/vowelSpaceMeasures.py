#!/usr/bin/env python3

import os
import math
import string
import parselmouth
import numpy as np


def getFormantRanges(F1, F2, F3):

    order = list(F1.keys())
    F1List, F2List, F3List = list(), list(), list()
    for timePoint in range(5):
        minF1 = np.nanmin([np.nanmin(F1[vowel][timePoint::5][0]) if len(F1[vowel][timePoint::5]) > 0 else np.nan for vowel in order])
        maxF1 = np.nanmax([np.nanmax(F1[vowel][timePoint::5][0]) if len(F1[vowel][timePoint::5]) > 0 else np.nan for vowel in order])
        F1List.append(maxF1 - minF1)

        minF2 = np.nanmin([np.nanmin(F2[vowel][timePoint::5][0]) if len(F2[vowel][timePoint::5]) > 0 else np.nan for vowel in order])
        maxF2 = np.nanmax([np.nanmax(F2[vowel][timePoint::5][0]) if len(F2[vowel][timePoint::5]) > 0 else np.nan for vowel in order])
        F2List.append(maxF2 - minF2)

        minF3 = np.nanmin([np.nanmin(F3[vowel][timePoint::5][0]) if len(F3[vowel][timePoint::5]) > 0 else np.nan for vowel in order])
        maxF3 = np.nanmax([np.nanmax(F3[vowel][timePoint::5][0]) if len(F3[vowel][timePoint::5]) > 0 else np.nan for vowel in order])
        F3List.append(maxF3 - minF3)

    return F1List, F2List, F3List


#unit normal vector of plane defined by points a, b, and c
def unit_normal(a, b, c):
    x = np.linalg.det([[1,a[1],a[2]],
         [1,b[1],b[2]],
         [1,c[1],c[2]]])
    y = np.linalg.det([[a[0],1,a[2]],
         [b[0],1,b[2]],
         [c[0],1,c[2]]])
    z = np.linalg.det([[a[0],a[1],1],
         [b[0],b[1],1],
         [c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    return (x/magnitude, y/magnitude, z/magnitude)


# https://stackoverflow.com/questions/12642256/find-area-of-polygon-from-xyz-coordinates
# https://en.wikipedia.org/wiki/Stokes%27_theorem
def PolyArea3D(poly):

    if len(poly) < 3: # not a plane - no area
        return 0
    total = [0, 0, 0]
    N = len(poly)
    for i in range(N):
        vi1 = poly[i]
        vi2 = poly[(i+1) % N]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]

    result = np.dot(total, unit_normal(poly[0], poly[1], poly[2]))

    return abs(result/2)


def getVowelArea3D(F1, F2, F3):

    vowelAreasByTime = list()

    # Make sure the order is the same so that we have x and y coordinates for the appropriate F1 and F2
    order = list(F1.keys())
    for timePoint in range(5):
        meanF1s = [np.nanmean(F1[vowel][timePoint::5]) for vowel in order]
        meanF2s = [np.nanmean(F2[vowel][timePoint::5]) for vowel in order]
        meanF3s = [np.nanmean(F3[vowel][timePoint::5]) for vowel in order]

        polygon = list()
        for first, second, third in zip(meanF1s, meanF2s, meanF3s):

            if not math.isnan(first) and not math.isnan(second) and not math.isnan(third):
                
                polygon.append((first, second, third))

        vowelAreasByTime.append(PolyArea3D(polygon))

    return vowelAreasByTime

# https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
# https://pubs.asha.org/doi/abs/10.1044/1092-4388%282008/041%29
# https://en.wikipedia.org/wiki/Shoelace_formula
def PolyArea2D(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def getVowelArea2D(F1, F2):

    vowelAreasByTime = list()

    # Make sure the order is the same so that we have x and y coordinates for the appropriate F1 and F2
    order = list(F1.keys())
    for timePoint in range(5):
        meanF1s = [np.nanmean(F1[vowel][timePoint::5]) for vowel in order]
        meanF2s = [np.nanmean(F2[vowel][timePoint::5]) for vowel in order]

        x, y = list(), list()
        for first, second in zip(meanF1s, meanF2s):

            if not math.isnan(first) and not math.isnan(second):
                
                x.append(first)
                y.append(second)

        vowelAreasByTime.append(PolyArea2D(x, y))

    return vowelAreasByTime


def getVocalicFormants(burg, i):

    outList = list()
    # Get Formants 1, 2, 3
    for j in range(1, 4):
        bark = burg.get_value_at_time(formant_number = j, time = i, unit = parselmouth.FormantUnit.HERTZ)
        outList.append(bark)

    return outList


def getVowelMidpoints(basename, sound, vowels):

    vowelCounter = 0
    F1, F2, F3 = dict(), dict(), dict()
    for vowel in vowels:

        vF1, vF2, vF3 = list(), list(), list()

        for start, end in vowels[vowel]:

            vowelCounter += end - start

            # Extract only the part for the short files
            if "-" in basename:
                _, _, startSIL, _, _, offsetA, _ = basename.split('_')[0].split('-')
                x = (int(startSIL) / 1000) + (int(offsetA) / 1000)
                startOffset = start - x
                endOffset = end - x

                spaces = np.linspace(0, endOffset - startOffset, num = 7)
                part = sound.extract_part(from_time = startOffset, to_time = endOffset) # Starts over at time 0

                burg = part.to_formant_burg()
                formants = np.array([getVocalicFormants(burg, i) for i in spaces[1:-1]]).flatten().tolist()

            # Just use the whole thing
            else:
                spaces = np.linspace(start, end, num = 7)
                part = sound.extract_part(from_time = start, to_time = end)

                burg = sound.to_formant_burg()
                formants = np.array([getVocalicFormants(burg, i) for i in spaces[1:-1]]).flatten().tolist()

            vF1.append(formants[::3])
            vF2.append(formants[1::3])
            vF3.append(formants[2::3])

        F1[vowel] = vF1
        F2[vowel] = vF2
        F3[vowel] = vF3

    # Given as the ratio of total vowel duration to the total duration of the file
    vowelRate = vowelCounter / sound.duration

    return vowelRate, F1, F2, F3


def locateTextGrid(wav):

    id = "{}.TextGrid".format(wav.split('-')[0].split('/')[-1])
    startDir = "/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/TextGrids"
    for root, dirs, files in os.walk(startDir, topdown=False):
        if id in files:
            return os.path.join(root, files[files.index(id)])


def removePhonesFromInterviewer(basename, tg, vowels):

    startWord = False
    startData = list()
    counter = 0
    a, b = 0.0, 0.0

    invList = list()

    with open(tg, 'r') as f:
        lastLine = ""
        for line in f:
            if startWord:
                if counter == 0:
                    startData.append(0)
                    counter += 1
                elif counter == 1:
                    startData.append(float(line.strip()))
                    counter += 1
                elif counter == 2:
                    startData.append(int(line.strip()))
                    counter = 10
                elif counter == 10:
                    a = float(line.strip())
                    counter += 1
                elif counter == 11:
                    b = float(line.strip())
                    counter += 1
                elif counter == 12:
                    if "INV" in line:
                        invList.append((a,b))
                    counter = 10
            else:
                if line == '"word"\n' and lastLine == '"IntervalTier"\n':
                    startWord = True
                lastLine = line

    returnVowels = dict()
    for vowel in vowels:
        goodList = list()
        for start, end in vowels[vowel]:
            if isinstance(start, tuple):
                s = start[1]
                e = end[-1]
            else:
                s = start
                e = end
            include = True
            for startInv, endInv in invList:
                if ((startInv <= s <= endInv) and (startInv <= e <= endInv)):
                    include = False
            if include:
                goodList.append((start, end))
        if len(goodList) > 0:
            returnVowels[vowel] = goodList

    return returnVowels


def getVowels(basename):

    tg = locateTextGrid(basename)

    startWord, isSegment = False, False
    startData = list()
    counter = 0
    totalDur = 0.0
    a, b = 0.0, 0.0
    vowels = dict()
    # We're dealing with a normalized file
    if "-" in basename:
        _, _, startSIL, _, _, offsetA, offsetB = basename.split('_')[0].split('-')
        startPoint = int(startSIL)
        endPoint = int(startSIL) + int(offsetB)
    # We're dealing with a full file
    else:
        startPoint = None
        endPoint = None
        isSegment = True

    with open(tg, 'r') as f:
        lastLine = ""
        for line in f:
            if line == '"IntervalTier"\n' and b == totalDur:
                if "-" in basename:
                    return vowels
                else:
                    return removePhonesFromInterviewer(basename, tg, vowels)
            if line == '<exists>\n':
                totalDur = float(lastLine.strip())
            if startWord:
                if counter == 0:
                    startData.append(0)
                    counter += 1
                elif counter == 1:
                    startData.append(float(line.strip()))
                    counter += 1
                elif counter == 2:
                    startData.append(int(line.strip()))
                    counter = 10
                elif counter == 10:
                    a = float(line.strip())
                    counter += 1
                elif counter == 11:
                    b = float(line.strip())
                    counter += 1
                elif counter == 12:
                    if not isinstance(startPoint, type(None)) and a < (startPoint / 1000) < b:
                        isSegment = True
                    elif not isinstance(endPoint, type(None)) and a < (endPoint / 1000) < b:
                        isSegment = False
                    # It's a vowel we're interested in
                    # LG is laughter, NS is noise, CG is cough, LS is lip smack, and SL can be silence  (https://phon.wordpress.ncsu.edu/lab-manual/forced-alignment/)
                    if isSegment:
                        if line != '"sil"\n' and line != '"lg"\n' and line != '"ns"\n' and line != '"cg"\n' and line != '"ls"\n' and line != '"sl"\n' and any(map(str.isdigit, line.strip())):
                            vowel = line.strip().translate(str.maketrans('', '', string.punctuation))[:-1]
                            if vowel not in vowels:
                                vowels[vowel] = list()
                            vowels[vowel].append((a,b))
                    counter = 10
            else:
                if line == '"phones"\n' and lastLine == '"IntervalTier"\n':
                    startWord = True
                lastLine = line


def vowelWork(basename, sound):

    # Get the relevant vowels in the file
    vowels = getVowels(basename)

    # Vowel Midpoints (5) F1, F2, and F3 values
    vowelRate, F1, F2, F3 = getVowelMidpoints(basename, sound, vowels)

    # 2D Vowel Space Area
    vowelArea2D = getVowelArea2D(F1, F2)

    # 3D Vowel Space Area
    vowelArea3D = getVowelArea3D(F1, F2, F3)

    # Ranges from first three formants
    rangeF1, rangeF2, rangeF3 = getFormantRanges(F1, F2, F3)

    return vowelRate, vowelArea2D, vowelArea3D, np.nanmean(rangeF1), np.nanmean(rangeF2), np.nanmean(rangeF3)


def main(file, wav):

    # Get the file's basename
    basename = os.path.basename(file).split('.')[0]

    # Create a parselmouth sound object
    sound = parselmouth.Sound(file)

    # Substitute the normalized data for parselmouth's data
    sound.values = wav.getData().T

    vowelRate, vA2D, vA3D, F1, F2, F3 = vowelWork(basename, sound)

    return vowelRate, np.nanmean(vA2D), np.nanmean(vA3D), F1, F2, F3


if __name__ == "__main__":

    # main("/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cc/S001.wav")
    main("/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Data/ADReSS-IS2020-data/train/Normalised_audio-chunks/cc/S077-240-9902-11373-1-0-1390.wav")
