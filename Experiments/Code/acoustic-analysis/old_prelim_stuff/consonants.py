#!/usr/bin/env python3

from normalityTests import makeHistograms


def defineNaturalClasses(df):

    # Each consonant is found within each subset dataframe by condition
    segments = set(df['Segment'])

    # These are, as far as I can tell, all of the possible consonantal natural classes
    # by place of articulation for the ARPABET consonant set
    bilabials = {'B', 'P', 'M', 'W', 'WH'}                          # 5
    labiodentals = {'F', 'V'}                                       # 2
    dentals = {'T', 'TH', 'DH'}                                     # 3
    alveolars = {'CH', 'JH', 'S', 'Z'}                              # 4
    postalveolars = {'D', 'N', 'R', 'DX', 'SH', 'ZH', 'L'}          # 7
    palatals = {'Y'}                                                # 1
    velars = {'K', 'G', 'NX', 'NG'}                                 # 4
    glottals = {'H', 'Q'}                                           # 2
                                                                    ## 28

    # These are, as far as I can tell, all of the possible consonantal natural classes
    # by manner of articulation for the ARPABET consonant set
    plosives = {'P', 'B', 'T', 'D', 'K', 'G', 'Q'}                  # 7
    nasals = {'M', 'N', 'NX', 'NG'}                                 # 4
    flaps = {'DX'}                                                  # 1
    fricatives = {'F', 'V', 'TH', 'DH', 'S', 'Z', 'SH', 'ZH', 'H'}  # 9
    affricates = {'CH', 'JH'}                                       # 2
    approximants = {'R', 'L', 'Y', 'W', 'WH'}                       # 5
                                                                    # 28
    
    # Return intersections
    namesList = ['allSegments', 'bilabials', 'labiodentals', 'dentals', 'alveolars',
                 'postalveolars', 'palatals', 'velars', 'glottals', 'plosives',
                 'nasals', 'flaps', 'fricatives', 'affricates', 'approximants']

    segmentSets = [segments,
                   segments.intersection(bilabials),
                   segments.intersection(labiodentals),
                   segments.intersection(dentals),
                   segments.intersection(alveolars),
                   segments.intersection(postalveolars),
                   segments.intersection(palatals),
                   segments.intersection(velars),
                   segments.intersection(glottals),
                   segments.intersection(plosives),
                   segments.intersection(nasals),
                   segments.intersection(flaps),
                   segments.intersection(fricatives),
                   segments.intersection(affricates),
                   segments.intersection(approximants)
                  ]

    return namesList, segmentSets
    

def main(df, exp_dir, z = 2.58):

    # Make histograms by condition
    descriptiveStats = list()
    namesList, segmentSets = defineNaturalClasses(df)
    for name, segments in zip(namesList[:], segmentSets[:]):

        X = makeHistograms(df, name, segments, exp_dir, 'consonant', z)
        for x in X:
            descriptiveStats.append(x)
    
    return descriptiveStats
