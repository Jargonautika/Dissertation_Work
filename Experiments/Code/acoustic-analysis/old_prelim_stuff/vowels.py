#!/usr/bin/env python3

from normalityTests import makeHistograms, zscoreFilter
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy import stats
import pandas as pd
import significanceTests
import numpy as np
import warnings
import math
import os

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore', category = FutureWarning)


# z_score filter by both F1 and F2
# http://www.ltcconline.net/greenl/courses/201/estimation/smallConfLevelTable.htm (1.96 is likely most common to get 95% of the data)
def z_score_by_f(df, z = 2.58):

    for measure in ['F1', 'F2']:
        scores = stats.zscore(df[measure])
        df['z_score_{}'.format(measure)] = scores
    filteredDF = df[(df['z_score_F1'].abs()<=z) & (df['z_score_F2'].abs()<=z)].drop(['z_score_F1', 'z_score_F2'], axis = 1) # 3 is too lenient and allows ~99.9999999% of everything in

    return filteredDF


def byTimeStamp(df, timeStamp, segments, exp_dir, name, z = 2.58):

    figure, axes = generate_subplots(len(segments), row_wise = True)

    # Iterate over the segments
    for segment, ax in zip(segments, axes):

        miniDF = df[df['Segment'] == segment]
        ccMini = miniDF[miniDF['Condition'] == 0]
        cdMini = miniDF[miniDF['Condition'] == 1]

        # Find rows where both F1 and F2 at our timestamp have real values
        filteredCC = z_score_by_f(ccMini[ccMini['Rep'] == timeStamp], z)
        filteredCD = z_score_by_f(cdMini[cdMini['Rep'] == timeStamp], z)

        # Save out by z score for R usage later
        # Save out the z score filtered data frame for this segment and measure for analysis in R later
        outputPath = '{}/data/acoustics/{}/{}'.format(exp_dir, str(z), 'vowelSpace')
        if not os.path.isdir(outputPath):
            os.mkdir(outputPath)
        combo = pd.concat([filteredCC, filteredCD])
        combo.to_csv(outputPath + '/{}_{}.csv'.format(segment, timeStamp), index = False)

        ccX = filteredCC['F2'].tolist() # F2 should be on the x-axis and high frequencies go at the left
        ccY = filteredCC['F1'].tolist() # F1 should be on the y-axis and high frequencies go at the top
        cdX = filteredCD['F2'].tolist()
        cdY = filteredCD['F1'].tolist()

        # for lilXYList in [ccX, ccY, cdX, cdY]:
        #     lilXYList.reverse()

        ax.scatter(ccX, ccY, s = 0.2, marker = '^', c = '#0868ac', alpha = 0.5, label = 'CC') # x-axis is F2; y-axis is F1
        ax.scatter(cdX, cdY, s = 0.2, marker = 'o', c = '#bae4bc', alpha = 0.5, label = 'CD')
        confidence_ellipse(np.array(ccX), np.array(ccY), ax, edgecolor='#0868ac') # Control
        confidence_ellipse(np.array(cdX), np.array(cdY), ax, edgecolor='#bae4bc') # Diagnosed
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.grid(True)
        ax.set_title(label=segment)

        # Run some brief significance statistics
        # Z = pd.concat([filteredCC, filteredCD])
        # crossTab = pd.crosstab(Z['Condition'], Z[measure]) # TODO start here. 
        # significanceTests.main(ccX, cdX, "{}_{}_{}_F2".format(name, timeStamp, segment))
        # significanceTests.main(ccY, cdY, "{}_{}_{}_F1".format(name, timeStamp, segment))

    figure.subplots_adjust(top=0.88)
    if timeStamp == 6:
        timeStamp = 'Avg'
    figure.suptitle('{} Vowel Space at Timestamp {}'.format(name, timeStamp))
    lines_labels = [ax.get_legend_handles_labels() for ax in figure.axes]
    handles, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    figure.legend(handles, labels, title = 'Legend', bbox_to_anchor=(1.05, 1), loc='upper left',)
    plt.tight_layout()
    plt.savefig("{}/reports/plots/vowelSpace/{}/{}_{}.jpg".format(exp_dir, str(z), name, timeStamp))
    plt.clf()


# Make a confidence ellipse around each vowel space
# https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):

    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# Stolen functions to define an arbitrary number of subplots
# https://stackoverflow.com/questions/28738836/matplotlib-with-odd-number-of-subplots
def choose_subplot_dimensions(k):
    if k < 4:
        return k, 1
    elif k < 11:
        return math.ceil(k/2), 2
    else:
        # I've chosen to have a maximum of 3 columns
        return math.ceil(k/3), 3


# Another stolen function as from above
def generate_subplots(k, row_wise=False):

    nrow, ncol = choose_subplot_dimensions(k)
    # Choose your share X and share Y parameters as you wish:
    figure, axes = plt.subplots(nrow, ncol,
                                sharex=True,
                                sharey=True)

    # Check if it's an array. If there's only one plot, it's just an Axes obj
    if not isinstance(axes, np.ndarray):
        return figure, [axes]
    else:
        # Choose the traversal you'd like: 'F' is col-wise, 'C' is row-wise
        axes = axes.flatten(order=('C' if row_wise else 'F'))

        # Delete any unused axes from the figure, so that they don't show
        # blank x- and y-axis lines
        for idx, ax in enumerate(axes[k:]):
            figure.delaxes(ax)

            # Turn ticks on for the last ax in each column, wherever it lands
            idx_to_turn_on_ticks = idx + k - ncol if row_wise else idx + k - 1
            for tk in axes[idx_to_turn_on_ticks].get_xticklabels():
                tk.set_visible(True)

        axes = axes[:k]
        return figure, axes


def makeVowelPlots(df, name, segments, exp_dir, z = 2.58):

    # Iterate over the number of observations extracted from each vowel at equally-spaced time-stamps
    for timeStamp in range(1,7): # 6 here means average

        byTimeStamp(df, timeStamp, segments, exp_dir, name, z)


def defineNaturalClasses(df):

    # Each consonant is found within each subset dataframe by condition
    segments = set(df['Segment'])

    # These are, as far as I can tell, all of the possible vocalic natural classes
    # by frontness for the ARPABET consonant set
    front = {'EH', 'IH', 'IY', 'AE'}
    central = {'ER', 'AX'}
    back = {'UW', 'UH', 'AA', 'AO', 'AH'}

    # These are, as far as I can tell, all of the possible vocalic natural classes
    # by HEIGHT for the ARPABET consonant set
    close = {'UW', 'IY'}
    mid = {'IH', 'UH', 'EH', 'ER', 'AO', 'AX', 'AH'}
    open = {'AA', 'AE'}

    diphthongs = {'OW', 'AW', 'OY', 'AY', 'EY'}
    
    # Return intersections
    namesList = ['allSegments', 'fronts', 'centrals', 'backs', 'closes',
                 'mids', 'opens', 'diphthongs']

    segmentSets = [segments,
                   segments.intersection(front),
                   segments.intersection(central),
                   segments.intersection(back),
                   segments.intersection(close),
                   segments.intersection(mid),
                   segments.intersection(open),
                   segments.intersection(diphthongs),
                  ]

    # Also do them all individually, irrespective of time stamp
    for segment in segments:
        namesList.append(segment)
        segmentSets.append({segment})

    return namesList, segmentSets
    

def main(df, vPlotDF, exp_dir, z = 2.58):

    ## Make histograms by condition
    descriptiveStats = list()
    namesList, segmentSets= defineNaturalClasses(df)
    for name, segments in zip(namesList[:], segmentSets[:]):

        # Simple histograms for duration and time collapsed all itself
        X = makeHistograms(df, name, segments, exp_dir, 'vowel', z)
        for x in X:
            descriptiveStats.append(x)

        # Simple histograms for everything else, broken down by time in the vowel
        filteredDF = df[df['Measure'] != 'Duration']
        M = filteredDF['Measure'].tolist()
        R = filteredDF['Rep'].tolist()
        MR = ["{}_{}".format(i, j) for i, j in zip(M,R)]
        filteredDF = filteredDF.drop(['Measure', 'Rep'], axis = 1)
        filteredDF['Measure'] = MR
        
        X = makeHistograms(filteredDF, name, segments, exp_dir, 'vowel', z)
        for x in X:
            descriptiveStats.append(x)

        # Plot and calculate the 2D vowel space
        makeVowelPlots(vPlotDF, name, segments, exp_dir, z)


    return descriptiveStats

# TODO Plot all of the vowels for the Control condition on one graph; do the same for the Diagnosed condition
    # Calculate the area of the vowel space
    # Control SHOULD have less confusion between other vowels during production

# TODO # Look at the regularity of F0 in terms of Jitter and Shimmer and Standard Deviation (may be correlated with the dynamic range) as it changes over time
    # We will probably need to subsect by Male/Female due to dimorphia
    # Calculate differences in spectral tilt
    # Calculate diff in central gravity, energy, etc. (on F0)
