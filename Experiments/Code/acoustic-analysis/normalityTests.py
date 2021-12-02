#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from matplotlib import pyplot as plt
pd.options.mode.chained_assignment = None

# Set the palette to the "colorblind" default palette:
sns.set_palette("colorblind")
sns.set_style("white")
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']


def calculateDescriptiveStats(x, ttype, segment, condition, measure):

    # x has to be less than 5000 and longer than 3 to make the Shapiro-Wilk Normalcy Test make sense and have the p-value mean something
    if not 3 <= len(x) < 5000:
        return None
    else:

        # Descriptive Statistics
        median = np.median(x)
        mean = np.mean(x)
        standardError = stats.sem(x) # DDOF = 1
        variance = np.var(x)
        std = np.std(x)
        coefVar = std / mean # The coefficient of variation is the ratio of the standard deviation to the mean

        # Tests for Normality of data distribution
        skewness = stats.skew(x)
        skew2SE = skewness / (2 * standardError) # If this is greater than abs(1) then then it is significant at p < 0.05; 
                                                # if it's greater than abs(1.29) then it's significant at p < 0.01. Interpret cautiously
                                                # since we have a limited number of samples @@Field 2012 Chapter 05
        kurtosis = stats.kurtosis(x)
        kurt2SE = kurtosis / (2 * standardError)
        WilkTestStatistic, pValue = stats.shapiro(x) # If this p-value is lower than 0.05 then those data are not normally distributed. We can
                                                    # have pretty high confidence of this since in most cases the number of data points is pretty low. 

        returnList = [ttype, segment, condition, measure]
        for value in [len(x), median, mean, standardError, variance, std, coefVar, skewness, skew2SE, kurtosis, kurt2SE, WilkTestStatistic, pValue]:
            returnList.append(round(value, 5))
        return returnList


# TODO Do significance testing with the spectral tilt of the spectra and the histograms by spectral moment
def plotHist(cc, cd, segment, measure, exp_dir, name, ttype, z = 2.58, x_axis_label = 'Frequency in Hz'):

    if len(cc) < 20 or len(cc) < 20:
        print('Measure {} for segment {} only has {} and {} observations for the control and diagnosed conditions, respectively. Passing.'.format(measure, segment, len(cc), len(cd)))
        return

    fig, axs = plt.subplots(2, 2)

    # Histograms and Density Plots
    kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})
    sns.distplot(cc, color = CB_color_cycle[0], label="cc", ax=axs[0,0], **kwargs)
    sns.distplot(cd, color = CB_color_cycle[2], label="cd", ax=axs[1,0], **kwargs)
    axs[0,0].set_xlabel('{}'.format(x_axis_label))
    axs[0,0].set_title('Control Condition')
    axs[0,0].legend(loc='upper right')
    axs[1,0].set_xlabel('{}'.format(x_axis_label))
    axs[1,0].set_title('Diagnosed Condition')
    axs[1,0].legend(loc='upper right')

    # Accompanying Quantile-Quantile Plots
    sm.qqplot(np.array(cc), stats.t, marker = '.', markerfacecolor = CB_color_cycle[0], markeredgecolor = CB_color_cycle[0], fit=True, line="s", ax = axs[0,1], alpha = 0.3)
    sm.qqplot(np.array(cd), stats.t, marker = '.', markerfacecolor = CB_color_cycle[2], markeredgecolor = CB_color_cycle[2], fit=True, line="s", ax = axs[1,1], alpha = 0.3)
    axs[0,1].get_lines()[1].set_color(CB_color_cycle[5])
    axs[1,1].get_lines()[1].set_color(CB_color_cycle[5])

    fig.suptitle('{}: {} {} {} at z = {}'.format(ttype, name, segment, measure, str(z)))
    fig.tight_layout()

    fig.savefig("{}/reports/plots/histograms/{}/{}_{}_{}_{}.jpg".format(exp_dir, str(z), name, ttype, segment, measure))
    plt.close(fig)


# Filter for Z-score with no outliers outside of n standard deviations from the mean
# https://kanoki.org/2020/04/23/how-to-remove-outliers-in-python/
# NOTE let's not do this so we have a clearer picture. We'll do smarter filtering later. May 2021
# NOTE NOTE LOL. Not doing this means that exactly 0 of the factors are normally distributed. June 2021
def zscoreFilter(ccCons, measure, z = 2.58):

    ccZ = ccCons[ccCons['Measure'] == measure] # Subset to the measure we're interested in at the moment
    ccZ = ccZ[ccZ['Value'].notnull()]          # Make sure we don't have NaN values
    scores = stats.zscore(ccZ['Value'])
    ccZ = ccZ.reindex(columns = ccZ.columns.to_list() + ['z_score'])
    for i, j in zip(ccZ['z_score'].index , scores):
        ccZ['z_score'][i] = j
    cc = ccZ.loc[ccZ['z_score'].abs()<=z]

    return cc


def iterateMeasures(ccCons, cdCons, segments, exp_dir, name, ttype, measure, z = 2.58):

    # Slightly more nuanced attempt to solve the normality issue. 
    ccDF = zscoreFilter(ccCons, measure, z)
    cdDF = zscoreFilter(cdCons, measure, z)

    cc = ccDF['Value'].to_list()
    cd = cdDF['Value'].to_list()

    # Save out the z score filtered data frame for this segment and measure for analysis in R later
    outputPath = '{}/data/acoustics/{}/{}'.format(exp_dir, str(z), ttype)
    if not os.path.isdir(outputPath):
        os.mkdir(outputPath)
    combo = pd.concat([ccDF, cdDF]).drop(['z_score'], axis = 1)
    combo.to_csv(outputPath + '/{}_{}.csv'.format(name, measure), index = False)

    if measure == 'Duration':
        plotHist(cc, cd, '-'.join(segments), measure, exp_dir, name, ttype, z, x_axis_label= 'Seconds')
    else:
        plotHist(cc, cd, '-'.join(segments), measure, exp_dir, name, ttype, z)

    x = calculateDescriptiveStats(cc, ttype, name, 'cc', measure)
    y = calculateDescriptiveStats(cd, ttype, name, 'cd', measure)

    # Sometimes we'll end up with vectors which violate some normality testing assumptions 
    # (we have too few or too many data points)
    if x and y:

        # Calculate homogeneity/heterogeneity of variance
        levene, pValue = stats.levene(cc, cd, center = 'mean') # TODO Switch this to median or trimmed if the tails are insanely right-skewed
        return x + [round(levene, 5), round(pValue, 5)], y + [round(levene, 5), round(pValue, 5)]
    
    else:
        return None, None

    # Z['discrete'] = pd.cut(Z[measure], bins=bins)
    # crossTab = pd.crosstab(Z['discrete'], Z[measure])
    # significanceTests.main(cc, cd, crossTab, "{}_{}_{}_{}".format(name, ttype, '-'.join(segments), measure))


def makeHistograms(df, name, segments, exp_dir, ttype, z = 2.58):

    # Save out basic descriptive statistics to compare CC to CD conditions
    outList = list()

    # Sometimes there's an empty intersection
    if len(segments) == 0:
        return list()

    # Break the dataframe into 'control' (cc/0) and 'diagnosed' (cd/1) conditions
    ccDF = df[df['Condition'] == 0]
    cdDF = df[df['Condition'] == 1]

    # Capture the histograms of all of the individual segments
    if name == 'allSegments':
        for segment in segments:

            ccCons = ccDF[ccDF['Segment'] == segment]
            cdCons = cdDF[cdDF['Segment'] == segment]

            for measure in list(set(df['Measure'])):

                x, y = iterateMeasures(ccCons, cdCons, segment, exp_dir, segment, ttype, measure, z)
                if x and y:
                    outList.append(x)
                    outList.append(y)

    # Capture the differences between natural classes
    else:

        ccCons = ccDF[ccDF['Segment'].isin(list(segments))]
        cdCons = cdDF[cdDF['Segment'].isin(list(segments))]

        for measure in list(set(df['Measure'])):

            x, y = iterateMeasures(ccCons, cdCons, segments, exp_dir, name, ttype, measure, z)
            if x and y:
                outList.append(x)
                outList.append(y)

    return outList   
