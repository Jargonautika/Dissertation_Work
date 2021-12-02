#!/usr/bin/env python3

import os
import pickle
import joblib
import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

#https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def plotCorrelation(df, finalDF, pca, features, exp_dir, numComps): # TODO might be able to put a scatterplot of the actual data points (didn't work for Helen)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('Principal Component 1', fontsize = 10)
    ax.set_ylabel('Principal Component 2', fontsize = 10)
    ax.set_zlabel('Principal Component 3', fontsize = 10)

    colList = df.columns[:6]
    i = 0

    textLocations = list()
    colors = ["r", "b", "g"]

    while i < len(colList):
        PCnum = colList[i][0]
        feats = df[colList[i]]
        values = df[colList[i+1]]

        c = colors[int(PCnum[-1]) - 1]

        for f, v in zip(feats, values):

            xCoords = [np.mean(finalDF.loc[:, 'PC1']), pca.components_[0][features.index(f)]]
            yCoords = [np.mean(finalDF.loc[:, 'PC2']), pca.components_[1][features.index(f)]]
            zCoords = [np.mean(finalDF.loc[:, 'PC3']), pca.components_[2][features.index(f)]]

            a = Arrow3D(xCoords, yCoords, zCoords, mutation_scale=20, lw=0.5, arrowstyle="-|>", color=c)
            ax.add_artist(a)

            textLocation = [xCoords[1] + 0.05, yCoords[1] + 0.05, zCoords[1] + 0.05]

            # #Slightly adjust text location to avoid overlapping text
            for h in range(3):
                if h == 0:
                    adjustment = 0.01 # NOTE moves the arrow placement (Helen says it kinda works); takes trial and error
                elif h == 1:
                    adjustment = 0.01 # NOTE
                elif h == 2:
                    adjustment = 0.02 # NOTE
                j = [t for t in textLocations if t[h] > textLocation[h] - adjustment and t[h] < textLocation[h] + adjustment]
                while len(j) > 0:
                    textLocation[h] += adjustment
                    j = [t for t in textLocations if t[h] > textLocation[h] - adjustment and t[h] < textLocation[h] + adjustment]

            textLocations.append(textLocation)
            ax.text(textLocation[0], textLocation[1], textLocation[2], f, fontsize=7, color=c)

        i += 2

    ax.set_xlim([-.5, .5])
    ax.set_ylim([-.5, .5])
    ax.set_zlim([-.5, .5])

    plt.title("Top Features Correlated with First 3 PCs for {} PCs".format(numComps))
    plt.draw()
    
    plt.savefig("{}/vectors/pca/{}_factor_CorrelationCircle.png".format(exp_dir, numComps))
    plt.clf()


def plotPCs3D(finalDF, exp_dir, numComps):

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.set_xlabel('Principal Component 1', fontsize = 10)
    ax.set_ylabel('Principal Component 2', fontsize = 10)
    ax.set_zlabel('Principal Component 3', fontsize = 10)
    ax.set_title('3 component PCA', fontsize = 15)

    targets = ['I', 'N']
    colors = ['g', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDF['label'] == target
        ax.scatter3D(finalDF.loc[indicesToKeep, 'PC1']
                , finalDF.loc[indicesToKeep, 'PC2']
                , finalDF.loc[indicesToKeep, 'PC3']
                , alpha = 0.5
                , c = color
                , s = 50)
    ax.legend(targets)
    #ax.grid()
    plt.savefig("{}/vectors/{}_factor_3_axes_PCA.png".format(exp_dir, numComps))

    plt.clf()
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.set_xlabel('Principal Component 1', fontsize = 10)
    ax.set_ylabel('Principal Component 2', fontsize = 10)
    ax.set_zlabel('Principal Component 3', fontsize = 10)
    ax.set_title('3 component PCA - Diagnosed', fontsize = 15)

    indicesToKeep = finalDF['label'] == 'cd'
    ax.scatter3D(finalDF.loc[indicesToKeep, 'PC1']
            , finalDF.loc[indicesToKeep, 'PC2']
            , finalDF.loc[indicesToKeep, 'PC3']
            , alpha = 0.5
            , c = 'g'
            , s = 50)
    ax.legend(targets)
    plt.savefig("{}/vectors/{}_factor_3_axes_PCA_Diagnosed.png".format(exp_dir, numComps))

    plt.clf()
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.set_xlabel('Principal Component 1', fontsize = 10)
    ax.set_ylabel('Principal Component 2', fontsize = 10)
    ax.set_zlabel('Principal Component 3', fontsize = 10)
    ax.set_title('3 component PCA - Control', fontsize = 15)

    indicesToKeep = finalDF['label'] == 'cc'
    ax.scatter3D(finalDF.loc[indicesToKeep, 'PC1']
            , finalDF.loc[indicesToKeep, 'PC2']
            , finalDF.loc[indicesToKeep, 'PC3']
            , alpha = 0.5
            , c = 'b'
            , s = 50)
    ax.legend(targets)
    plt.savefig("{}/vectors/{}_factor_3_axes_PCA_Control.png".format(exp_dir, numComps))


def plotPCs(finalDF, exp_dir, numComps):

    #Plot the first two PCs in different colors for ironic and non-ironic samples
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    
    targets = ['I', 'N']
    colors = ['g', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDF['label'] == target
        ax.scatter(finalDF.loc[indicesToKeep, 'PC1']
                , finalDF.loc[indicesToKeep, 'PC2']
                , c = color
                , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.savefig("{}/vectors/pca/{}_factor_2_axes_PCA.png".format(exp_dir, numComps))

    plt.clf()
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA - Diagnosed', fontsize = 20)
    
    indicesToKeep = finalDF['label'] == 'cd'
    ax.scatter(finalDF.loc[indicesToKeep, 'PC1']
            , finalDF.loc[indicesToKeep, 'PC2']
            , c = 'g'
            , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.savefig("{}/vectors/pca/{}_factor_2_axes_PCA_Diagnosed.png".format(exp_dir, numComps))

    plt.clf()
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA - Control', fontsize = 20)
    
    indicesToKeep = finalDF['label'] == 'cc'
    ax.scatter(finalDF.loc[indicesToKeep, 'PC1']
            , finalDF.loc[indicesToKeep, 'PC2']
            , c = 'b'
            , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.savefig("{}/vectors/pca/{}_factor_2_axes_PCA_Control.png".format(exp_dir, numComps))


def plotVariance(v, exp_dir, numComps):

    #Plot the percentage of variance explained with the addition of PCs
    plt.ylabel("% Variance Explained")
    plt.xlabel("# of Features")
    plt.title("PCA Analysis")
    plt.ylim(0, 100)
    plt.style.context("seaborn-whitegrid")

    plt.plot(v)
    plt.savefig("{}/vectors/pca/{}_factor_PCA_analysis.png".format(exp_dir, numComps))


def prepAndRun(exp_dir, df, numComps):

    #Data preparation
    features = df.columns[2:].tolist()
    n = df[df["label"] == "cc"] # Makes a separate object for the Control examples
    n = n.loc[:, features].values 

    x = df.loc[:, features].values
    y = df.loc[:, ["label"]].values

    # NOTE I have already accomplished the standard scaling and data imputation by the time the data get here
    ## Scale data and impute missing values
    # scaler = StandardScaler()
    # scaler.fit(n)
    # x = scaler.transform(x)

    # imp_mean = SimpleImputer()
    # x = imp_mean.fit_transform(x) # Just in case you're missing something

    # PCA
    pca = PCA(n_components=numComps)
    principalComponents = pca.fit_transform(x) 

    PCnames = ["PC{}".format(i) for i in range(1, numComps+1)] # Get PC names

    # Data frames with PC values per sample
    principalDF = pd.DataFrame(data = principalComponents, columns = PCnames)
    finalDF = pd.concat([df[['speaker','label']], principalDF], axis = 1)
    finalDF.to_csv("{}/vectors/pca/{}factorPCA.csv".format(exp_dir, numComps), index = False)

    # Compute variance explained by each PC
    variance = pca.explained_variance_ratio_
    var = np.cumsum(np.round(variance, decimals=3)*100)

    print(variance)
    print(var)

    varDF = pd.DataFrame(index=["Proportion of Variance", "Cumulative Proportion"])
    for n, v, c in zip(PCnames, variance, var):
        varDF[n] = [v, c]
    varDF.to_csv("{}/vectors/pca/{}_factor_PCA_variance.csv".format(exp_dir, numComps))

    #Plotting
    plotVariance(var, exp_dir, numComps)
    plotPCs(finalDF, exp_dir, numComps)
    plotPCs3D(finalDF, exp_dir, numComps)

    #Dataframe of  for top features for each PC
    header = pd.MultiIndex.from_product([PCnames,
                                        ['Features', 'Correlation']])
    outDF = pd.DataFrame(columns=header)
    for p, n in zip(pca.components_, PCnames):
        inds = np.argpartition(np.abs(p), -5)[-5:] # NOTE these 5s already have to be the same; gives the top n features for each PC
        top_contribs = p[inds]
        top_feats = np.array(features)[inds]
        outDF[n, 'Features'] = top_feats
        outDF[n, 'Correlation'] = top_contribs
    outDF.to_csv("{}/vectors/pca/{}_factor_PCA_top_feats.csv".format(exp_dir, numComps))

    print(outDF)

    #Plotting
    plotCorrelation(outDF, finalDF, pca, features, exp_dir, numComps)


def getDataFrame(args):

    # Read in the scaled and imputed data
    trainSpeakerDict = joblib.load('{}/vectors/classifiers/{}-trainSpeakerDict.pkl'.format(args.exp_dir, args.scope))
    devSpeakerDict = joblib.load('{}/vectors/classifiers/{}-devSpeakerDict.pkl'.format(args.exp_dir, args.scope))
    
    # Put it in the right format
    myList = list()
    conditionsList = [trainSpeakerDict, devSpeakerDict]
    for i, conditionDict in enumerate(conditionsList):
        for spkr in conditionDict:
            label, data = conditionsList[i][spkr]
            for j in range(data.shape[0]):
                myList.append([spkr, label] + data[j].tolist())

    # Get the names
    names = ['speaker', 'label']
    with open(os.path.join('/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Code/PYTHON/AUDITORY', 'realNames.pkl'), 'rb') as f:
        featureNames = pickle.load(f)

    names += featureNames

    # Don't forget the intelligibility metrics!
    # Each of these dbSNRs is, for its noise type, roughly associated with perceptual intelligibility at 
    # roughly 70%, 50%, and 30% levels.
    levels  = [70, 50, 30]
    for level in levels:
        names += ['DWGP-SMN_{}'.format(level)]
        names += ['DWGP-SSN_{}'.format(level)]
        names += ['SII-SMN_{}'.format(level)]
        names += ['SII-SSN_{}'.format(level)]
        names += ['STI-SMN_{}'.format(level)]
        names += ['STI-SSN_{}'.format(level)]

    df = pd.DataFrame(myList, columns = names)
    return df


def main():

    parser = argparse.ArgumentParser(description='Description of part of pipeline.')
    parser.add_argument('exp_dir', nargs='?', type=str, help='Temporary experiment directory.', default = '/tmp/tmp.OgrVUR6iE4')
    parser.add_argument('scope', nargs = '?', type = str, help = "speaker level global or utterance level local", default = 'auditory-local')

    args = parser.parse_args()
    df = getDataFrame(args)

    # Number of principal components
    numComps = [3, 4, 5, 6] 
    for n in numComps: 
        prepAndRun(args.exp_dir, df, n)


if __name__ == "__main__":

    main()

    # NOTE Helen ran a logistic regression model (using R) on what goes to ../Data; 
    # NOTE I should also be able to run a linear regression model (using R) using the MMSE scores as well. 
