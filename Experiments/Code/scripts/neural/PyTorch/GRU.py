#!/usr/bin/env python3

import os
import glob
import torch
import joblib
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from torch import optim
from matplotlib import pyplot as plt
from feedForwardNet import Feedforward
from sklearn.preprocessing import LabelEncoder


def plotIt(net, exp_dir, title = ""):

    plt.plot(net.loss_arr, 'r-', label='loss')
    plt.plot(net.acc_arr, 'b-', label='train accuracy')
    plt.legend(loc='best')
    plt.title(title)
    plt.xlabel("Epoch")

    plt.savefig(os.path.join(exp_dir, 'reports', 'Feed_Forward_NN_loss.pdf'))


# From a matrix of unrelated vectors to a tensor of utterances
def reFrameData(data, utterances, padding = 0):

    last_seen=utterances[0]
    returnData = []
    tmpData = [data[0]]
    zeroShape = data[0].shape

    for i, v in enumerate(utterances[1:]):
        if v == last_seen:
            tmpData.append(data[i+1])
        else:
            last_seen = v
            if padding > 0:
                # Pad out the tensors right # TODO question: does padding with a '0' mean anything when MFCCs can cross over 0 and sometimes be negative?
                for j in range(padding - len(tmpData)):
                    tmpData.append(np.zeros(zeroShape))
            returnData.append(np.array(tmpData))
            tmpData = [data[i+1]]

    # Get that last one
    if padding > 0:
        # Pad out the tensors right
        for j in range(padding - len(tmpData)):
            tmpData.append(np.zeros(zeroShape))
    returnData.append(np.array(tmpData))

    return returnData


# We need this information for padding
def getLongestSequence(utterances, y_train):

    result=1
    max_result=0
    last_seen=utterances[0]
    new_y_train = [y_train[0]]

    for i, v in enumerate(utterances[1:]):
        if v == last_seen:
            result += 1
        else:
            if result > max_result:
                max_result = result
            last_seen = v
            result = 1
            new_y_train.append(y_train[i+1])

    # just in case the longest sequence would be at the end of the list...
    if result > max_result:
        max_result = result

    return max_result, new_y_train


def getData(exp, data, byFrame, experiment):

    y_train = joblib.load('{}/vectors/classifiers/{}-y_train.pkl'.format(exp, experiment))
    X_train = joblib.load('{}/vectors/classifiers/{}-X_train.pkl'.format(exp, experiment))
    trainUtterances = joblib.load('{}/vectors/classifiers/{}-trainUtterances.pkl'.format(exp, experiment))

    y_test = joblib.load('{}/vectors/classifiers/{}-y_test.pkl'.format(exp, experiment))
    X_test = joblib.load('{}/vectors/classifiers/{}-X_test.pkl'.format(exp, experiment))
    devUtterances = joblib.load('{}/vectors/classifiers/{}-devUtterances.pkl'.format(exp, experiment))
    
    # Encode from categorical to numerical
    le = LabelEncoder()
    le.fit(y_train)

    # Reformat the data to make sense in tensors so we have sequencing information
    trainMax, y_train = getLongestSequence(trainUtterances, y_train)
    X_train = reFrameData(X_train, trainUtterances, trainMax)
    assert len(X_train) == len(y_train)
    _, y_test = getLongestSequence(devUtterances, y_test)
    X_test = reFrameData(X_test, devUtterances)
    assert len(X_test) == len(y_test)

    return torch.tensor(X_train), le.transform(y_train), X_test, le.transform(y_test)


def main(exp_dir, data_dir, random_state, byFrame, experiment, RUNNUM, loocv_not_withheld):

    # Load in the data
    X, y, X_test, y_test = getData(exp_dir, data_dir, byFrame, experiment)

    # Convert the data into tensors
    X_train, Y_train, X_val, Y_val = map(torch.from_numpy, (X, y, X_test, y_test)) # TODO start here
    X_train = X_train.float()
    Y_train = Y_train.float()
    X_val = X_val.float()
    Y_val = Y_val.float()

    # Initialize the model
    net = Feedforward(X_train.shape[-1], os.path.join(exp_dir, "nn_checkpoints"))   # X_train, Y_train, os.path.join(exp_dir, "nn_checkpoints"))

    # Evaluate the model to get a baseline
    net.eval()
    y_pred = net(X_val)
    before_train = net.criterion(y_pred.squeeze(), Y_val)
    print('Test loss before training', before_train.item())

    # Train up the model
    net.train()
    epochs = 10000
    for epoch in range(epochs):

        net.optimizer.zero_grad()

        # Forward pass
        y_pred = net(X_train)

        # Compute and saveloss
        loss = net.criterion(y_pred.squeeze(), Y_train)
        net.loss_arr.append(loss.item())

        # Compute and save accuracy
        net.acc_arr.append(net.accuracy(y_pred, Y_train))

        print('Epoch {}: train loss: {}\t train accuracy: {}'.format(epoch, loss.item(), net.acc_arr[-1]))

        # Backward pass
        loss.backward()
        net.optimizer.step()

        if epoch % 100 == 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': net.optimizer.state_dict(),
                        'loss': loss
                        }, os.path.join(net.ckpt_dest, "feedForwardNN_{}".format(epoch))
                       )

    # Evaluate the model
    net.eval()
    y_pred = net(X_val)
    after_train = net.criterion(y_pred.squeeze(), Y_val)

    print('Test loss after Training', after_train.item())

    # Visualize the model
    # plotIt(net, exp_dir, "Feed Forward Neural Network")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Call all of the regression algorithms and consolidate a global report.')
    parser.add_argument('exp_dir', nargs='?', type=str, help='Temporary experiment directory.', default='/tmp/tmp.withheldMFCC')
    parser.add_argument('data_dir', nargs='?', type=str, help='Location for the map for the scalar score (e.g. MMSE).', default='../Data/ADReSS-IS2020-data/train')
    parser.add_argument('random_state', nargs='?', type=int, help='Affects the ordering of the indices, which controls the randomness of each fold in KFold validation.', default=1)
    parser.add_argument('by_frame', nargs='?', type=str, help='True if we need to run this by frame or False if we are using COMPARE or something else distilled.', default="True")
    parser.add_argument('run_num', nargs='?', type=str, help='Which runthrough we are on.', default='1897')
    parser.add_argument('loocv_not_withheld', nargs='?', type=str, help='If True, we will do 5 fold leave-one-out cross validation; if False, we are training on all the training data and testing on the withheld test data ', default='False')

    args = parser.parse_args()

    # This distinction may not actually matter for the neural networking implementation
    if args.by_frame == "True":
        for experiment in ['raw']: #['raw', 'averaged', 'flattened']: #, 'averaged_and_flattened']:
            print('Now working on {}'.format(experiment))
            main(args.exp_dir, args.data_dir, args.random_state, True, experiment, args.run_num, args.loocv_not_withheld)
    else:
        main(args.exp_dir, args.data_dir, args.random_state, False, "compare", args.run_num, args.loocv_not_withheld)
