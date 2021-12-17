#!/usr/bin/env python3

import torch
torch.manual_seed(42)

CUDA_LAUNCH_BLOCKING = "1"

import os
import glob
import torch
import joblib
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from GRU import GRU
from torch import optim
from torch.optim import Adam
from matplotlib import pyplot as plt
from feedForwardNet import Feedforward
from dataLoader import customDataLoader
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder


def plotIt(net, exp_dir, title = ""):

    plt.plot(net.loss_arr, 'r-', label='loss')
    plt.plot(net.acc_arr, 'b-', label='train accuracy')
    plt.legend(loc='best')
    plt.title(title)
    plt.xlabel("Epoch")

    plt.savefig(os.path.join(exp_dir, 'reports', 'Gated_Recurrent_NN_loss.pdf'))


# From a matrix of unrelated vectors to a tensor of utterances
def reFrameData(data, utterances, padding = 0):

    last_seen=utterances[0]
    returnData = []
    tmpData = [data[0]]
    fillShape = data[0].shape

    for i, v in enumerate(utterances[1:]):
        if v == last_seen:
            tmpData.append(data[i+1])
        else:
            last_seen = v
            if padding > 0:
                # Pad out the tensors right
                for j in range(padding - len(tmpData)):
                    tmpData.append(np.full(fillShape, 0.0)) # 1000 here means "not a real cepstral coefficient; please ignore"
            returnData.append(np.array(tmpData))
            tmpData = [data[i+1]]

    # Get that last one
    if padding > 0:
        # Pad out the tensors right
        for j in range(padding - len(tmpData)):
            tmpData.append(np.full(fillShape, 0.0))
    returnData.append(np.array(tmpData))

    return returnData, fillShape


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
    X_train, fillShape = reFrameData(X_train, trainUtterances, trainMax)
    assert len(X_train) == len(y_train)
    _, y_test = getLongestSequence(devUtterances, y_test)
    X_test, _ = reFrameData(X_test, devUtterances)
    assert len(X_test) == len(y_test)

    # Convert the data into tensors
    W = torch.tensor(X_train, dtype = torch.double)
    x = le.transform(y_train)
    Y = [torch.tensor(x, dtype = torch.double) for x in X_test]
    z = le.transform(y_test)
    x, z = map(torch.from_numpy, (x, z))
    x = x.float()
    z = z.float()

    # Get the final indexes with actual values in them
    last_idxes = ((W != 0.0).sum(dim = 2).sum(dim=1)/fillShape[0] - 1).int()

    # Put the data on the GPU
    W = W.to('cuda:0')
    x = x.to('cuda:0')
    Y = [y.to('cuda:0') for y in Y]
    z = z.to('cuda:0')
    last_idxes = last_idxes.to('cuda:0')

    trainData = DataLoader(dataset = customDataLoader(W, x, last_idxes), batch_size = 50)
    testData = DataLoader(dataset = customDataLoader(Y, z), batch_size = 1)

    return trainData, testData, fillShape


def main(exp_dir, data_dir, random_state, byFrame, experiment, RUNNUM, loocv_not_withheld):

    # Load in the data
    X_trainData, X_testData, fillShape = getData(exp_dir, data_dir, byFrame, experiment)

    # Initialize the model
    ckpt_dest = os.path.join(exp_dir, 'nn_checkpoints')
    net = GRU(feat_size=fillShape[0], embed_size=256, hidden_size=512, dropout=0.3, bidirectional=True, num_classes=2, ckpt_dest = ckpt_dest)
    net.to('cuda:0')

    #Hyperparameters
    epochs = 500
    learning_rate = 0.001
    optimizer = Adam(net.parameters(), lr=learning_rate)

    # # Do a single pass for debugging
    # j = len(X_trainData)
    # total_loss = 0.0
    # for i, batch in enumerate(X_trainData):
    #     print(i, i/j)
    #     examples = batch["data"]
    #     oneHotLabels = batch["oneHotLabels"]
    #     idxes = batch["indices"]
    #     # Pass prediction through a softmax layer
    #     prediction = torch.softmax(net(examples, idxes), 0)
    #     loss = net.criterion(prediction, oneHotLabels)
    #     total_loss += loss.item()
    #     loss.backward()
    #     optimizer.step()

    # # Evaluate the model first to get a baseline
    # y_pred = net.getPredictions(X_testData)
    # before_train = net.accuracy(y_pred, X_testData.dataset.labels)
    # del y_pred
    # print('Test loss before training', total_loss)
    # print('Test accuracy before training', before_train)

    # Train up the model
    for epoch in range(epochs):

        total_accuracy = list()
        optimizer.zero_grad()
        total_loss = 0.0

        # Forward pass in batches for memory's sake
        for j, batch in enumerate(X_trainData):
            examples = batch["data"]
            labels = batch["labels"]
            oneHotLabels = batch["oneHotLabels"]
            idxes = batch["indices"]
            prediction = net(examples, idxes)
            prediction = prediction.squeeze(0)
            probs = torch.softmax(prediction, 0)
            probs = prediction.clamp(0, 1) # https://stackoverflow.com/questions/66456541/runtimeerror-cuda-error-device-side-assert-triggered-on-loss-function
            prediction = torch.argmax(probs, dim = 1)

            # Compute loss
            loss = net.criterion(probs, oneHotLabels)

            # For checking the loss by epoch instead of by batch
            total_loss += loss.item()
            total_accuracy.append(net.accuracy(prediction, labels).item())

            # Backward pass
            loss.backward()
            optimizer.step()

        # Compute and save accuracy
        net.acc_arr.append(np.mean(total_accuracy))
        net.loss_arr.append(total_loss)

        if epoch % 5 == 0:
            print('Epoch {}: train loss: {}\t train accuracy: {}'.format(epoch, loss.item(), net.acc_arr[-1]))
            torch.save({'epoch'                 : epoch,
                        'model_state_dict'      : net.state_dict(),
                        'optimizer_state_dict'  : optimizer.state_dict(),
                        'loss'                  : loss,
                        'accuracy'              : net.acc_arr[-1]
                        }, os.path.join(net.ckpt_dest, "GRU_Epoch_{}".format(epoch))
                       )    
    # Evaluate the model
    y_pred = net.getPredictions(X_testData)
    after_train = net.accuracy(y_pred, X_testData.dataset.labels)

    print('Test loss after Training', after_train)

    # Visualize the model
    plotIt(net, exp_dir, "Gated Recurrent Network")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Call all of the regression algorithms and consolidate a global report.')
    parser.add_argument('exp_dir', nargs='?', type=str, help='Temporary experiment directory.', default='/tmp/tmp.withheldMFCC')
    parser.add_argument('data_dir', nargs='?', type=str, help='Location for the map for the scalar score (e.g. MMSE).', default='../Data/ADReSS-IS2020-data/train')
    parser.add_argument('random_state', nargs='?', type=int, help='Affects the ordering of the indices, which controls the randomness of each fold in KFold validation.', default=1)
    parser.add_argument('by_frame', nargs='?', type=str, help='True if we need to run this by frame or False if we are using COMPARE or something else distilled.', default="True")
    parser.add_argument('run_num', nargs='?', type=str, help='Which runthrough we are on.', default='1897')
    parser.add_argument('loocv_not_withheld', nargs='?', type=str, help='If True, we will do 5 fold leave-one-out cross validation; if False, we are training on all the training data and testing on the withheld test data ', default='False')

    args = parser.parse_args()
    assert torch.cuda.is_available(), "Cuda isn't online; figure that out to run this script."

    # This distinction may not actually matter for the neural networking implementation
    if args.by_frame == "True":
        for experiment in ['raw']: #['raw', 'averaged', 'flattened']: #, 'averaged_and_flattened']:
            # print('Now working on {}'.format(experiment))
            main(args.exp_dir, args.data_dir, args.random_state, True, experiment, args.run_num, args.loocv_not_withheld)
    else:
        main(args.exp_dir, args.data_dir, args.random_state, False, "compare", args.run_num, args.loocv_not_withheld)
