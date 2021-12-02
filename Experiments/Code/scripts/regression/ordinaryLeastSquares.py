#!/usr/bin/env python3

from regressionValidation import RegressionValidator
from sklearn.linear_model import LinearRegression


def validate(reg, X_test, y_true, speakerDict, byFrame):

    return RegressionValidator("Ordinary Least Squares", reg, X_test, y_true, speakerDict, byFrame)


def model(X_train, y_train):

    return LinearRegression().fit(X_train, y_train)


def main(X_train, y_train, X_test, y_test, speakerDict, byFrame):

    # Construct the model
    reg = model(X_train, y_train)

    # Validate the model
    scores = validate(reg, X_test, y_test, speakerDict, byFrame)

    return scores, reg

