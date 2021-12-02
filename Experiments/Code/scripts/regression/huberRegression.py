#!/usr/bin/env python3

from regressionValidation import RegressionValidator
from sklearn.linear_model import HuberRegressor
from sklearn.exceptions import ConvergenceWarning


def validate(reg, X_test, y_true, speakerDict, byFrame):

    return RegressionValidator("Huber", reg, X_test, y_true, speakerDict, byFrame)


def model(X_train, y_train):

    return HuberRegressor().fit(X_train, y_train)


def main(X_train, y_train, X_test, y_test, speakerDict, byFrame):

    # Construct the model
    reg = model(X_train, y_train)

    # Validate the model
    scores = validate(reg, X_test, y_test, speakerDict, byFrame)

    return scores, reg

