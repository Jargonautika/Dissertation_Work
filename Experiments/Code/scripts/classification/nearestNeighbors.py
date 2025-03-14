#!/usr/bin/env python3

from classificationValidation import ClassificationValidator
from sklearn.neighbors import KNeighborsClassifier


def validate(clf, X_test, y_true, labelClasses, speakerDict):

    return ClassificationValidator("K Nearest Neighbors", clf, X_test, y_true, labelClasses, speakerDict)


def model(X_train, y_train):

    return KNeighborsClassifier(3).fit(X_train, y_train)


def main(X_train, y_train, X_test, y_test, labelClasses, speakerDict):

    # Construct the model
    clf = model(X_train, y_train)

    # Validate the model
    scores = validate(clf, X_test, y_test, labelClasses, speakerDict)

    return scores, clf

