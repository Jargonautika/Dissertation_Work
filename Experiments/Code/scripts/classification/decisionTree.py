#!/usr/bin/env python3

from classificationValidation import ClassificationValidator
from sklearn.tree import DecisionTreeClassifier


def validate(clf, X_test, y_true, labelClasses, speakerDict):

    return ClassificationValidator("Decision Tree", clf, X_test, y_true, labelClasses, speakerDict)


def model(X_train, y_train):

    return DecisionTreeClassifier(max_depth = 5).fit(X_train, y_train)


def main(X_train, y_train, X_test, y_test, labelClasses, speakerDict):

    # Construct the model
    clf = model(X_train, y_train)

    # Validate the model
    scores = validate(clf, X_test, y_test, labelClasses, speakerDict)

    return scores, clf

