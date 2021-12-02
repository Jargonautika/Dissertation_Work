#!/usr/bin/env python3

from classificationValidation import ClassificationValidator
from sklearn.svm import SVC


def validate(clf, X_test, y_true, labelClasses, speakerDict):

    return ClassificationValidator("Linear Support Vector", clf, X_test, y_true, labelClasses, speakerDict)


def model(X_train, y_train):

    print('start train')
    model = SVC(kernel = 'linear', C=0.025).fit(X_train, y_train)
    print('end train')
    return model


def main(X_train, y_train, X_test, y_test, labelClasses, speakerDict):

    # Construct the model
    clf = model(X_train, y_train)

    # Validate the model
    scores = validate(clf, X_test, y_test, labelClasses, speakerDict)

    return scores, clf

