#!/usr/bin/env python3

import numpy as np

class Consolidator:
    "Get cross-validated means across classification runs."

    def __init__(self, classifiers):

        self.classifiers = classifiers
        self.name = self.classifiers[0].name

    def _get_means(self):

        allValues = list()
        for clf in self.classifiers:
            labels, values = clf._make_report()
            allValues.append(values[1:])

        means = np.mean(allValues, axis = 0).tolist()

        return labels, [self.name] + means

