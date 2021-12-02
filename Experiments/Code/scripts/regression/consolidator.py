#!/usr/bin/env python3

import numpy as np

class Consolidator:
    "Get cross-validated means across regression runs."

    def __init__(self, regressors):

        self.regressors = regressors
        self.name = self.regressors[0].name

    def _get_means(self):

        allValues = list()
        for rgr in self.regressors:
            values, labels = rgr._make_report()
            allValues.append(values[1:])

        means = np.mean(allValues, axis = 0).tolist() 

        return labels, [self.name] + means

