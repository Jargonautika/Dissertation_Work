#!/usr/bin/env python3

from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score, mean_poisson_deviance, mean_gamma_deviance
import numpy as np

class RegressionValidator:
    "Get scores for various mathematical operations related to regression models."

    def __init__(self, name, regressor, X_test, y_true, speakerDict, byFrame = True):

        self.name = name
        self.reg = regressor
        self.X = X_test
        self.y = y_true
        self.pred = self.reg.predict(self.X)
        self.speakerDict = speakerDict
        self.byFrame = byFrame
        
        self.ev = explained_variance_score(self.y, self.pred)
        self.me = max_error(self.y, self.pred)
        self.mae = mean_absolute_error(self.y, self.pred)
        self.mse = mean_squared_error(self.y, self.pred)
        self.rmse = mean_squared_error(self.y, self.pred, squared = False)
        # self.msle = mean_squared_log_error(self.y, self.pred) # ValueError: Mean Squared Logarithmic Error cannot be used when targets contain negative values.
        self.medae = median_absolute_error(self.y, self.pred)
        self.r2 = r2_score(self.y, self.pred)
        try:
            self.mpd = mean_poisson_deviance(self.y, self.pred) # ValueError: Mean Tweedie deviance error with power=1 can only be used on non-negative y and strictly positive y_pred. (Huber issue with CoMpArE)
        except:
            self.mpd = np.nan
        try:
            self.mgd = mean_gamma_deviance(self.y, self.pred) # ValueError: Mean Tweedie deviance error with power=2 can only be used on strictly positive y and y_pred.
        except:
            self.mgd = np.nan
        if self.byFrame:
            self.howClose = list(self._winnerTakeAll())
            self.closenessAverage = sum(self.howClose)/len(self.howClose)
        else:
            self.closenessAverage = np.nan


    def _winnerTakeAll(self):

        for speaker in self.speakerDict:
            pred = self.reg.predict(self.speakerDict[speaker][1]).tolist()
            avg = int(sum(pred)/len(pred))
            delta = abs(avg - self.speakerDict[speaker][0])
            yield delta/30 # MMSE scores run on a scale of 0-30, so this accounts for how close we guessed right


    def _make_report(self):
       
        return [self.name, self.ev, self.me, self.mae, self.mse, self.rmse, self.medae, self.r2, self.mpd, self.mgd, self.closenessAverage], ['Algorithm', 'Explained Variance', 'Maximum Error', 'Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error', 'Median Absolute Error', 'R2 Score', 'Mean Poisson Deviance', 'Mean Gamma Deviance', 'Winner Takes All']

