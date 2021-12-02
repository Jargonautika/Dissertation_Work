#!/usr/bin/env python3

from sklearn.metrics import accuracy_score, f1_score, fbeta_score, hamming_loss, jaccard_score, log_loss, precision_score, recall_score, zero_one_loss, confusion_matrix

class ClassificationValidator:
    "Get scores for various mathematical operations related to classification models."

    def __init__(self, name, classifier, X_test, y_true, labelClasses, speakerDict):

        self.name = name
        self.clf = classifier
        self.X = X_test
        self.y = y_true
        self.target_names = labelClasses
        self.pred = self.clf.predict(self.X)
        self.speakerDict = speakerDict

        self.accuracy = accuracy_score(self.y, self.pred)
        self.f1 = f1_score(self.y, self.pred)
        self.fbeta = fbeta_score(self.y, self.pred, average = 'weighted', beta = 0.5)
        self.hamming = hamming_loss(self.y, self.pred)
        self.jaccard = jaccard_score(self.y, self.pred)
        self.crossEntropyLoss = log_loss(self.y, self.pred)
        self.precision = precision_score(self.y, self.pred)
        self.recall = recall_score(self.y, self.pred)
        self.zeroOneLoss = zero_one_loss(self.y, self.pred)
        self.winLoss = self._winnerTakeAll()
        self.winPercentage = self.winLoss.count(True)/len(self.winLoss)
        self.confusionMatrix = confusion_matrix(self.y, self.pred, labels=self.clf.classes_)


    def _winnerTakeAll(self):

        winnerWinnerChickenDinner = list()
        for speaker in self.speakerDict:
            pred = self.clf.predict(self.speakerDict[speaker][1]).tolist()
            winner = max(set(pred), key=pred.count)
            if [winner] == self.speakerDict[speaker][0]:
                winnerWinnerChickenDinner.append(True)
            else:
                winnerWinnerChickenDinner.append(False) 
        return winnerWinnerChickenDinner


    def _make_report(self):

        return ['Classifier', 'F Beta Score', 'Hamming Loss', 'Jaccard Score', 'Cross Entropy Loss', 'Zero One Loss', 'Precision', 'Recall', 'Accuracy', 'F Measure', 'Winner Takes All'], [self.name, self.fbeta, self.hamming, self.jaccard, self.crossEntropyLoss, self.zeroOneLoss, self.precision, self.recall, self.accuracy, self.f1, self.winPercentage]
