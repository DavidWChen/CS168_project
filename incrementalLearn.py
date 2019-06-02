import numpy as np
import MAEDRanking


class XYArray:
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y


def incrementalLearn(data, labels, options):
    batchSize = 500# 100
    modelSize = 100 #25
    nbTrainingSamples = len(data)

    model = XYArray(None,None)
    candidates = XYArray(None,None)

    for i in range(0, nbTrainingSamples, batchSize):

        if model.X is None:
            candidates.X = data[i:i+batchSize]
        else:
            candidates.X = np.concatenate((model.X, data[i:i+batchSize]), axis=0)

        if model.Y is None:
            candidates.Y = labels[i:i+batchSize]
        else:
            candidates.Y = np.concatenate((model.Y, labels[i:i+batchSize]), axis=0)

        rank = MAEDRanking.MAEDRanking(candidates, modelSize, {})

        model.X = np.array([candidates.X[r[0]] for r in rank])
        model.Y = np.array([candidates.Y[r[0]] for r in rank])

    return model
