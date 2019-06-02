# function [model] = incrementalLearn(data, labels, options)
# % Incremental Learning
# %
# % Input:
# %   data               - Data matrix NXM, where N is the number of data
# %                          points and M is then number of variables
# %   labels             - Label/response for each data point (Nx1)
# %   options            - Struct value: 
# %       modelSize (default: 25)  - number of samples in the model
# %       batchSize (default: 100) - number of samples in each training batch
# %       nbTrainingSamples  (default: N)  - total bunber of training samples 
# %
# % Output:
# %   model              - selected data points 
# %       model.X        - input features
# %       model.Y        - labels/responses
# %

# if(isfield(options,'batchSize'))
#     batchSize = options.batchSize;
# else
#     batchSize = 100;
# end

# if(isfield(options,'modelSize'))
#     modelSize = options.modelSize;
# else
#     modelSize = 25;
# end

# if(isfield(options,'nbTrainingSamples'))
#     nbTrainingSamples = options.nbTrainingSamples;
# else
#     nbTrainingSamples = size(data,1);
# end

# model.X = [];
# model.Y = [];

# for j=1:batchSize:nbTrainingSamples-batchSize
    
#     candidates.X = [model.X; data(j:j+batchSize-1,:)];
#     candidates.Y = [model.Y; labels(j:j+batchSize-1,:)];
    
#     rank = MAEDRanking(candidates, modelSize, options);    
    
#     model.X = candidates.X(rank,:);
#     model.Y = candidates.Y(rank,:);
# end

import numpy as np
import MAEDRanking

class XYArray:
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y


def incrementalLearn(data, labels, options):
    batchSize = 100
    modelSize = 25
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

        # print(np.transpose(rank+1))

        model.X = np.array([candidates.X[r[0]] for r in rank])

        # print(model.X[0][0])
        model.Y = np.array([candidates.Y[r[0]] for r in rank])


    return model






