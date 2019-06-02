import constructKernel
import numpy as np


def MAEDRanking(candidates, selectNum, options):
    ReguAlpha = 0.01
    fea = candidates.X
    K = constructKernel.constructKernel(fea,[],{})
    splitCandi = np.ones((K.shape[1],1), dtype=int)
    smpRank = np.zeros((selectNum,1), dtype=int)

    for x in range(selectNum):

        splitCandiInd = np.array(indices(splitCandi, lambda x: x == 1))
        numerD = np.sum(np.power(K[:,splitCandiInd],2),0)
        denomD = (np.transpose(np.diag(K[splitCandiInd,:][:,splitCandiInd]))+ReguAlpha)
        DValue = np.divide(numerD, denomD)
        
        idx = np.argmax(DValue)
        CandiIdx = indices(splitCandi, lambda x: x != 0)
        smpRank[x] = CandiIdx[idx]
        splitCandi[CandiIdx[idx]] = False
            
        numerK = np.matmul(np.transpose([K[:,CandiIdx[idx]]]), [K[CandiIdx[idx],:]])
        denomK = K[CandiIdx[idx],CandiIdx[idx]]+ReguAlpha
        K = np.subtract(K, np.divide(numerK, denomK))

    return smpRank

def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

