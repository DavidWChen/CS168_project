# function [smpRank] = MAEDRanking(candidates, selectNum, options)
# %Reference:
# %
# %   [1] Deng Cai and Xiaofei He, "Manifold Adaptive Experimental Design for
# %   Text Categorization", IEEE Transactions on Knowledge and Data
# %   Engineering, vol. 24, no. 4, pp. 707-719, 2012.
# %


# fea = candidates.X;
# labels = candidates.Y;

# nSmp = size(fea,1);

# if(~isfield(options,'ReguBeta'))
#     options.ReguBeta = .1;
# end

# if(~isfield(options,'bLDA'))
#     options.bLDA = 0;
# end

# if(~isfield(options,'ReguAlpha'))
#     options.ReguAlpha = 0.01;
# end

# [K,Dist,options] = constructKernel(fea,[],options);

# options.gnd = labels;

# if isfield(options,'ReguBeta') && options.ReguBeta > 0
#     if isfield(options,'W')
#         W = options.W;
#     else
#         if isfield(options,'k')
#             Woptions.k = options.k;
#         else
#             Woptions.k = 5;
#         end
        
#         tmpD = Dist;
#         Woptions.t = mean(mean(tmpD));
#         if isfield(options,'gnd')
#             Woptions.WeightMode = 'HeatKernel';
#             Woptions.NeighborMode='Supervised';
#             Woptions.bLDA=options.bLDA;
#             Woptions.gnd=options.gnd;
#         end
#         W = constructW(fea,Woptions);
#     end
#     D = full(sum(W,2));
#     L = spdiags(D,0,nSmp,nSmp)-W;
#     K = (speye(size(K,1))+options.ReguBeta*K*L)\K;
#     K = max(K,K');
# end

# splitCandi = true(size(K,2),1);
# smpRank = zeros(selectNum,1);

# for sel = 1:selectNum
#     DValue = sum(K(:,splitCandi).^2,1)./(diag(K(splitCandi,splitCandi))'+options.ReguAlpha);
#     [~,idx] = max(DValue);
#     CandiIdx = find(splitCandi);
#     smpRank(sel) = CandiIdx(idx);
#     splitCandi(CandiIdx(idx)) = false;
#     K = K - (K(:,CandiIdx(idx))*K(CandiIdx(idx),:))/(K(CandiIdx(idx),CandiIdx(idx))+options.ReguAlpha);
# end

# end
import constructKernel
import numpy as np
def MAEDRanking(candidates, selectNum, options):

    ReguAlpha = 0.01
    
    fea = candidates.X

    K = constructKernel.constructKernel(fea,[],{})



    splitCandi = np.ones((K.shape[1],1), dtype=int)
    smpRank = np.zeros((selectNum,1), dtype=int)




    for x in range(selectNum):


        # print('blep')
        splitCandiInd = np.array(indices(splitCandi, lambda x: x == 1))
        

        # print(splitCandiInd.shape)

        part1 = np.sum(np.power(K[:,splitCandiInd],2),0)

        part0 = K[splitCandiInd,:][:,splitCandiInd]
        
        part2 = (np.transpose(np.diag(part0))+ReguAlpha)


        # print(part1.shape)
        # print(part2.shape)
        

        DValue = np.divide(part1,part2)
        
        idx = np.argmax(DValue)
        
        


        CandiIdx = indices(splitCandi, lambda x: x != 0)


        smpRank[x] = CandiIdx[idx]##


        splitCandi[CandiIdx[idx]] = False
            
        partZ=np.transpose([K[:,CandiIdx[idx]]])
        partY=[K[CandiIdx[idx],:]]
        # print(partZ)
        # print(partY)
        partA = np.matmul(partZ, partY)
        # print(partA)

        partB = K[CandiIdx[idx],CandiIdx[idx]]+ReguAlpha
        # print(partB)

        partC = np.divide(partA, partB)

        # print(partC)
            


        # #imitate matlab right divide
        # partB = np.linalg.pinv(K[CandiIdx[idx],CandiIdx[idx]])


        # partC = np.matmul(partA, partB)

        K = np.subtract(K, partC)
        # print(K)






    return smpRank




def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]












